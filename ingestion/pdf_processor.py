"""
PDF Processing Pipeline:

Step 0: Detect browser-generated PDFs and use sort=True for correct reading order.

Get text from PDF:
1: Extract all pages as (raw_text, page_number) tuples.
    1a: OCR fallback for scanned/image-based PDFs (if enabled).

Normalize text:
2: Strip trailing page numbers from each page's raw text.
3: Header/footer dedup (on raw text before normalization).
4: Normalize each page text.
5: Concatenate all pages, tracking character offsets per page.

Chunk text:
6: Chunk the full document: 
    - Semantic chunking 
    - Fixed chunking
7: Map each raw chunk to page number(s) via offset ranges.
8: Apply overlap (only for fixed chunking).

Return chunks:
9: Build result — collapse newlines in non-table text.
10: Return the chunks.
"""

import logging
import re
from collections import Counter
from typing import List, Dict, Tuple, Any
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# Structural patterns that strongly indicate code/technical content.
# Intentionally avoids prose keywords (return, if) that appear in normal sentences.
_CODE_HINTS = re.compile(
    r"[{};]"            # braces and semicolons
    r"|->|=>"           # pointer dereference, fat arrow
    r"|0x[0-9a-fA-F]"  # hex literals
    r"|::\w"            # C++ scope resolution
    r"|\b__\w+"         # __attribute__, __int64, __builtin_*
)

# A line likely ends mid-sentence (soft-wrapped prose) if it ends with a
# lowercase letter or comma — not with closing punctuation or code tokens.
_SOFT_WRAP_END = re.compile(r"[a-z,]$")


def _looks_like_code(raw_line: str, stripped_line: str) -> bool:
    """Return True if the line is likely code or technical content."""
    # Indented lines are almost always code or structured output
    if raw_line != raw_line.lstrip():
        return True
    if _CODE_HINTS.search(stripped_line):
        return True
    # Short lines ending with code-boundary punctuation
    if len(stripped_line) < 80 and stripped_line.endswith(("(", "{", ":")):
        return True
    return False


def _normalize_pdf_text(text: str) -> str:
    """
    Normalize text extracted from PDFs:
    - Normalize line endings
    - Convert bullet-like unicode chars to a common form
    - Merge hard-wrapped lines inside paragraphs (prose only)
    - Preserve code blocks, indented lines, and technical content
    - Preserve paragraph breaks
    """
    if not text:
        return ""

    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Common bullet chars -> "-"
    text = text.replace("\u2022", "-").replace("\u25cf", "-").replace("\u2219", "-")

    # Trim trailing spaces on lines
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Collapse 3+ newlines to 2 (keep paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Merge soft-wrapped prose lines. Only merge when:
    #   - neither line looks like code/technical content
    #   - the current line ends mid-sentence (lowercase letter or comma)
    # This preserves code blocks, assembly, hex dumps, and structured output.
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.strip()

        if line == "":
            out.append("")  # paragraph break marker
            i += 1
            continue

        if i + 1 < len(lines):
            raw_nxt = lines[i + 1]
            nxt = raw_nxt.strip()

            is_list = bool(re.match(r"^(\-|\*|\u2022|\d+[\.\)]|[a-zA-Z][\.\)])\s+", line))
            next_is_list = bool(re.match(r"^(\-|\*|\u2022|\d+[\.\)]|[a-zA-Z][\.\)])\s+", nxt))

            if nxt != "" and not is_list and not next_is_list:
                # Hyphenated word break: "exam-\nple" -> "example"
                if line.endswith("-") and nxt and nxt[0].islower():
                    out.append(line[:-1] + nxt)
                    i += 2
                    continue
                # Merge only if both lines look like prose and current ends mid-sentence
                if (not _looks_like_code(raw_line, line)
                        and not _looks_like_code(raw_nxt, nxt)
                        and bool(_SOFT_WRAP_END.search(line))):
                    out.append(line + " " + nxt)
                    i += 2
                    continue

        out.append(line)
        i += 1

    # Rebuild with paragraphs
    normalized = "\n".join(out)
    normalized = re.sub(r"\n\s*\n", "\n\n", normalized)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized).strip()
    return normalized


def _split_sentences(text: str) -> List[str]:
    """
    A practical sentence splitter (regex-based). Hardened for academic text.
    """
    # Protect common abbreviations to reduce bad splits
    protected = [
        # General
        "e.g.", "i.e.", "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "vs.", "etc.",
        "Fig.", "Eq.", "No.", "St.", "Inc.", "Ltd.", "Sr.", "Jr.", "Corp.", "Assn.",
        # Academic
        "et al.", "al.", "Sec.", "Vol.", "Rev.", "Ref.", "Ch.", "App.", "Dept.",
        # Units
        "ft.", "in.", "oz.",
        # Months
        "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec.",
        # Days
        "Mon.", "Tue.", "Wed.", "Thu.", "Fri.", "Sat.", "Sun.",
    ]
    placeholder = "§§§"
    for ab in protected:
        text = text.replace(ab, ab.replace(".", placeholder))

    # Protect decimal numbers (e.g. 3.14, 0.05)
    text = re.sub(r"(\d)\.(\d)", r"\1" + placeholder + r"\2", text)

    # Protect single-letter abbreviations / initials (e.g. A., J. K.)
    text = re.sub(r"\b([A-Z])\.(?=\s|$)", r"\1" + placeholder, text)

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    parts = [p.replace(placeholder, ".").strip() for p in parts if p.strip()]
    return parts if parts else [text.strip()]


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int = 500) -> List[str]:
    """
    Chunk text with preference for paragraph boundaries, then sentence, then word.
    Chunks are sized to leave room for overlap (content_budget = chunk_size - chunk_overlap).
    Overlap is applied separately by _apply_overlap.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    text = text.strip()
    if not text:
        return []

    content_budget = chunk_size - chunk_overlap

    # Paragraph-like blocks
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

    chunks: List[str] = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for block in blocks:
        # If block fits, add it
        if len(block) <= content_budget:
            if not current:
                current = block
            elif len(current) + 2 + len(block) <= content_budget:
                current = current + "\n\n" + block
            else:
                flush()
                current = block
            continue

        # Block too large: flush current, then split block by sentences
        flush()
        sentences = _split_sentences(block)

        # Build chunk(s) from sentences
        buf = ""
        for s in sentences:
            if len(s) > content_budget:
                # Sentence too large: split by words
                if buf.strip():
                    chunks.append(buf.strip())
                    buf = ""
                words = s.split()
                wbuf = ""
                for w in words:
                    if not wbuf:
                        wbuf = w
                    elif len(wbuf) + 1 + len(w) <= content_budget:
                        wbuf = wbuf + " " + w
                    else:
                        chunks.append(wbuf.strip())
                        wbuf = w
                if wbuf.strip():
                    chunks.append(wbuf.strip())
                continue

            if not buf:
                buf = s
            elif len(buf) + 1 + len(s) <= content_budget:
                buf = buf + " " + s
            else:
                chunks.append(buf.strip())
                buf = s

        if buf.strip():
            chunks.append(buf.strip())

    flush()
    return chunks


def _apply_overlap(chunks: List[str], chunk_overlap: int, chunk_size: int) -> List[str]:
    """
    Prepend trailing sentences from the previous chunk as overlap context.
    Hard-clamps result to chunk_size at a word boundary.
    """
    if chunk_overlap <= 0 or len(chunks) <= 1:
        return list(chunks)

    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_sentences = _split_sentences(chunks[i - 1])
        overlap_sentences = []
        total_len = 0

        for sentence in reversed(prev_sentences):
            sentence_len = len(sentence) + 1
            if total_len + sentence_len <= chunk_overlap:
                overlap_sentences.insert(0, sentence)
                total_len += sentence_len
            else:
                break

        if overlap_sentences:
            combined = " ".join(overlap_sentences) + " " + chunks[i]
        else:
            combined = chunks[i]

        # Hard-clamp to chunk_size at word boundary
        if len(combined) > chunk_size:
            truncated = combined[:chunk_size]
            # Find last space to avoid cutting mid-word
            last_space = truncated.rfind(" ")
            if last_space > chunk_size // 2:
                combined = truncated[:last_space]
            else:
                combined = truncated

        overlapped.append(combined)

    return overlapped


def _detect_and_remove_headers_footers(
    pages: List[Tuple[str, int]], threshold: float = 0.5, max_line_len: int = 120
) -> List[Tuple[str, int]]:
    """
    Detect and remove repeated header/footer lines across pages.
    Only considers first 3 and last 3 lines of each page.
    Lines appearing on >threshold fraction of pages are removed.
    Skips docs with <3 pages.
    """
    if len(pages) < 3:
        return pages

    num_pages = len(pages)
    candidate_counter: Counter = Counter()

    for page_text, _ in pages:
        lines = page_text.strip().split("\n")
        # First 3 and last 3 lines (may overlap for short pages, use set)
        candidate_lines = set()
        for line in lines[:3]:
            stripped = line.strip()
            if stripped and len(stripped) <= max_line_len:
                candidate_lines.add(stripped)
        for line in lines[-3:]:
            stripped = line.strip()
            if stripped and len(stripped) <= max_line_len:
                candidate_lines.add(stripped)
        for line in candidate_lines:
            candidate_counter[line] += 1

    # Lines appearing on more than threshold fraction of pages
    repeated_lines = set()
    for line, count in candidate_counter.items():
        if count / num_pages > threshold:
            repeated_lines.add(line)

    if not repeated_lines:
        return pages

    cleaned = []
    for page_text, page_num in pages:
        lines = page_text.split("\n")
        filtered = [l for l in lines if l.strip() not in repeated_lines]
        cleaned.append(("\n".join(filtered), page_num))

    return cleaned


def _split_large_chunk(text: str, max_size: int) -> List[str]:
    """
    Split an oversized chunk at sentence boundaries to stay near max_size.
    """
    if len(text) <= max_size:
        return [text]
    sentences = _split_sentences(text)
    result: List[str] = []
    buf = ""
    for s in sentences:
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= max_size:
            buf = buf + " " + s
        else:
            result.append(buf)
            buf = s
    if buf:
        result.append(buf)
    return result


def _semantic_chunk_text(text: str, embedding_model: str, max_chunk_size: int = 1500) -> List[str]:
    """
    Split text into chunks at topic boundaries using embedding similarity.
    Uses SemanticChunker from langchain-experimental.
    Chunks exceeding max_chunk_size are further split at sentence boundaries.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_experimental.text_splitter import SemanticChunker

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
        buffer_size=3,
        min_chunk_size=300,
    )
    raw = chunker.split_text(text)
    # Post-split oversized chunks at sentence boundaries
    result: List[str] = []
    for chunk in raw:
        result.extend(_split_large_chunk(chunk, max_chunk_size))
    return result


def _extract_pages_pymupdf(file_path: str) -> Tuple[List[Tuple[str, int]], int]:
    """Extract per-page text using PyMuPDF. Returns (pages, total_page_count)."""
    doc = fitz.open(file_path)

    if doc.is_encrypted:
        if not doc.authenticate(""):
            doc.close()
            raise Exception("PDF is encrypted and cannot be decrypted with empty password")

    if doc.page_count == 0:
        doc.close()
        raise Exception("PDF has no pages")

    # Detect browser-generated PDFs and use sort=True for correct reading order
    meta = doc.metadata or {}
    producer = (meta.get("producer") or "").lower()
    creator = (meta.get("creator") or "").lower()
    browser_hints = ["safari", "chrome", "chromium", "firefox", "mozilla", "webkit",
                     "wkhtmltopdf", "quartz pdfcontext", "headless"]
    use_sort = any(hint in producer or hint in creator for hint in browser_hints)

    text_blocks: List[Tuple[str, int]] = []
    for i in range(doc.page_count):
        try:
            page = doc.load_page(i)
            page_text = page.get_text(sort=use_sort) or ""
            if page_text.strip():
                text_blocks.append((page_text, i + 1))
        except Exception:
            continue

    page_count = doc.page_count
    doc.close()
    return text_blocks, page_count


_docling_converter = None


def _get_docling_converter():
    """Build the Docling converter once per process; its layout/OCR models are large."""
    global _docling_converter
    if _docling_converter is None:
        from docling.document_converter import DocumentConverter
        _docling_converter = DocumentConverter()
    return _docling_converter


def _extract_pages_docling(file_path: str) -> Tuple[List[Tuple[str, int]], int]:
    """
    Extract per-page text using Docling's layout-aware PDF converter.
    Returns (pages, total_page_count) in the same shape as _extract_pages_pymupdf.
    Docling handles reading order and scanned-page OCR internally.
    """
    logger.info("docling: converting %s", file_path)
    try:
        result = _get_docling_converter().convert(file_path)
    except Exception as e:
        logger.exception("docling: conversion failed for %s", file_path)
        if "encrypt" in str(e).lower():
            raise Exception(f"PDF is encrypted and cannot be decrypted: {e}")
        raise
    logger.info("docling: conversion succeeded for %s", file_path)

    doc = result.document
    page_count = doc.num_pages() if hasattr(doc, "num_pages") else len(doc.pages)
    if page_count == 0:
        raise Exception("PDF has no pages")

    pages_text: Dict[int, List[str]] = {}
    for item, _level in doc.iterate_items():
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        page_no = prov[0].page_no
        # Tables render better as markdown; everything else exposes plain .text
        if hasattr(item, "export_to_markdown"):
            text = item.export_to_markdown(doc)
        else:
            text = getattr(item, "text", "")
        if text and text.strip():
            pages_text.setdefault(page_no, []).append(text)

    text_blocks = [("\n\n".join(texts), page_no) for page_no, texts in sorted(pages_text.items())]
    return text_blocks, page_count


def extract_text_from_pdf(
    file_path: str,
    chunk_size: int = 2200,
    chunk_overlap: int = 400,
    chunking_strategy: str = "fixed",
    embedding_model: str = "intfloat/multilingual-e5-base",
    ocr_fallback: bool = True,
    extraction_backend: str = "pymupdf",
) -> List[Dict[str, Any]]:
    """
    Extract text from PDF and split into chunks with page numbers.
    Concatenates all pages for cross-page chunking.

    Args:
        file_path: Path to the PDF file
        chunk_size: Maximum characters per chunk
        chunk_overlap: Character overlap between chunks (good for RAG)
        extraction_backend: "pymupdf" (fast, default) or "docling" (layout-aware,
            better table/reading-order handling, built-in OCR for scanned pages)

    Returns:
        List of dictionaries with 'text', 'page_number', 'page_numbers', and optionally
        'ocr_used' (True when text was obtained via OCR fallback) keys.

    Raises:
        Exception: If PDF cannot be read, is encrypted, or OCR also fails
    """
    logger.info("extract_text_from_pdf: using extraction_backend=%r for %s", extraction_backend, file_path)
    if extraction_backend == "docling":
        text_blocks, page_count = _extract_pages_docling(file_path)
    elif extraction_backend == "pymupdf":
        text_blocks, page_count = _extract_pages_pymupdf(file_path)
    else:
        raise ValueError(f"Unknown extraction_backend: {extraction_backend!r}. Use 'pymupdf' or 'docling'.")

    pages_with_text = len(text_blocks)

    ocr_used = False
    if not text_blocks:
        if not ocr_fallback:
            raise Exception(
                f"No text could be extracted from the PDF. "
                f"This may be an image-based (scanned) PDF. "
                f"Processed {page_count} pages, found text on {pages_with_text} pages."
            )
        # OCR fallback for scanned/image-based PDFs
        from ingestion.ocr import extract_text_with_ocr
        ocr_text = extract_text_with_ocr(file_path)
        text_blocks = [(ocr_text, 1)]
        ocr_used = True

    # Step 2: Strip trailing page numbers from each page's raw text
    cleaned_blocks: List[Tuple[str, int]] = []
    for page_text, page_num in text_blocks:
        lines = page_text.rstrip().split("\n")
        if lines and re.match(r"^\s*\d{1,4}\s*$", lines[-1]):
            page_text = "\n".join(lines[:-1])
        cleaned_blocks.append((page_text, page_num))
    text_blocks = cleaned_blocks

    # Step 3: Header/footer dedup (on raw text before normalization)
    text_blocks = _detect_and_remove_headers_footers(text_blocks)

    # Step 4: Normalize each page
    normalized_pages: List[Tuple[str, int]] = []
    for page_text, page_num in text_blocks:
        normalized = _normalize_pdf_text(page_text)
        if normalized:
            normalized_pages.append((normalized, page_num))

    if not normalized_pages:
        normalized = _normalize_pdf_text(text_blocks[0][0])
        chunk = {"text": normalized, "page_number": text_blocks[0][1], "page_numbers": [text_blocks[0][1]]}
        if ocr_used:
            chunk["ocr_used"] = True
        return [chunk]

    # Step 5: Concatenate all pages, tracking character offsets per page
    full_text = ""
    page_offsets: List[Tuple[int, int, int]] = []  # (start, end, page_num)
    for page_text, page_num in normalized_pages:
        start = len(full_text)
        if full_text:
            full_text += "\n\n"
            start = len(full_text)
        full_text += page_text
        end = len(full_text)
        page_offsets.append((start, end, page_num))

    # Step 6: Chunk the full document
    if chunking_strategy == "semantic":
        raw_chunks = _semantic_chunk_text(full_text, embedding_model)
    else:
        raw_chunks = _chunk_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not raw_chunks:
        chunk = {"text": full_text, "page_number": normalized_pages[0][1], "page_numbers": [normalized_pages[0][1]]}
        if ocr_used:
            chunk["ocr_used"] = True
        return [chunk]

    # Step 7: Map each raw chunk to page number(s) via offset ranges.
    # Chunks are produced in order from full_text, so we advance search_pos
    # after each match to avoid re-scanning text we've already passed.
    chunk_page_mappings: List[List[int]] = []
    search_pos = 0
    for chunk_text in raw_chunks:
        chunk_start = full_text.find(chunk_text, search_pos)
        if chunk_start == -1:
            # Fallback: assign to last known page
            chunk_page_mappings.append([normalized_pages[-1][1]])
            continue
        chunk_end = chunk_start + len(chunk_text)
        pages_for_chunk = [
            p_num for p_start, p_end, p_num in page_offsets
            if chunk_start < p_end and chunk_end > p_start
        ]
        chunk_page_mappings.append(pages_for_chunk or [normalized_pages[-1][1]])
        search_pos = chunk_end

    # Step 8: Apply overlap (only for fixed chunking)
    if chunking_strategy == "semantic":
        final_chunks = raw_chunks
    else:
        final_chunks = _apply_overlap(raw_chunks, chunk_overlap, chunk_size)

    # Step 9: Build result — collapse newlines in non-table text
    all_chunks = []
    for i, chunk_text in enumerate(final_chunks):
        pages = chunk_page_mappings[i] if i < len(chunk_page_mappings) else [1]
        # Replace single \n with space, but preserve newlines in table-like lines
        # (lines with 2+ consecutive spaces acting as column separators)
        lines = chunk_text.split("\n")
        merged = []
        for line in lines:
            is_table_line = bool(re.search(r"  {2,}", line))
            if merged and not is_table_line:
                prev_is_table = bool(re.search(r"  {2,}", merged[-1]))
                if not prev_is_table:
                    merged[-1] = merged[-1] + " " + line
                    continue
            merged.append(line)
        chunk_text = "\n".join(merged)

        chunk = {
            "text": chunk_text,
            "page_number": pages[0],
            "page_numbers": pages,
        }
        if ocr_used:
            chunk["ocr_used"] = True
        all_chunks.append(chunk)

    return all_chunks
