import re
from collections import Counter
from typing import List, Dict, Tuple, Any
import fitz  # PyMuPDF

def _normalize_pdf_text(text: str) -> str:
    """
    Normalize text extracted from PDFs:
    - Normalize line endings
    - Convert bullet-like unicode chars to a common form
    - Merge hard-wrapped lines inside paragraphs
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

    # Merge wrapped lines: replace single newlines within paragraphs with spaces.
    # Keep double newlines as paragraph separators.
    # Also keep newlines before list items / headings heuristically.
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "":
            out.append("")  # paragraph break marker
            i += 1
            continue

        # If next line exists and is not empty, decide whether to merge
        if i + 1 < len(lines):
            nxt = lines[i + 1].strip()

            # Keep newline if current or next looks like a list item
            is_list = bool(re.match(r"^(\-|\*|\u2022|\d+[\.\)]|[a-zA-Z][\.\)])\s+", line))
            next_is_list = bool(re.match(r"^(\-|\*|\u2022|\d+[\.\)]|[a-zA-Z][\.\)])\s+", nxt))

            if nxt != "" and not is_list and not next_is_list:
                # Merge if line doesn't end a sentence strongly (but still merge often in PDFs)
                # Also avoid merging if line ends with hyphenated word break: "exam-\nple" -> "example"
                if line.endswith("-") and nxt and nxt[0].islower():
                    out.append(line[:-1] + nxt)
                    i += 2
                    continue
                else:
                    out.append(line + " " + nxt)
                    i += 2
                    continue

        out.append(line)
        i += 1

    # Rebuild with paragraphs
    normalized = "\n".join(out)
    # Restore paragraph breaks from empty lines, collapse multiple empties to two newlines
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


def extract_text_from_pdf(
    file_path: str,
    chunk_size: int = 2200,
    chunk_overlap: int = 400,
    chunking_strategy: str = "fixed",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[Dict[str, Any]]:
    """
    Extract text from PDF and split into chunks with page numbers.
    Concatenates all pages for cross-page chunking.

    Args:
        file_path: Path to the PDF file
        chunk_size: Maximum characters per chunk
        chunk_overlap: Character overlap between chunks (good for RAG)

    Returns:
        List of dictionaries with 'text', 'page_number', and 'page_numbers' keys

    Raises:
        Exception: If PDF cannot be read or is encrypted
    """
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

    # Step 1: Extract all pages as (raw_text, page_number) tuples
    text_blocks: List[Tuple[str, int]] = []
    pages_with_text = 0

    for i in range(doc.page_count):
        try:
            page = doc.load_page(i)
            page_text = page.get_text(sort=use_sort) or ""
            if page_text.strip():
                text_blocks.append((page_text, i + 1))
                pages_with_text += 1
        except Exception:
            continue

    page_count = doc.page_count
    doc.close()

    if not text_blocks:
        raise Exception(
            f"No text could be extracted from the PDF. "
            f"This may be an image-based (scanned) PDF. "
            f"Processed {page_count} pages, found text on {pages_with_text} pages."
        )

    # Step 1b: Strip trailing page numbers from each page's raw text
    cleaned_blocks: List[Tuple[str, int]] = []
    for page_text, page_num in text_blocks:
        lines = page_text.rstrip().split("\n")
        if lines and re.match(r"^\s*\d{1,4}\s*$", lines[-1]):
            page_text = "\n".join(lines[:-1])
        cleaned_blocks.append((page_text, page_num))
    text_blocks = cleaned_blocks

    # Step 2: Header/footer dedup (on raw text before normalization)
    text_blocks = _detect_and_remove_headers_footers(text_blocks)

    # Step 3: Normalize each page
    normalized_pages: List[Tuple[str, int]] = []
    for page_text, page_num in text_blocks:
        normalized = _normalize_pdf_text(page_text)
        if normalized:
            normalized_pages.append((normalized, page_num))

    if not normalized_pages:
        normalized = _normalize_pdf_text(text_blocks[0][0])
        return [{"text": normalized, "page_number": text_blocks[0][1], "page_numbers": [text_blocks[0][1]]}]

    # Step 4: Concatenate all pages, tracking character offsets per page
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

    # Step 5: Chunk the full document
    if chunking_strategy == "semantic":
        raw_chunks = _semantic_chunk_text(full_text, embedding_model)
    else:
        raw_chunks = _chunk_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not raw_chunks:
        return [{"text": full_text, "page_number": normalized_pages[0][1], "page_numbers": [normalized_pages[0][1]]}]

    # Step 6: Map each raw chunk to page number(s) via offset ranges
    chunk_page_mappings: List[List[int]] = []
    for chunk_text in raw_chunks:
        chunk_start = full_text.find(chunk_text)
        if chunk_start == -1:
            # Fallback: assign to last known page
            chunk_page_mappings.append([normalized_pages[-1][1]])
            continue
        chunk_end = chunk_start + len(chunk_text)
        pages_for_chunk = []
        for p_start, p_end, p_num in page_offsets:
            if chunk_start < p_end and chunk_end > p_start:
                pages_for_chunk.append(p_num)
        if not pages_for_chunk:
            chunk_page_mappings.append([normalized_pages[-1][1]])
        else:
            chunk_page_mappings.append(pages_for_chunk)

    # Step 7: Apply overlap (only for fixed chunking)
    if chunking_strategy == "semantic":
        final_chunks = raw_chunks
    else:
        final_chunks = _apply_overlap(raw_chunks, chunk_overlap, chunk_size)

    # Step 8: Build result — collapse newlines in non-table text
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

        all_chunks.append({
            "text": chunk_text,
            "page_number": pages[0],
            "page_numbers": pages,
        })

    return all_chunks
