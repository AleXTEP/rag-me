import re
from typing import List, Dict, Tuple, Any
from PyPDF2 import PdfReader

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

            # Keep newline if current looks like a heading or list item
            is_heading = (len(line) < 80 and line.isupper())
            is_list = bool(re.match(r"^(\-|\*|\u2022|\d+[\.\)]|[a-zA-Z][\.\)])\s+", line))
            next_is_list = bool(re.match(r"^(\-|\*|\u2022|\d+[\.\)]|[a-zA-Z][\.\)])\s+", nxt))

            if nxt != "" and not is_heading and not is_list and not next_is_list:
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
    A practical sentence splitter (regex-based). Not perfect, but better than naive.
    """
    # Protect some common abbreviations to reduce bad splits
    protected = [
        "e.g.", "i.e.", "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "vs.", "etc.",
        "Fig.", "Eq.", "No.", "St.", "Inc.", "Ltd."
    ]
    placeholder = "§§§"
    for ab in protected:
        text = text.replace(ab, ab.replace(".", placeholder))

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    parts = [p.replace(placeholder, ".").strip() for p in parts if p.strip()]
    return parts if parts else [text.strip()]


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int = 500) -> List[str]:
    """
    Chunk text with preference for paragraph boundaries, then sentence, then word.
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
        if len(block) <= chunk_size:
            if not current:
                current = block
            elif len(current) + 2 + len(block) <= chunk_size:
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
            if len(s) > chunk_size:
                # Sentence too large: split by words
                words = s.split()
                wbuf = ""
                for w in words:
                    if not wbuf:
                        wbuf = w
                    elif len(wbuf) + 1 + len(w) <= chunk_size:
                        wbuf = wbuf + " " + w
                    else:
                        chunks.append(wbuf.strip())
                        wbuf = w
                if wbuf.strip():
                    chunks.append(wbuf.strip())
                buf = ""
                continue

            if not buf:
                buf = s
            elif len(buf) + 1 + len(s) <= chunk_size:
                buf = buf + " " + s
            else:
                chunks.append(buf.strip())
                buf = s

        if buf.strip():
            chunks.append(buf.strip())

    flush()

    # Apply overlap between chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
                continue
            
            # Get last N sentences from previous chunk for overlap
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
                overlapped.append(" ".join(overlap_sentences) + " " + chunk)
            else:
                overlapped.append(chunk)
        
        chunks = overlapped

    return chunks



def extract_text_from_pdf(file_path: str, chunk_size: int = 800, chunk_overlap: int = 300) -> List[Dict[str, Any]]:
    """
    Extract text from PDF and split into chunks with page numbers.

    Args:
        file_path: Path to the PDF file
        chunk_size: Maximum characters per chunk
        chunk_overlap: Character overlap between chunks (good for RAG)

    Returns:
        List of dictionaries with 'text' and 'page_number' keys

    Raises:
        Exception: If PDF cannot be read or is encrypted
    """
    try:
        reader = PdfReader(file_path)

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as e:
                raise Exception(f"PDF is encrypted and cannot be decrypted: {str(e)}")

        if len(reader.pages) == 0:
            raise Exception("PDF has no pages")

        # Extract text from each page, keeping track of page numbers
        text_blocks: List[Tuple[str, int]] = []
        pages_with_text = 0

        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    # Page numbers are 1-indexed for user display
                    text_blocks.append((page_text, i + 1))
                    pages_with_text += 1
            except Exception:
                # skip problematic pages
                continue

        if not text_blocks:
            raise Exception(
                f"No text could be extracted from the PDF. "
                f"This may be an image-based (scanned) PDF. "
                f"Processed {len(reader.pages)} pages, found text on {pages_with_text} pages."
            )

        # Chunk each page separately and combine with page numbers
        all_chunks = []
        for page_text, page_num in text_blocks:
            normalized = _normalize_pdf_text(page_text)
            if not normalized:
                continue
            
            page_chunks = _chunk_text(normalized, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            # Add page number to each chunk
            for chunk_text in page_chunks:
                all_chunks.append({
                    "text": chunk_text,
                    "page_number": page_num
                })
        
        # Fallback: if no chunks, return the full normalized text from first page
        if not all_chunks:
            normalized = _normalize_pdf_text(text_blocks[0][0])
            return [{"text": normalized, "page_number": text_blocks[0][1]}]
        
        return all_chunks

    except Exception:
        raise
