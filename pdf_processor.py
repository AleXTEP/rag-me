import re
from typing import List
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


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int = 100) -> List[str]:
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

    # Apply overlap (character-based) between final chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, c in enumerate(chunks):
            if i == 0:
                overlapped.append(c)
                continue
            prev = overlapped[-1]
            overlap = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
            overlapped.append((overlap + "\n" + c).strip())
        chunks = overlapped

    return chunks


def extract_text_from_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    """
    Extract text from PDF and split into chunks.

    Args:
        file_path: Path to the PDF file
        chunk_size: Maximum characters per chunk
        chunk_overlap: Character overlap between chunks (good for RAG)

    Returns:
        List of text chunks

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

        full_text_parts = []
        pages_with_text = 0

        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    full_text_parts.append(page_text)
                    pages_with_text += 1
            except Exception:
                # skip problematic pages
                continue

        raw_text = "\n\n&&&".join(full_text_parts).strip()
        if not raw_text:
            raise Exception(
                f"No text could be extracted from the PDF. "
                f"This may be an image-based (scanned) PDF. "
                f"Processed {len(reader.pages)} pages, found text on {pages_with_text} pages."
            )

        normalized = _normalize_pdf_text(raw_text)
        chunks = _chunk_text(normalized, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return chunks if chunks else [normalized]

    except Exception:
        raise
