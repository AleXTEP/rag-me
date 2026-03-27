"""
OCR (Optical Character Recognition) integration.

Provides a fallback for image-based (scanned) PDFs that yield no extractable text.
Mirrors the HyDE provider pattern — local Tesseract by default, pluggable external providers.
"""

import base64
import logging

from config import OCR_PROVIDER, OCR_API_KEY, OCR_API_URL

logger = logging.getLogger(__name__)


def extract_text_with_ocr(pdf_path: str) -> str:
    """Extract text from a PDF using the configured OCR provider."""
    provider = OCR_PROVIDER.lower()
    logger.info(f"Running OCR on {pdf_path!r} using provider {provider!r}")
    if provider == "tesseract":
        return _tesseract(pdf_path)
    elif provider == "deepseek":
        return _deepseek(pdf_path)
    elif provider == "custom":
        return _custom_http(pdf_path)
    else:
        raise ValueError(
            f"Unknown OCR_PROVIDER: {provider!r}. Use 'tesseract', 'deepseek', or 'custom'."
        )


def _tesseract(pdf_path: str) -> str:
    """Convert each PDF page to an image and run Tesseract OCR."""
    from pdf2image import convert_from_path
    import pytesseract

    images = convert_from_path(pdf_path)
    if not images:
        raise Exception("pdf2image returned no pages")

    pages_text = []
    for img in images:
        text = pytesseract.image_to_string(img)
        if text.strip():
            pages_text.append(text)

    result = "\n\n".join(pages_text)
    if not result.strip():
        raise Exception("Tesseract OCR produced no text")
    return result


def _deepseek(pdf_path: str) -> str:
    """Send PDF pages as base64-encoded images to the DeepSeek OCR API."""
    import httpx
    from pdf2image import convert_from_path
    import io

    if not OCR_API_KEY:
        raise ValueError("OCR_API_KEY must be set for DeepSeek OCR provider")

    endpoint = OCR_API_URL or "https://api.deepseek.com/v1/ocr"
    images = convert_from_path(pdf_path)

    pages_text = []
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        response = httpx.post(
            endpoint,
            headers={"Authorization": f"Bearer {OCR_API_KEY}"},
            json={"image": b64},
            timeout=60,
        )
        response.raise_for_status()
        text = response.json().get("text", "")
        if text.strip():
            pages_text.append(text)

    result = "\n\n".join(pages_text)
    if not result.strip():
        raise Exception("DeepSeek OCR produced no text")
    return result


def _custom_http(pdf_path: str) -> str:
    """POST the PDF file bytes to a custom OCR endpoint."""
    import httpx

    if not OCR_API_URL:
        raise ValueError("OCR_API_URL must be set for custom OCR provider")

    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    headers = {}
    if OCR_API_KEY:
        headers["Authorization"] = f"Bearer {OCR_API_KEY}"

    response = httpx.post(
        OCR_API_URL,
        headers=headers,
        content=file_bytes,
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()
    text = data.get("text", "")
    if not text.strip():
        raise Exception("Custom OCR endpoint returned no text")
    return text
