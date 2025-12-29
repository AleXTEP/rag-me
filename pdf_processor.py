from pypdf import PdfReader
from typing import List
import re
import logging

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path: str, chunk_size: int = 1000) -> List[str]:
    """
    Extract text from PDF and split into chunks.
    
    Args:
        file_path: Path to the PDF file
        chunk_size: Maximum characters per chunk
    
    Returns:
        List of text chunks
    
    Raises:
        Exception: If PDF cannot be read or is encrypted
    """
    try:
        reader = PdfReader(file_path)
        
        # Check if PDF is encrypted
        if reader.is_encrypted:
            try:
                # Try to decrypt with empty password (common case)
                reader.decrypt("")
            except Exception as e:
                logger.warning(f"PDF is encrypted and cannot be decrypted: {str(e)}")
                raise Exception(f"PDF is encrypted and cannot be decrypted: {str(e)}")
        
        num_pages = len(reader.pages)
        logger.info(f"Processing PDF with {num_pages} pages")
        
        if num_pages == 0:
            raise Exception("PDF has no pages")
        
        text = ""
        pages_with_text = 0
        
        # Extract text from all pages
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += page_text + "\n"
                    pages_with_text += 1
                else:
                    logger.warning(f"Page {i+1} returned no text (may be image-based)")
            except Exception as e:
                logger.warning(f"Error extracting text from page {i+1}: {str(e)}")
                continue
        
        text = text.strip()
        
        if not text:
            logger.error(f"No text extracted from PDF. Pages processed: {num_pages}, Pages with text: {pages_with_text}")
            raise Exception(
                f"No text could be extracted from the PDF. "
                f"This may be an image-based (scanned) PDF. "
                f"Processed {num_pages} pages, found text on {pages_with_text} pages."
            )
        
        logger.info(f"Successfully extracted text from {pages_with_text}/{num_pages} pages")
        
    except Exception as e:
        logger.error(f"Error reading PDF file: {str(e)}")
        raise
    
    # Check if text has paragraph separators (\n\n)
    has_paragraph_separators = "\n\n" in text
    
    chunks = []
    
    if has_paragraph_separators:
        # Split by paragraphs if they exist
        current_chunk = ""
        
        for paragraph in text.split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            if len(current_chunk) + len(paragraph) + 1 <= chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    else:
        # No paragraph separators - split by character count or sentences
        # Try to split by sentences first (period followed by space or newline)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) > 1:
            # Split by sentences
            current_chunk = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            
            # Add remaining chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        else:
            # No sentence separators either - split by character count
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
    
    return chunks if chunks else [text]

