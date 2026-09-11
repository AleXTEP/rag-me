import logging
import os
from worker.celery_app import celery_app
from config import WEAVIATE_URL, ELASTICSEARCH_URL, EMBEDDING_MODEL, UPLOAD_DIR, CHUNKING_STRATEGY, USE_ELASTICSEARCH, EXTRACTION_BACKEND
from ingestion.pdf_processor import extract_text_from_pdf
from stores import get_store

logger = logging.getLogger(__name__)

@celery_app.task(name="process_pdf")
def process_pdf_task(file_path: str, file_id: str, filename: str):
    """
    Celery task to process PDF file:
    1. Extract text from PDF
    2. Embed the text
    3. Store in vector database

    Args:   
        file_path: Path to the uploaded file
        file_id: Unique identifier for the file
        filename: Original filename of the uploaded file
    """
    try:
        # Extract text from PDF
        try:
            text_chunks = extract_text_from_pdf(
                file_path,
                chunking_strategy=CHUNKING_STRATEGY,
                embedding_model=EMBEDDING_MODEL,
                extraction_backend=EXTRACTION_BACKEND,
            )
        except Exception as e:
            # Provide detailed error message
            error_msg = str(e)
            if "encrypted" in error_msg.lower():
                return {
                    "status": "error",
                    "message": f"PDF is encrypted and cannot be processed: {error_msg}"
                }
            elif "no text" in error_msg.lower() or "image-based" in error_msg.lower():
                return {
                    "status": "error",
                    "message": f"{error_msg} Consider using OCR to extract text from scanned PDFs."
                }
            else:
                return {
                    "status": "error",
                    "message": f"Error extracting text from PDF: {error_msg}"
                }

        if not text_chunks:
            return {
                "status": "error",
                "message": "No text extracted from PDF. The PDF may be image-based (scanned) or corrupted."
            }

        ocr_used = any(chunk.get("ocr_used") for chunk in text_chunks)

        # Get singleton store instances
        vector_store = get_store("weaviate", weaviate_url=WEAVIATE_URL, embedding_model=EMBEDDING_MODEL)

        # Insert with compensating rollback: if Elasticsearch fails after Weaviate succeeds, roll back.
        weaviate_ok = False
        try:
            vector_store.add_documents(file_id, text_chunks, filename)
            weaviate_ok = True

            if USE_ELASTICSEARCH:
                elasticsearch_store = get_store("elasticsearch", elasticsearch_url=ELASTICSEARCH_URL)
                elasticsearch_store.add_documents(file_id, text_chunks, filename)
        except Exception as e:
            if weaviate_ok:
                try:
                    vector_store.delete_document(file_id)
                except Exception:
                    pass  # best-effort rollback
            raise

        result = {
            "status": "success",
            "file_id": file_id,
            "chunks_processed": len(text_chunks),
        }
        if ocr_used:
            result["ocr_used"] = True
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        except OSError:
            logger.exception("Failed to remove uploaded file %s", file_path)
