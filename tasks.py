import os
from celery_app import celery_app
from config import WEAVIATE_URL, ELASTICSEARCH_URL, EMBEDDING_MODEL, UPLOAD_DIR, CHUNKING_STRATEGY
from pdf_processor import extract_text_from_pdf
from stores import get_store

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

        # Get singleton store instances
        vector_store = get_store("weaviate", weaviate_url=WEAVIATE_URL, embedding_model=EMBEDDING_MODEL)
        elasticsearch_store = get_store("elasticsearch", elasticsearch_url=ELASTICSEARCH_URL)

        # Store embeddings in Weaviate
        vector_store.add_documents(file_id, text_chunks, filename)

        # Store documents in Elasticsearch for keyword search
        elasticsearch_store.add_documents(file_id, text_chunks, filename)

        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

        return {
            "status": "success",
            "file_id": file_id,
            "chunks_processed": len(text_chunks)
        }
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        return {"status": "error", "message": str(e)}
