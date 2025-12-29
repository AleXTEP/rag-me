from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import os
import uuid
from config import UPLOAD_DIR, WEAVIATE_URL, ELASTICSEARCH_URL, EMBEDDING_MODEL
from tasks import process_pdf_task
from vector_store import VectorStore
from elasticsearch_store import ElasticsearchStore

app = FastAPI(title="RAG System API", version="1.0.0")

class SearchRequest(BaseModel):
    q: str = Field(..., description="Search query")
    limit: Optional[int] = Field(5, ge=1, le=50, description="Number of results to return")

@app.get("/")
def root():
    return {"message": "RAG System API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF file for processing.
    The file will be queued for processing by Celery.
    """
    # Validate file type
    if file.filename and not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Check if filename already exists
    original_filename = file.filename or ""
    if original_filename:
        vector_store = None
        try:
            vector_store = VectorStore(WEAVIATE_URL, EMBEDDING_MODEL)
            if vector_store.filename_exists(original_filename):
                raise HTTPException(
                    status_code=409, 
                    detail=f"File with filename '{original_filename}' already exists in the system"
                )
        finally:
            if vector_store:
                vector_store.close()
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename or "")[1]
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Queue task for processing
        original_filename = file.filename or f"{file_id}{file_extension}"
        task = process_pdf_task.delay(file_path, file_id, original_filename)
        
        return JSONResponse({
            "status": "queued",
            "file_id": file_id,
            "task_id": task.id,
            "filename": file.filename,
            "message": "File uploaded and queued for processing"
        })
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.get("/task/{task_id}")
def get_task_status(task_id: str):
    """
    Get the status of a processing task.
    """
    from celery_app import celery_app
    task = celery_app.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Task is waiting to be processed'
        }
    elif task.state == 'PROGRESS':
        response = {
            'state': task.state,
            'status': task.info.get('status', 'Processing...')
        }
    elif task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'result': task.result
        }
    else:
        response = {
            'state': task.state,
            'error': str(task.info) if task.info else 'Unknown error'
        }
    
    return response

@app.get("/documents")
def list_documents():
    """
    Get a list of all documents in the system with their metadata.
    Returns file_id, filename, and chunk_count for each document.
    """
    vector_store = None
    try:
        vector_store = VectorStore(WEAVIATE_URL, EMBEDDING_MODEL)
        documents = vector_store.get_all_documents()
        return JSONResponse({
            "documents": documents,
            "count": len(documents)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")
    finally:
        if vector_store:
            vector_store.close()

@app.post("/search")
def search_documents(body: SearchRequest = Body(..., description="Search request (JSON body)")):
    """
    Search for similar documents using semantic search.
    Accepts JSON body with 'q' (search query) and optional 'limit' (number of results).
    """
    search_query = body.q
    search_limit = body.limit or 5
    
    vector_store = None
    try:
        vector_store = VectorStore(WEAVIATE_URL, EMBEDDING_MODEL)
        results = vector_store.search(search_query, n_results=search_limit )
        return JSONResponse(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")
    finally:
        if vector_store:
            vector_store.close()

@app.post("/search/keywords")
def search_by_keywords(body: SearchRequest = Body(..., description="Keyword search request (JSON body)")):
    """
    Search for documents by keywords using Elasticsearch.
    Accepts JSON body with 'q' (search query) and optional 'limit' (number of results).
    """
    search_query = body.q
    search_limit = body.limit or 5
    
    elasticsearch_store = None
    try:
        elasticsearch_store = ElasticsearchStore(ELASTICSEARCH_URL)
        results = elasticsearch_store.search(search_query, n_results=search_limit)
        return JSONResponse(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")
    finally:
        if elasticsearch_store:
            elasticsearch_store.close()

@app.post("/search/double")
def search_double(body: SearchRequest = Body(..., description="Double search request (JSON body)")):
    """
    Search for documents using both Weaviate (semantic search) and Elasticsearch (keyword search).
    Combines results using Reciprocal Rank Fusion (RRF) algorithm.
    Accepts JSON body with 'q' (search query) and optional 'limit' (number of results per source).
    """
    search_query = body.q
    search_limit = body.limit or 5
    
    # RRF constant (typically 60)
    RRF_K = 60
    
    vector_store = None
    elasticsearch_store = None
    
    try:
        # Search both stores in parallel
        vector_store = VectorStore(WEAVIATE_URL, EMBEDDING_MODEL)
        weaviate_results = vector_store.search(search_query, n_results=search_limit)
        
        elasticsearch_store = ElasticsearchStore(ELASTICSEARCH_URL)
        elasticsearch_results = elasticsearch_store.search(search_query, n_results=search_limit)
        
        # Track ranks for each result in each source
        # Key: (file_id, chunk_index), Value: dict with result data and ranks
        combined_results = {}
        
        # Process Weaviate results with ranks (1-indexed)
        for rank, obj in enumerate(weaviate_results.get("objects", []), start=1):
            key = (obj.get("file_id"), obj.get("chunk_index"))
            if key not in combined_results:
                combined_results[key] = {
                    **obj,
                    "sources": ["weaviate"],
                    "weaviate_rank": rank,
                    "elasticsearch_rank": None
                }
            else:
                # Update existing result with Weaviate rank
                combined_results[key]["weaviate_rank"] = rank
                if "weaviate" not in combined_results[key]["sources"]:
                    combined_results[key]["sources"].append("weaviate")
                # Preserve Weaviate-specific metadata
                if "distance" in obj:
                    combined_results[key]["distance"] = obj["distance"]
        
        # Process Elasticsearch results with ranks (1-indexed)
        for rank, obj in enumerate(elasticsearch_results.get("objects", []), start=1):
            key = (obj.get("file_id"), obj.get("chunk_index"))
            if key not in combined_results:
                combined_results[key] = {
                    **obj,
                    "sources": ["elasticsearch"],
                    "weaviate_rank": None,
                    "elasticsearch_rank": rank
                }
            else:
                # Update existing result with Elasticsearch rank
                combined_results[key]["elasticsearch_rank"] = rank
                if "elasticsearch" not in combined_results[key]["sources"]:
                    combined_results[key]["sources"].append("elasticsearch")
                # Preserve Elasticsearch-specific metadata
                if "score" in obj:
                    combined_results[key]["score"] = obj["score"]
                if "highlight" in obj:
                    combined_results[key]["highlight"] = obj["highlight"]
        
        # Calculate RRF scores for each result
        # RRF_score = sum(1 / (k + rank)) for each source where result appears
        for key, result in combined_results.items():
            rrf_score = 0.0
            
            # Add contribution from Weaviate rank if present
            if result.get("weaviate_rank") is not None:
                rrf_score += 1.0 / (RRF_K + result["weaviate_rank"])
            
            # Add contribution from Elasticsearch rank if present
            if result.get("elasticsearch_rank") is not None:
                rrf_score += 1.0 / (RRF_K + result["elasticsearch_rank"])
            
            result["rrf_score"] = rrf_score
        
        # Convert to list and sort by RRF score (descending)
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)
        
        # Limit to requested number of results
        final_results = final_results[:search_limit]
        
        return JSONResponse({
            "objects": final_results,
            "weaviate_count": len(weaviate_results.get("objects", [])),
            "elasticsearch_count": len(elasticsearch_results.get("objects", [])),
            "combined_count": len(final_results),
            "unique_count": len(combined_results),
            "rrf_k": RRF_K
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in double search: {str(e)}")
    finally:
        if vector_store:
            vector_store.close()
        if elasticsearch_store:
            elasticsearch_store.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

