from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import os
import uuid

from config import UPLOAD_DIR, WEAVIATE_URL, ELASTICSEARCH_URL, EMBEDDING_MODEL
from tasks import process_pdf_task
from stores import get_store, close_all, WeaviateStore, ElasticsearchStore
from reranker import rerank


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize both store singletons
    get_store("weaviate", weaviate_url=WEAVIATE_URL, embedding_model=EMBEDDING_MODEL)
    get_store("elasticsearch", elasticsearch_url=ELASTICSEARCH_URL)
    yield
    # Shutdown: close all connections
    close_all()


app = FastAPI(title="RAG System API", version="1.0.0", lifespan=lifespan)


def get_weaviate_store() -> WeaviateStore:
    return get_store("weaviate", weaviate_url=WEAVIATE_URL, embedding_model=EMBEDDING_MODEL)


def get_elasticsearch_store() -> ElasticsearchStore:
    return get_store("elasticsearch", elasticsearch_url=ELASTICSEARCH_URL)


def combine_search_results_with_rrf(weaviate_results: dict, elasticsearch_results: dict, search_limit: int, rrf_k: int = 60):
    """
    Combine Weaviate and Elasticsearch search results using Reciprocal Rank Fusion (RRF) algorithm.

    Args:
        weaviate_results: Results from Weaviate search
        elasticsearch_results: Results from Elasticsearch search
        search_limit: Maximum number of results to return
        rrf_k: RRF constant (default 60)

    Returns:
        Tuple of (final_results, combined_results, weaviate_count, elasticsearch_count)
    """
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
            rrf_score += 1.0 / (rrf_k + result["weaviate_rank"])

        # Add contribution from Elasticsearch rank if present
        if result.get("elasticsearch_rank") is not None:
            rrf_score += 1.0 / (rrf_k + result["elasticsearch_rank"])

        result["rrf_score"] = rrf_score

    # Convert to list and sort by RRF score (descending)
    final_results = list(combined_results.values())
    final_results.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)

    # Limit to requested number of results
    final_results = final_results[:search_limit]

    weaviate_count = len(weaviate_results.get("objects", []))
    elasticsearch_count = len(elasticsearch_results.get("objects", []))

    return final_results, combined_results, weaviate_count, elasticsearch_count

class SearchRequest(BaseModel):
    q: str = Field(..., description="Search query")
    limit: Optional[int] = Field(5, ge=1, le=50, description="Number of results to return")

class SearchRequestWithContext(BaseModel):
    q: str = Field(..., description="Search query")
    limit: Optional[int] = Field(5, ge=1, le=50, description="Number of results to return")
    context_chunks: Optional[int] = Field(5, ge=0, le=20, description="Number of chunks before and after each result to include")

@app.get("/")
def root():
    return {"message": "RAG System API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    vector_store: WeaviateStore = Depends(get_weaviate_store)
):
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
        if vector_store.filename_exists(original_filename):
            raise HTTPException(
                status_code=409,
                detail=f"File with filename '{original_filename}' already exists in the system"
            )

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
def list_documents(
    elasticsearch_store: ElasticsearchStore = Depends(get_elasticsearch_store)
):
    """
    Get a list of all documents in the system with their metadata.
    Returns file_id, filename, and chunk_count for each document.
    Uses Elasticsearch for faster retrieval via aggregations.
    """
    try:
        documents = elasticsearch_store.get_all_documents()
        return JSONResponse({
            "documents": documents,
            "count": len(documents)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

@app.delete("/documents/{file_id}")
def delete_document(
    file_id: str,
    vector_store: WeaviateStore = Depends(get_weaviate_store),
    elasticsearch_store: ElasticsearchStore = Depends(get_elasticsearch_store)
):
    """
    Delete a document and all its chunks by file_id.
    Removes the document from both Weaviate (vector store) and Elasticsearch.

    Args:
        file_id: Unique identifier for the document to delete

    Returns:
        Status message with deletion results
    """
    try:
        weaviate_deleted = vector_store.delete_document(file_id)
        elasticsearch_deleted = elasticsearch_store.delete_document(file_id)

        if weaviate_deleted == 0 and elasticsearch_deleted == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Document with file_id '{file_id}' not found"
            )

        return JSONResponse({
            "status": "success",
            "file_id": file_id,
            "weaviate_chunks_deleted": weaviate_deleted,
            "elasticsearch_chunks_deleted": elasticsearch_deleted,
            "message": f"Document '{file_id}' deleted successfully"
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.post("/search")
def search_documents(
    body: SearchRequest = Body(..., description="Search request (JSON body)"),
    vector_store: WeaviateStore = Depends(get_weaviate_store)
):
    """
    Search for similar documents using semantic search.
    Accepts JSON body with 'q' (search query) and optional 'limit' (number of results).
    """
    search_query = body.q
    search_limit = body.limit or 5

    try:
        results = vector_store.search(search_query, n_results=search_limit)
        return JSONResponse(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")

@app.post("/search/keywords")
def search_by_keywords(
    body: SearchRequest = Body(..., description="Keyword search request (JSON body)"),
    elasticsearch_store: ElasticsearchStore = Depends(get_elasticsearch_store)
):
    """
    Search for documents by keywords using Elasticsearch.
    Accepts JSON body with 'q' (search query) and optional 'limit' (number of results).
    """
    search_query = body.q
    search_limit = body.limit or 5
    try:
        results = elasticsearch_store.search(search_query, n_results=search_limit)
        return JSONResponse(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")

@app.post("/search/double")
def search_double(
    body: SearchRequest = Body(..., description="Double search request (JSON body)"),
    vector_store: WeaviateStore = Depends(get_weaviate_store),
    elasticsearch_store: ElasticsearchStore = Depends(get_elasticsearch_store)
):
    """
    Search for documents using both Weaviate (semantic search) and Elasticsearch (keyword search).
    Combines results using Reciprocal Rank Fusion (RRF) algorithm.
    Accepts JSON body with 'q' (search query) and optional 'limit' (number of results per source).
    """
    search_query = body.q
    search_limit = body.limit or 5
    fetch_limit = search_limit * 4
    RRF_K = 60
    try:
        weaviate_results = vector_store.search(search_query, n_results=fetch_limit)
        elasticsearch_results = elasticsearch_store.search(search_query, n_results=fetch_limit)
        rrf_results, combined_results, weaviate_count, elasticsearch_count = combine_search_results_with_rrf(
            weaviate_results, elasticsearch_results, fetch_limit, RRF_K
        )
        final_results = rerank(search_query, rrf_results, top_n=search_limit)
        return JSONResponse({
            "objects": final_results,
            "weaviate_count": weaviate_count,
            "elasticsearch_count": elasticsearch_count,
            "combined_count": len(final_results),
            "unique_count": len(combined_results),
            "rrf_k": RRF_K
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in double search: {str(e)}")

@app.post("/search/double/context")
def search_double_with_context(
    body: SearchRequestWithContext = Body(..., description="Double search request with context chunks (JSON body)"),
    vector_store: WeaviateStore = Depends(get_weaviate_store),
    elasticsearch_store: ElasticsearchStore = Depends(get_elasticsearch_store)
):
    """
    Search for documents using both Weaviate (semantic search) and Elasticsearch (keyword search).
    Combines results using Reciprocal Rank Fusion (RRF) algorithm.

    Groups results by document, then for each document:
    - Collects all selected chunk indices
    - Extends each selected chunk with N chunks before and N chunks after
    - Merges all chunk ranges without duplicates
    - Retrieves all chunks for the document and joins the text

    Returns results grouped by document with augmented chunk ranges.
    Accepts JSON body with 'q' (search query), optional 'limit' (number of results per source),
    and optional 'context_chunks' (number of chunks before/after each selected chunk, default 5).
    """
    search_query = body.q
    search_limit = body.limit or 5
    fetch_limit = search_limit * 4
    context_chunks = body.context_chunks or 5
    RRF_K = 60
    try:
        weaviate_results = vector_store.search(search_query, n_results=fetch_limit)
        elasticsearch_results = elasticsearch_store.search(search_query, n_results=fetch_limit)
        rrf_results, combined_results, weaviate_count, elasticsearch_count = combine_search_results_with_rrf(
            weaviate_results, elasticsearch_results, fetch_limit, RRF_K
        )
        final_results = rerank(search_query, rrf_results, top_n=search_limit)
        # Group results by document (file_id)
        documents = {}
        for result in final_results:
            file_id = result.get("file_id")
            chunk_index = result.get("chunk_index")

            if file_id not in documents:
                documents[file_id] = {
                    "file_id": file_id,
                    "filename": result.get("filename", ""),
                    "chunk_count": result.get("chunk_count"),
                    "selected_chunk_indices": [],
                    "results": []
                }

            documents[file_id]["selected_chunk_indices"].append(chunk_index)
            documents[file_id]["results"].append(result)

        # For each document, calculate merged chunk ranges
        documents_with_context = []
        for file_id, doc_data in documents.items():
            selected_indices = sorted(set(doc_data["selected_chunk_indices"]))
            chunk_count = doc_data["chunk_count"]

            # Calculate chunk ranges for each selected chunk and merge them
            all_chunk_indices = set()
            for chunk_index in selected_indices:
                # Calculate range for this chunk (N before and N after)
                start_idx = max(0, chunk_index - context_chunks)
                if chunk_count is not None:
                    end_idx = min(chunk_count - 1, chunk_index + context_chunks)
                else:
                    end_idx = chunk_index + context_chunks

                # Add all indices in this range
                for idx in range(start_idx, end_idx + 1):
                    all_chunk_indices.add(idx)

            # Convert to sorted list
            augmented_chunk_indices = sorted(all_chunk_indices)

            # Retrieve all chunks for this document in one go
            all_chunks = elasticsearch_store.get_chunks_by_range(file_id, augmented_chunk_indices)

            # Sort by chunk_index (should already be sorted, but ensure it)
            all_chunks.sort(key=lambda x: x["chunk_index"])

            # Join text from all chunks
            joined_text = "\n".join([chunk["text"] for chunk in all_chunks])

            # Create document result
            document_result = {
                "file_id": file_id,
                "filename": doc_data["filename"],
                "selected_chunk_indices": selected_indices,
                "augmented_chunk_indices": augmented_chunk_indices,
                "chunk_count": chunk_count,
                "selected_count": len(selected_indices),
                "augmented_count": len(augmented_chunk_indices),
                "chunks": all_chunks,
                "joined_text": joined_text,
                "results": doc_data["results"]  # Original search results for this document
            }
            documents_with_context.append(document_result)

        return JSONResponse({
            "documents": documents_with_context,
            "document_count": len(documents_with_context),
            "weaviate_count": weaviate_count,
            "elasticsearch_count": elasticsearch_count,
            "combined_count": len(final_results),
            "unique_count": len(combined_results),
            "rrf_k": RRF_K,
            "context_chunks": context_chunks
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in double search with context: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
