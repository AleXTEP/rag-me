from contextlib import asynccontextmanager
import os
import uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Depends
from fastapi.responses import JSONResponse

from config import UPLOAD_DIR, WEAVIATE_URL, ELASTICSEARCH_URL, EMBEDDING_MODEL, USE_ELASTICSEARCH
from tasks import process_pdf_task
from stores import get_store, close_all, WeaviateStore, ElasticsearchStore
from reranker import rerank_cross_encoder
from schemas import SearchRequest, SearchRequestWithContext
from fusion import combine_search_results_with_rrf
from deps import get_weaviate_store, get_elasticsearch_store
from hyde import generate_hypothetical_document


print(USE_ELASTICSEARCH, "USE_ELASTICSEARCH")

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_store("weaviate", weaviate_url=WEAVIATE_URL, embedding_model=EMBEDDING_MODEL)
    if USE_ELASTICSEARCH:
        get_store("elasticsearch", elasticsearch_url=ELASTICSEARCH_URL)
    yield
    close_all()


app = FastAPI(title="RAG System API", version="1.0.0", lifespan=lifespan)


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
    if file.filename and not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    original_filename = file.filename or ""
    if original_filename:
        if vector_store.filename_exists(original_filename):
            raise HTTPException(
                status_code=409,
                detail=f"File with filename '{original_filename}' already exists in the system"
            )

    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename or "")[1]
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{file_extension}")

    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

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
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.get("/task/{task_id}")
def get_task_status(task_id: str):
    from celery_app import celery_app
    task = celery_app.AsyncResult(task_id)

    if task.state == 'PENDING':
        response = {'state': task.state, 'status': 'Task is waiting to be processed'}
    elif task.state == 'PROGRESS':
        response = {'state': task.state, 'status': task.info.get('status', 'Processing...')}
    elif task.state == 'SUCCESS':
        response = {'state': task.state, 'result': task.result}
    else:
        response = {'state': task.state, 'error': str(task.info) if task.info else 'Unknown error'}

    return response

@app.get("/documents")
def list_documents(
    vector_store: WeaviateStore = Depends(get_weaviate_store),
    elasticsearch_store: Optional[ElasticsearchStore] = Depends(get_elasticsearch_store)
):
    try:
        if elasticsearch_store is not None:
            documents = elasticsearch_store.get_all_documents()
        else:
            documents = vector_store.get_all_documents()
        return JSONResponse({"documents": documents, "count": len(documents)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

@app.delete("/documents/{file_id}")
def delete_document(
    file_id: str,
    vector_store: WeaviateStore = Depends(get_weaviate_store),
    elasticsearch_store: Optional[ElasticsearchStore] = Depends(get_elasticsearch_store)
):
    try:
        weaviate_deleted = vector_store.delete_document(file_id)
        elasticsearch_deleted = elasticsearch_store.delete_document(file_id) if elasticsearch_store else 0

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

@app.post("/search/vector")
def search_documents(
    body: SearchRequest = Body(...),
    vector_store: WeaviateStore = Depends(get_weaviate_store)
):
    """Semantic vector search using Weaviate."""
    try:
        results = vector_store.search(body.q, n_results=body.limit or 5)
        return JSONResponse(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")

@app.post("/search/keywords")
def search_by_keywords(
    body: SearchRequest = Body(...),
    vector_store: WeaviateStore = Depends(get_weaviate_store),
    elasticsearch_store: Optional[ElasticsearchStore] = Depends(get_elasticsearch_store)
):
    """
    Keyword search. Uses Elasticsearch when available, falls back to Weaviate BM25.
    """
    search_query = body.q
    search_limit = body.limit or 5
    try:
        if elasticsearch_store is not None:
            results = elasticsearch_store.search(search_query, n_results=search_limit)
        else:
            results = vector_store.hybrid_search(search_query, n_results=search_limit, alpha=0.0)
        return JSONResponse(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")

@app.post("/search/double")
def search_double(
    body: SearchRequest = Body(...),
    vector_store: WeaviateStore = Depends(get_weaviate_store),
    elasticsearch_store: Optional[ElasticsearchStore] = Depends(get_elasticsearch_store)
):
    """
    Hybrid search combining semantic and keyword results.
    Uses Weaviate+Elasticsearch with RRF fusion when ES is enabled,
    or Weaviate's built-in hybrid search when ES is disabled.
    """
    search_query = body.q
    search_limit = body.limit or 5
    fetch_limit = search_limit * 4
    RRF_K = 60
    try:
        vector_query = generate_hypothetical_document(search_query) if body.use_hyde else search_query
        print(f"Search query: {search_query}")
        print(f"Vector query: {vector_query}")

        if elasticsearch_store is None:
            hybrid_results = vector_store.hybrid_search(vector_query, n_results=fetch_limit)
            final_results = rerank_cross_encoder(search_query, hybrid_results["objects"], top_n=search_limit)
            return JSONResponse({
                "objects": final_results,
                "combined_count": len(final_results),
            })

        weaviate_results = vector_store.search(vector_query, n_results=fetch_limit)
        elasticsearch_results = elasticsearch_store.search(search_query, n_results=fetch_limit)
        rrf_results, combined_results, weaviate_count, elasticsearch_count = combine_search_results_with_rrf(
            weaviate_results, elasticsearch_results, fetch_limit, RRF_K
        )
        final_results = rerank_cross_encoder(search_query, rrf_results, top_n=search_limit)
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
    body: SearchRequestWithContext = Body(...),
    vector_store: WeaviateStore = Depends(get_weaviate_store),
    elasticsearch_store: Optional[ElasticsearchStore] = Depends(get_elasticsearch_store)
):
    """
    Hybrid search with context expansion around matched chunks.
    Groups results by document, expands each hit by N chunks before/after,
    and returns merged text per document.
    """
    search_query = body.q
    search_limit = body.limit or 5
    fetch_limit = search_limit * 4
    context_chunks = body.context_chunks or 3
    RRF_K = 60
    try:
        vector_query = generate_hypothetical_document(search_query) if body.use_hyde else search_query

        if elasticsearch_store is None:
            hybrid_results = vector_store.hybrid_search(vector_query, n_results=fetch_limit)
            final_results = rerank_cross_encoder(search_query, hybrid_results["objects"], top_n=search_limit)
            chunk_store = vector_store
            weaviate_count = len(hybrid_results["objects"])
            elasticsearch_count = 0
            unique_count = len(final_results)
        else:
            weaviate_results = vector_store.search(vector_query, n_results=fetch_limit)
            elasticsearch_results = elasticsearch_store.search(search_query, n_results=fetch_limit)
            rrf_results, combined_results, weaviate_count, elasticsearch_count = combine_search_results_with_rrf(
                weaviate_results, elasticsearch_results, fetch_limit, RRF_K
            )
            final_results = rerank_cross_encoder(search_query, rrf_results, top_n=search_limit)
            chunk_store = elasticsearch_store
            unique_count = len(combined_results)

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

        # For each document, expand selected chunks with N before/after and fetch merged text
        documents_with_context = []
        for file_id, doc_data in documents.items():
            selected_indices = sorted(set(doc_data["selected_chunk_indices"]))
            chunk_count = doc_data["chunk_count"]

            all_chunk_indices = set()
            for chunk_index in selected_indices:
                start_idx = max(0, chunk_index - context_chunks)
                end_idx = min(chunk_count - 1, chunk_index + context_chunks) if chunk_count is not None else chunk_index + context_chunks
                for idx in range(start_idx, end_idx + 1):
                    all_chunk_indices.add(idx)

            augmented_chunk_indices = sorted(all_chunk_indices)
            all_chunks = chunk_store.get_chunks_by_range(file_id, augmented_chunk_indices)
            all_chunks.sort(key=lambda x: x["chunk_index"])
            joined_text = "\n".join([chunk["text"] for chunk in all_chunks])

            documents_with_context.append({
                "file_id": file_id,
                "filename": doc_data["filename"],
                "selected_chunk_indices": selected_indices,
                "augmented_chunk_indices": augmented_chunk_indices,
                "chunk_count": chunk_count,
                "selected_count": len(selected_indices),
                "augmented_count": len(augmented_chunk_indices),
                "chunks": all_chunks,
                "joined_text": joined_text,
                "results": doc_data["results"]
            })

        return JSONResponse({
            "documents": documents_with_context,
            "document_count": len(documents_with_context),
            "weaviate_count": weaviate_count,
            "elasticsearch_count": elasticsearch_count,
            "combined_count": len(final_results),
            "unique_count": unique_count,
            "rrf_k": RRF_K if elasticsearch_store else None,
            "context_chunks": context_chunks
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in double search with context: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
