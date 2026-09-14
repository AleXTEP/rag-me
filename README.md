# RAG-me

A Retrieval-Augmented Generation backend built with FastAPI, Celery, Weaviate, and Elasticsearch. Supports vector, keyword, and hybrid search with RRF fusion, cross-encoder reranking, HyDE query expansion, and OCR fallback for scanned PDFs.


## Components

| Component | Role |
|---|---|
| FastAPI | REST API server |
| Celery + Redis | Async task queue for PDF ingestion |
| Weaviate | Vector store — semantic search + BM25 fallback |
| Elasticsearch | Full-text keyword search (optional) |
| PyMuPDF (fitz) | PDF text extraction |
| Tesseract (OCR) | Fallback for scanned/image-based PDFs |
| sentence-transformers | Embedding generation (`intfloat/multilingual-e5-base` default: multilingual, 512-token window) |
| CrossEncoder | Reranking (`ms-marco-MiniLM-L-6-v2` default) |
| HyDE | Hypothetical document expansion before vector search |


## Project Structure

```
rag/
├── api/
│   ├── main.py              # Route definitions
│   ├── deps.py              # Store dependency injection
│   └── schemas.py           # Request/response models
├── ingestion/
│   ├── pdf_processor.py     # PDF extraction + chunking pipeline
│   ├── ocr.py               # OCR fallback
│   └── tasks.py             # Celery ingestion task
├── retrieval/
│   ├── fusion.py            # RRF fusion
│   ├── reranker.py          # Cross-encoder reranking
│   └── hyde.py              # HyDE query expansion
├── stores/
│   ├── weaviate_store.py    # Weaviate operations
│   ├── elasticsearch_store.py
│   └── base.py
├── worker/
│   └── celery_app.py        # Celery configuration
├── config.py                # Env-var config
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── tests/
```

#### `ingestion/`
- `pdf_processor.py` — Full PDF pipeline: browser-PDF detection, page extraction, header/footer dedup, text normalization, fixed or semantic chunking with overlap, page offset mapping, OCR fallback.
- `ocr.py` — Tesseract-based OCR for image-only PDFs, with optional API-based providers.
- `tasks.py` — Celery task that runs the ingestion pipeline and writes chunks to Weaviate and Elasticsearch.

#### `retrieval/`
- `fusion.py` — Reciprocal Rank Fusion (RRF).
- `reranker.py` — Cross-encoder reranking. Scores (query, chunk) pairs and returns top-N.
- `hyde.py` — HyDE: generates a 3–5 sentence hypothetical answer via an LLM, embeds that instead of the raw query. Affects only the vector half; keyword matching and reranking stay on the user's words.
- `pipeline.py` — Shared hybrid retrieval used by both `/search/double` endpoints. Selects the retrieval mode, applies HyDE, fuses, reranks.

#### `stores/`
- `weaviate_store.py` — Weaviate client: upsert chunks with embeddings, vector search, alpha-blended hybrid search, chunk range fetch, document delete.
- `elasticsearch_store.py` — ES client: index chunks, BM25 keyword search, chunk range fetch, document delete.
- `base.py` — Shared store interface.

#### `api/`
- `main.py` — Route definitions. Retrieval itself lives in `retrieval/pipeline.py`.
- `deps.py` — FastAPI dependency injection for store singletons.
- `schemas.py` — Pydantic request models.

#### `worker/`
- `celery_app.py` — Celery application configuration (broker: Redis).

<!-- ## Ingestion Pipeline

```
POST /upload

save file to disk
enqueue Celery task
  - extract text per page (PyMuPDF)
  - OCR fallback if no text found
  - strip page numbers, remove repeated headers/footers
  - normalize text (line merging, bullet normalization)
  - chunk: fixed (paragraph → sentence → word) or semantic (embedding similarity)
  - apply overlap (fixed only)
  - map chunks to page numbers via character offsets
  - write to Weaviate (vectors) + Elasticsearch (text)
  - delete temp file
```

## Search Pipeline (hybrid)

```
POST /search/double or /search/double/context

Optional HyDE: query, LLM, hypothetical passage, embed

Elasticsearch enabled: mode "rrf":
  - Weaviate vector search (fetch_limit = limit × 4)
  - Elasticsearch BM25 search (fetch_limit)
  - RRF fusion (k=60) over both result sets

Elasticsearch disabled: mode "weaviate_hybrid":
  - Weaviate built-in hybrid(), vector/BM25 blended by `alpha` in one call

Optional CrossEncoder reranking (always on the raw query), top N

/context variant: group by document, expand ±N chunks, merge text
``` -->

## Configuration

All settings via environment variables (`.env` supported):

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker + result backend |
| `WEAVIATE_URL` | `http://localhost:8080` | Weaviate instance |
| `ELASTICSEARCH_URL` | `http://localhost:9200` | Elasticsearch instance |
| `USE_ELASTICSEARCH` | `true` | Disable to use Weaviate BM25 for keyword search |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | Embedding model (multilingual, 512-token window) |
| `EMBEDDING_QUERY_PREFIX` | `query: ` | Prefix prepended before embedding queries (E5 convention; set empty for other models) |
| `EMBEDDING_PASSAGE_PREFIX` | `passage: ` | Prefix prepended before embedding chunks |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CrossEncoder model |
| `CHUNKING_STRATEGY` | `fixed` | `fixed` or `semantic` |
| `UPLOAD_DIR` | `./uploads` | Temp directory for uploaded PDFs |
| `HYDE_PROVIDER` | `claude` | `claude` \| `openai` \| `local` |
| `HYDE_MODEL` | `claude-haiku-4-5-20251001` | LLM model for HyDE |
| `HYDE_API_KEY` | — | API key (falls back to SDK env defaults) |
| `HYDE_LOCAL_URL` | `http://localhost:11434` | Ollama base URL |
| `OCR_PROVIDER` | `tesseract` | `tesseract` \| `deepseek` \| `custom` |
| `OCR_API_KEY` | — | API key for non-Tesseract OCR |
| `OCR_API_URL` | — | Endpoint for custom OCR provider |

## Quick Start

```bash
docker-compose up -d
```

Services:
- API: `http://localhost:8000`
- Weaviate: `http://localhost:8080`
- Elasticsearch: `http://localhost:9200`
- Redis: `localhost:6379`

Allow 30–60 seconds for health checks before the API accepts requests.

## API Endpoints

#### `GET /health`
Returns `{"status": "healthy"}`.

---

#### `POST /upload`
Upload a PDF for async ingestion.

**Body:** `multipart/form-data`, field `file` (`.pdf` only).

**Response:**
```json
{
  "status": "queued",
  "file_id": "<uuid>",
  "task_id": "<celery-task-id>",
  "filename": "doc.pdf",
  "message": "File uploaded and queued for processing"
}
```

Returns `409` if the filename already exists.

---

#### `GET /task/{task_id}`
Poll ingestion task status.

**States:** `PENDING` | `PROGRESS` | `SUCCESS` | `FAILURE`

**Response (SUCCESS):**
```json
{
  "state": "SUCCESS",
  "status": "success",
  "result": {
    "file_id": "<uuid>",
    "filename": "doc.pdf",
    "chunks_processed": 42
  }
}
```

---

#### `GET /documents`
List all indexed documents with chunk counts.

```json
{
  "documents": [{"file_id": "<uuid>", "filename": "doc.pdf", "chunk_count": 42}],
  "count": 1
}
```

---

#### `DELETE /documents/{file_id}`
Remove a document from Weaviate and Elasticsearch.

```json
{
  "status": "success",
  "file_id": "<uuid>",
  "weaviate_chunks_deleted": 42,
  "elasticsearch_chunks_deleted": 42
}
```

---

#### `POST /search/vector`
Pure vector similarity search via Weaviate.

**Body:**
```json
{"q": "query text", "limit": 5}
```

**Response:** `{"objects": [{...chunk fields, "distance": 0.12}]}`

---

#### `POST /search/keywords`
BM25 keyword search. Uses Elasticsearch when enabled, falls back to Weaviate BM25.

**Body:**
```json
{"q": "query text", "limit": 5}
```

**Response:** `{"objects": [{...chunk fields, "score": 0.95}]}`

---

#### `POST /search/double`
Hybrid search: vector + keyword → RRF fusion → cross-encoder rerank.

Supports `use_hyde: true` to expand the query via HyDE before vector search, and
`rerank: false` to skip the cross-encoder. `alpha` applies only when Elasticsearch is
disabled and is ignored in RRF mode.

**Body:**
```json
{"q": "query text", "limit": 5, "use_hyde": false, "rerank": true, "alpha": 0.5}
```

**Response:**
```json
{
  "objects": [{
    "file_id": "<uuid>",
    "filename": "doc.pdf",
    "chunk_index": 5,
    "text": "...",
    "rerank_score": 4.21,
    "rrf_score": 0.033
  }],
  "retrieval_mode": "rrf",
  "vector_count": 20,
  "keyword_count": 20,
  "unique_count": 31,
  "combined_count": 5,
  "rrf_k": 60
}
```

`retrieval_mode` names the strategy that actually ran, so a result is self-describing.
When Elasticsearch is disabled the same endpoint returns the alpha-blend diagnostics instead:

```json
{
  "objects": [{"...": "...", "rerank_score": 4.21, "score": 0.87}],
  "retrieval_mode": "weaviate_hybrid",
  "alpha": 0.5,
  "candidate_count": 20,
  "combined_count": 5
}
```

---

#### `POST /search/double/context`
Hybrid search with chunk-window expansion, grouped by document.

Runs the same retrieval pipeline as `/search/double`, then for each matched document: collects all hit chunk indices, expands each by `±context_chunks`, fetches the full range from the store, and joins the text.

**Body:**
```json
{"q": "query text", "limit": 5, "context_chunks": 3, "use_hyde": false, "rerank": true}
```

**Response:**
```json
{
  "documents": [{
    "file_id": "<uuid>",
    "filename": "doc.pdf",
    "selected_chunk_indices": [5, 8],
    "augmented_chunk_indices": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "chunk_count": 42,
    "selected_count": 2,
    "augmented_count": 10,
    "chunks": [{...}],
    "joined_text": "full merged context string",
    "results": [{...reranked chunk hits}]
  }],
  "document_count": 1,
  "retrieval_mode": "rrf",
  "vector_count": 20,
  "keyword_count": 20,
  "unique_count": 31,
  "combined_count": 5,
  "rrf_k": 60,
  "context_chunks": 3
}
```

