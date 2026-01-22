# RAG System

A comprehensive RAG (Retrieval-Augmented Generation) system built with Python, FastAPI, Celery, Redis, Weaviate, and Elasticsearch. This system provides both semantic (vector) and keyword-based search capabilities, with hybrid search using Reciprocal Rank Fusion (RRF) for optimal results.

## Features

- **PDF File Upload**: Upload PDF files via REST API with automatic processing
- **Asynchronous Processing**: Background processing with Celery and Redis
- **Dual Search Capabilities**:
  - **Semantic Search**: Vector-based similarity search using Weaviate
  - **Keyword Search**: Full-text keyword search using Elasticsearch
  - **Hybrid Search**: Combines both search methods using Reciprocal Rank Fusion (RRF)
- **Text Extraction**: Automatic text extraction from PDFs with chunking
- **Text Embedding**: Uses sentence transformers for generating embeddings
- **Document Management**: List and delete documents with metadata
- **Context Retrieval**: Retrieve surrounding chunks for enriched search results

## Architecture

- **FastAPI**: REST API server
- **Celery**: Asynchronous task queue
- **Redis**: Message broker and result backend
- **Weaviate**: Vector database for semantic search
- **Elasticsearch**: Full-text search engine for keyword search
- **Sentence Transformers**: Embedding generation

## Prerequisites

- Docker and Docker Compose

## Quick Start with Docker Compose

The easiest way to run the entire system:

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# View logs for a specific service
docker-compose logs -f api
docker-compose logs -f celery-worker
docker-compose logs -f weaviate
docker-compose logs -f elasticsearch

# Stop all services
docker-compose down

# Stop and remove volumes (clears data)
docker-compose down -v

# Rebuild after code changes
docker-compose up -d --build
```

This will start:
- **Weaviate** on port 8080 (vector database)
- **Elasticsearch** on port 9200 (search engine)
- **Redis** on port 6379 (message broker)
- **API Server** on port 8000
- **Celery Worker** for background processing

The API will be available at `http://localhost:8000`

**Note**: Services may take 30-60 seconds to fully start. Health checks ensure services are ready before the API starts.

## Manual Setup (Local Development)

If you prefer to run services locally:

### Prerequisites

- Python 3.8+
- Redis server running (default: localhost:6379)
- Docker (for running Weaviate and Elasticsearch)

### Installation

1. Clone the repository and navigate to the project directory:
```bash
cd rag2
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file (optional, defaults are provided):
```bash
REDIS_URL=redis://localhost:6379/0
WEAVIATE_URL=http://localhost:8080
ELASTICSEARCH_URL=http://localhost:9200
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
UPLOAD_DIR=./uploads
```

### Running the System

#### 1. Start Weaviate

Start Weaviate using Docker:
```bash
docker run -d \
  --name weaviate \
  -p 8080:8080 \
  -p 50051:50051 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -v weaviate_data:/var/lib/weaviate \
  semitechnologies/weaviate:1.24.0
```

#### 2. Start Elasticsearch

Start Elasticsearch using Docker:
```bash
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e discovery.type=single-node \
  -e xpack.security.enabled=false \
  -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0
```

#### 3. Start Redis

Make sure Redis is running:
```bash 
redis-server
```

#### 4. Start Celery Worker

In a separate terminal:
```bash
celery -A celery_app worker --loglevel=info
```

#### 5. Start the API Server

In another terminal:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check

```bash
GET /health

curl "http://localhost:8000/health"
```

Response:
```json
{
  "status": "healthy"
}
```

### Upload PDF File

Upload a PDF file for processing. The file will be queued for asynchronous processing.

```bash
POST /upload
Content-Type: multipart/form-data

curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_document.pdf"
```

Response:
```json
{
  "status": "queued",
  "file_id": "uuid-here",
  "task_id": "celery-task-id",
  "filename": "your_document.pdf",
  "message": "File uploaded and queued for processing"
}
```

**Note**: If a file with the same filename already exists, you'll receive a 409 Conflict error.

### Check Task Status

Monitor the processing status of an uploaded file.

```bash
GET /task/{task_id}

curl "http://localhost:8000/task/{task_id}"
```

Response (PENDING):
```json
{
  "state": "PENDING",
  "status": "Task is waiting to be processed"
}
```

Response (SUCCESS):
```json
{
  "state": "SUCCESS",
  "result": {
    "file_id": "uuid-here",
    "filename": "your_document.pdf",
    "chunks_processed": 42,
    "status": "completed"
  }
}
```

### List Documents

Get a list of all documents in the system with their metadata.

```bash
GET /documents

curl "http://localhost:8000/documents"
```

Response:
```json
{
  "documents": [
    {
      "file_id": "uuid-here",
      "filename": "document.pdf",
      "chunk_count": 42
    }
  ],
  "count": 1
}
```

### Delete Document

Delete a document and all its chunks by file_id. Removes the document from both Weaviate and Elasticsearch.

```bash
DELETE /documents/{file_id}

curl -X DELETE "http://localhost:8000/documents/{file_id}"
```

Response:
```json
{
  "status": "success",
  "file_id": "uuid-here",
  "weaviate_chunks_deleted": 42,
  "elasticsearch_chunks_deleted": 42,
  "message": "Document 'uuid-here' deleted successfully"
}
```

### Semantic Search

Search for similar documents using vector similarity (semantic search via Weaviate).

```bash
POST /search
Content-Type: application/json

curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "your search query",
    "limit": 5
  }'
```

Request Body:
```json
{
  "q": "your search query",
  "limit": 5  // Optional, default: 5, max: 50
}
```

Response:
```json
{
  "objects": [
    {
      "file_id": "uuid-here",
      "filename": "document.pdf",
      "chunk_index": 5,
      "chunk_count": 42,
      "text": "chunk text content...",
      "distance": 0.123,
      "page_number": 1
    }
  ]
}
```

### Keyword Search

Search for documents by keywords using Elasticsearch full-text search.

```bash
POST /search/keywords
Content-Type: application/json

curl -X POST "http://localhost:8000/search/keywords" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "your keywords",
    "limit": 5
  }'
```

Request Body:
```json
{
  "q": "your keywords",
  "limit": 5  // Optional, default: 5, max: 50
}
```

Response:
```json
{
  "objects": [
    {
      "file_id": "uuid-here",
      "filename": "document.pdf",
      "chunk_index": 5,
      "chunk_count": 42,
      "text": "chunk text content...",
      "score": 0.95,
      "highlight": {
        "text": ["highlighted <em>keywords</em> in context"]
      },
      "page_number": 1
    }
  ]
}
```

### Hybrid Search (Double Search)

Search using both semantic and keyword search, combining results using Reciprocal Rank Fusion (RRF). This provides the best of both worlds - semantic understanding and keyword matching.

```bash
POST /search/double
Content-Type: application/json

curl -X POST "http://localhost:8000/search/double" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "your search query",
    "limit": 5
  }'
```

Request Body:
```json
{
  "q": "your search query",
  "limit": 5  // Optional, default: 5, max: 50
}
```

Response:
```json
{
  "objects": [
    {
      "file_id": "uuid-here",
      "filename": "document.pdf",
      "chunk_index": 5,
      "chunk_count": 42,
      "text": "chunk text content...",
      "sources": ["weaviate", "elasticsearch"],
      "weaviate_rank": 1,
      "elasticsearch_rank": 2,
      "rrf_score": 0.0328,
      "distance": 0.123,
      "score": 0.95,
      "page_number": 1
    }
  ],
  "weaviate_count": 5,
  "elasticsearch_count": 5,
  "combined_count": 8,
  "unique_count": 8,
  "rrf_k": 60
}
```

### Hybrid Search with Context

Similar to hybrid search, but retrieves surrounding chunks for each result to provide more context. Useful for RAG applications where you need complete context around matching chunks.

```bash
POST /search/double/context
Content-Type: application/json

curl -X POST "http://localhost:8000/search/double/context" \
  -H "Content-Type: application/json" \
  -d '{
    "q": "your search query",
    "limit": 5,
    "context_chunks": 5
  }'
```

Request Body:
```json
{
  "q": "your search query",
  "limit": 5,  // Optional, default: 5, max: 50
  "context_chunks": 5  // Optional, default: 5, max: 20 (chunks before/after each result)
}
```

Response:
```json
{
  "objects": [
    {
      "file_id": "uuid-here",
      "filename": "document.pdf",
      "chunk_index": 5,
      "chunk_count": 42,
      "text": "chunk text content...",
      "sources": ["weaviate", "elasticsearch"],
      "weaviate_rank": 1,
      "elasticsearch_rank": 2,
      "rrf_score": 0.0328,
      "original_chunk_index": 5,
      "context_chunks": 5,
      "retrieved_chunk_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      "surrounding_chunks": [
        {
          "chunk_index": 0,
          "text": "chunk 0 text...",
          "page_number": 1
        },
        // ... more chunks
      ],
      "joined_text": "chunk 0 text...\nchunk 1 text...\n...",
      "chunk_count_in_context": 11,
      "page_number": 1
    }
  ],
  "weaviate_count": 5,
  "elasticsearch_count": 5,
  "combined_count": 8,
  "unique_count": 8,
  "rrf_k": 60,
  "context_chunks": 5
}
```

## Project Structure

```
rag2/
├── main.py                  # FastAPI application and endpoints
├── celery_app.py            # Celery configuration
├── tasks.py                 # Celery tasks for PDF processing
├── pdf_processor.py         # PDF text extraction and chunking
├── vector_store.py          # Weaviate vector database operations
├── elasticsearch_store.py   # Elasticsearch operations
├── config.py                # Configuration settings
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker image definition
├── docker-compose.yml       # Docker Compose configuration
├── uploads/                 # Temporary storage for uploaded PDFs
├── tests/                   # Test scripts and sample PDFs
│   ├── test_upload.sh
│   ├── test_query.sh
│   ├── test_query_keywords.sh
│   ├── test_query_double.sh
│   └── TESTING.md
└── README.md                # This file
```

## How It Works

1. **Upload**: Client uploads a PDF file via the `/upload` endpoint
2. **Queue**: File is saved and a Celery task is queued in Redis
3. **Processing**: Celery worker picks up the task and:
   - Extracts text from the PDF using PyPDF2
   - Splits text into chunks (overlapping chunks for better context)
   - Generates embeddings using sentence transformers
   - Stores embeddings in Weaviate vector database
   - Stores text chunks in Elasticsearch for keyword search
4. **Status**: Client can check task status using the task ID
5. **Search**: Multiple search endpoints available:
   - Semantic search via Weaviate
   - Keyword search via Elasticsearch
   - Hybrid search combining both with RRF

## Reciprocal Rank Fusion (RRF)

The hybrid search (`/search/double` and `/search/double/context`) uses Reciprocal Rank Fusion to combine results from both Weaviate and Elasticsearch:

- **RRF Score**: `sum(1 / (k + rank))` for each source where the result appears
- **Default k**: 60 (standard RRF constant)
- Results are sorted by RRF score in descending order
- Duplicate results (same file_id + chunk_index) are merged with combined ranks

This approach ensures that results appearing high in both search results get boosted, while still including unique results from either source.

## Configuration

Configuration is managed via environment variables (with defaults):

- `REDIS_URL`: Redis connection URL (default: `redis://localhost:6379/0`)
- `WEAVIATE_URL`: Weaviate instance URL (default: `http://localhost:8080`)
- `ELASTICSEARCH_URL`: Elasticsearch instance URL (default: `http://localhost:9200`)
- `EMBEDDING_MODEL`: Sentence transformer model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `UPLOAD_DIR`: Directory for temporary file storage (default: `./uploads`)

## Notes

- PDF files are automatically deleted after processing
- The default embedding model is `all-MiniLM-L6-v2` (fast and efficient, ~80MB)
- Weaviate and Elasticsearch run in Docker and persist data in Docker volumes
- Uploaded files are temporarily stored in `./uploads` (or configured `UPLOAD_DIR`)
- Make sure all services (Weaviate, Elasticsearch, Redis) are running before processing files
- Filenames must be unique - duplicate filenames will be rejected with a 409 Conflict error
- The system uses overlapping chunks for better context preservation

## Testing

See `tests/TESTING.md` for detailed testing instructions and example scripts.

## License

[Add your license here]
