# RAG System

A simple RAG (Retrieval-Augmented Generation) system built with Python, FastAPI, Celery, Redis, and Weaviate.

## Features

- PDF file upload via REST API
- Asynchronous processing with Celery and Redis
- Text extraction from PDFs
- Text embedding using sentence transformers
- Vector storage with Weaviate

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

# Stop all services
docker-compose down

# Stop and remove volumes (clears data)
docker-compose down -v

# Rebuild after code changes
docker-compose up -d --build
```

This will start:
- **Weaviate** on port 8080
- **Redis** on port 6379
- **API Server** on port 8000
- **Celery Worker** for background processing

The API will be available at `http://localhost:8000`

## Manual Setup (Local Development)

If you prefer to run services locally:

### Prerequisites

- Python 3.8+
- Redis server running (default: localhost:6379)
- Docker (for running Weaviate)

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
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
UPLOAD_DIR=./uploads
```

### Running the System

### 1. Start Weaviate

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
  semitechnologies/weaviate:latest
```

Or using Docker Compose (create `docker-compose.yml`):
```yaml
version: '3.8'
services:
  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"
      - "50051:50051"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: /var/lib/weaviate
    volumes:
      - weaviate_data:/var/lib/weaviate

volumes:
  weaviate_data:
```

Then run:
```bash
docker-compose up -d
```

### 2. Start Redis

Make sure Redis is running:
```bash
redis-server
```

### 3. Start Celery Worker

In a separate terminal:
```bash
celery -A celery_app worker --loglevel=info
```

### 4. Start the API Server

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

### Upload PDF File
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

### Check Task Status
```bash
GET /task/{task_id}

curl "http://localhost:8000/task/{task_id}"
```

### Health Check
```bash
GET /health

curl "http://localhost:8000/health"
```

## Project Structure

```
rag2/
├── main.py              # FastAPI application
├── celery_app.py        # Celery configuration
├── tasks.py             # Celery tasks
├── pdf_processor.py     # PDF text extraction
├── vector_store.py      # Vector database operations
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker image definition
├── docker-compose.yml   # Docker Compose configuration
├── .dockerignore        # Files to exclude from Docker build
└── README.md           # This file
```

## How It Works

1. **Upload**: Client uploads a PDF file via the `/upload` endpoint
2. **Queue**: File is saved and a Celery task is queued in Redis
3. **Processing**: Celery worker picks up the task and:
   - Extracts text from the PDF
   - Splits text into chunks
   - Generates embeddings using sentence transformers
   - Stores embeddings in Weaviate vector database
4. **Status**: Client can check task status using the task ID

## Notes

- PDF files are automatically deleted after processing
- The default embedding model is `all-MiniLM-L6-v2` (fast and efficient)
- Weaviate runs in Docker and persists data in a Docker volume
- Uploaded files are temporarily stored in `./uploads`
- Make sure Weaviate is running before processing files
