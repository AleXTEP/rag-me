import os
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
# E5-family models expect these prefixes; set both to "" for models that don't use them.
EMBEDDING_QUERY_PREFIX = os.getenv("EMBEDDING_QUERY_PREFIX", "query: ")
EMBEDDING_PASSAGE_PREFIX = os.getenv("EMBEDDING_PASSAGE_PREFIX", "passage: ")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "fixed")
EXTRACTION_BACKEND = os.getenv("EXTRACTION_BACKEND", "pymupdf")
USE_ELASTICSEARCH = os.getenv("USE_ELASTICSEARCH", "true").lower() == "true"

# HyDE (Hypothetical Document Embeddings)
HYDE_PROVIDER = os.getenv("HYDE_PROVIDER", "claude")   # claude | openai | local
HYDE_MODEL = os.getenv("HYDE_MODEL", "claude-haiku-4-5-20251001")
HYDE_API_KEY = os.getenv("HYDE_API_KEY", "")           # falls back to env SDK defaults
HYDE_LOCAL_URL = os.getenv("HYDE_LOCAL_URL", "http://localhost:11434")

# OCR (fallback for scanned/image-based PDFs)
OCR_PROVIDER = os.getenv("OCR_PROVIDER", "tesseract")  # tesseract | deepseek | custom
OCR_API_KEY = os.getenv("OCR_API_KEY", "")
OCR_API_URL = os.getenv("OCR_API_URL", "")

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

