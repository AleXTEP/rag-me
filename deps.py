"""FastAPI dependencies for store injection."""

from typing import Optional

from config import WEAVIATE_URL, ELASTICSEARCH_URL, EMBEDDING_MODEL, USE_ELASTICSEARCH
from stores import get_store, WeaviateStore, ElasticsearchStore


def get_weaviate_store() -> WeaviateStore:
    return get_store("weaviate", weaviate_url=WEAVIATE_URL, embedding_model=EMBEDDING_MODEL)


def get_elasticsearch_store() -> Optional[ElasticsearchStore]:
    if not USE_ELASTICSEARCH:
        return None
    return get_store("elasticsearch", elasticsearch_url=ELASTICSEARCH_URL)
