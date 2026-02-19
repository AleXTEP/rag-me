"""FastAPI dependencies for store injection."""

from config import WEAVIATE_URL, ELASTICSEARCH_URL, EMBEDDING_MODEL
from stores import get_store, WeaviateStore, ElasticsearchStore


def get_weaviate_store() -> WeaviateStore:
    return get_store("weaviate", weaviate_url=WEAVIATE_URL, embedding_model=EMBEDDING_MODEL)


def get_elasticsearch_store() -> ElasticsearchStore:
    return get_store("elasticsearch", elasticsearch_url=ELASTICSEARCH_URL)
