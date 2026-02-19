from stores.base import BaseStore
from stores.weaviate_store import WeaviateStore
from stores.elasticsearch_store import ElasticsearchStore

_instances: dict[str, BaseStore] = {}


def get_store(store_type: str, **kwargs) -> BaseStore:
    """
    Get or create a singleton store instance.

    Args:
        store_type: "weaviate" or "elasticsearch"
        **kwargs: Constructor arguments for the store

    Returns:
        Singleton BaseStore instance
    """
    if store_type not in _instances:
        if store_type == "weaviate":
            _instances[store_type] = WeaviateStore(**kwargs)
        elif store_type == "elasticsearch":
            _instances[store_type] = ElasticsearchStore(**kwargs)
        else:
            raise ValueError(f"Unknown store type: {store_type}")
    return _instances[store_type]


def close_all():
    """Close all store instances."""
    for store in _instances.values():
        store.close()
    _instances.clear()
