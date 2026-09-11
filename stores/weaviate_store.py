import uuid
from typing import List
from urllib.parse import urlparse

import weaviate
from sentence_transformers import SentenceTransformer
from weaviate.classes.config import Property, DataType
from weaviate.collections.classes.filters import Filter

from stores.base import BaseStore
from config import EMBEDDING_QUERY_PREFIX, EMBEDDING_PASSAGE_PREFIX


class WeaviateStore(BaseStore):
    def __init__(self, weaviate_url: str, embedding_model: str):
        parsed = urlparse(weaviate_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 8080

        self.client = weaviate.connect_to_local(
            host=host,
            port=port,
            grpc_port=50051
        )

        self.embedding_model = SentenceTransformer(embedding_model)

        self.collection_name = "Document"
        self._ensure_collection()

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_model.encode([EMBEDDING_PASSAGE_PREFIX + t for t in texts]).tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embedding_model.encode([EMBEDDING_QUERY_PREFIX + query]).tolist()[0]

    def _ensure_collection(self):
        if not self.client.collections.exists(self.collection_name):
            self.client.collections.create(
                name=self.collection_name,
                properties=[
                    Property(name="file_id", data_type=DataType.TEXT),
                    Property(name="filename", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="chunk_count", data_type=DataType.INT),
                    Property(name="page_number", data_type=DataType.INT),
                    Property(name="text", data_type=DataType.TEXT),
                ]
            )

    def add_documents(self, file_id: str, text_chunks: List, filename: str = ""):
        collection = self.client.collections.get(self.collection_name)

        chunk_texts = []
        chunk_pages = []
        for chunk in text_chunks:
            if isinstance(chunk, dict):
                chunk_texts.append(chunk["text"])
                chunk_pages.append(chunk.get("page_number", 1))
            else:
                chunk_texts.append(chunk)
                chunk_pages.append(1)

        embeddings = self.embed_passages(chunk_texts)

        chunk_count = len(chunk_texts)
        page_count = len(set(chunk_pages))

        with collection.batch.dynamic() as batch:
            for i, (chunk_text, embedding, page_num) in enumerate(zip(chunk_texts, embeddings, chunk_pages)):
                chunk_id = f"{file_id}_chunk_{i}"
                batch.add_object(
                    properties={
                        "file_id": file_id,
                        "filename": filename,
                        "chunk_index": i,
                        "chunk_count": chunk_count,
                        "page_count": page_count,
                        "page_number": page_num,
                        "text": chunk_text,
                    },
                    vector=embedding,
                    uuid=uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)
                )

    def filename_exists(self, filename: str) -> bool:
        if not filename:
            return False

        collection = self.client.collections.get(self.collection_name)

        results = collection.query.fetch_objects(
            limit=1,
            filters=Filter.by_property("filename").equal(filename)
        )

        return len(results.objects) > 0

    def search(self, query: str, n_results: int = 5):
        collection = self.client.collections.get(self.collection_name)

        query_embedding = self.embed_query(query)

        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=n_results,
            return_metadata=["distance", "certainty"]
        )

        formatted_results = {
            "objects": [
                {
                    "id": str(obj.uuid),
                    "text": obj.properties["text"],
                    "file_id": obj.properties["file_id"],
                    "filename": obj.properties["filename"],
                    "chunk_index": obj.properties["chunk_index"],
                    "chunk_count": obj.properties["chunk_count"],
                    "page_number": obj.properties.get("page_number", 1),
                    "distance": obj.metadata.distance if obj.metadata else None
                }
                for obj in results.objects
            ]
        }

        return formatted_results

    def hybrid_search(
        self,
        keyword_query: str,
        vector_query: str,
        n_results: int = 5,
        alpha: float = 0.5,
    ):
        """
        Weaviate built-in hybrid search: BM25 over `keyword_query`, vector search
        over the embedding of `vector_query`, blended by `alpha`
        (0 = pure BM25, 1 = pure vector).

        The two arguments differ only under HyDE, where BM25 needs the user's
        actual words while the vector half gets the hypothetical passage.
        Without HyDE, callers pass the same string twice.
        """
        collection = self.client.collections.get(self.collection_name)
        query_embedding = self.embed_query(vector_query)
        results = collection.query.hybrid(
            query=keyword_query,
            vector=query_embedding,
            alpha=alpha,
            limit=n_results,
            return_metadata=["score"]
        )
        return {
            "objects": [
                {
                    "id": str(obj.uuid),
                    "text": obj.properties["text"],
                    "file_id": obj.properties["file_id"],
                    "filename": obj.properties["filename"],
                    "chunk_index": obj.properties["chunk_index"],
                    "chunk_count": obj.properties["chunk_count"],
                    "page_number": obj.properties.get("page_number", 1),
                    "score": obj.metadata.score if obj.metadata else None,
                }
                for obj in results.objects
            ]
        }

    def get_all_documents(self):
        collection = self.client.collections.get(self.collection_name)

        results = collection.query.fetch_objects(
            limit=10000,
            filters=Filter.by_property("chunk_index").equal(0)
        )

        documents = [
            {
                "file_id": obj.properties["file_id"],
                "filename": obj.properties.get("filename", ""),
                "chunk_count": obj.properties.get("chunk_count", 0)
            }
            for obj in results.objects
        ]

        return documents

    def delete_document(self, file_id: str) -> int:
        collection = self.client.collections.get(self.collection_name)

        results = collection.query.fetch_objects(
            limit=10000,
            filters=Filter.by_property("file_id").equal(file_id)
        )

        chunk_count = len(results.objects)

        if chunk_count == 0:
            return 0

        collection.data.delete_many(
            where=Filter.by_property("file_id").equal(file_id)
        )

        return chunk_count

    def get_chunks_by_range(self, file_id: str, chunk_indices: List[int]):
        if not chunk_indices:
            return []

        collection = self.client.collections.get(self.collection_name)

        results = collection.query.fetch_objects(
            limit=len(chunk_indices),
            filters=(
                Filter.by_property("file_id").equal(file_id) &
                Filter.by_property("chunk_index").contains_any(list(chunk_indices))
            )
        )

        chunks = [
            {
                "chunk_index": obj.properties["chunk_index"],
                "text": obj.properties["text"],
                "file_id": obj.properties["file_id"],
                "filename": obj.properties["filename"],
                "page_number": obj.properties.get("page_number", 1)
            }
            for obj in results.objects
        ]

        chunks.sort(key=lambda x: x["chunk_index"])
        return chunks

    def close(self):
        if self.client:
            self.client.close()
