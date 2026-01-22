import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.collections.classes.filters import Filter
from sentence_transformers import SentenceTransformer
from typing import List
import uuid
from urllib.parse import urlparse

# Module-level singleton instance
_vector_store_instance = None

def get_vector_store(weaviate_url: str, embedding_model: str) -> 'VectorStore':
    """
    Get or create a singleton VectorStore instance.
    This ensures the embedding model is only loaded once per process.
    
    Args:
        weaviate_url: URL to Weaviate instance
        embedding_model: Name of the sentence transformer model
    
    Returns:
        VectorStore instance (singleton)
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore(weaviate_url, embedding_model)
    return _vector_store_instance

class VectorStore:
    def __init__(self, weaviate_url: str, embedding_model: str):
        """
        Initialize vector store with Weaviate and embedding model.
        
        Args:
            weaviate_url: URL to Weaviate instance (e.g., http://localhost:8080)
            embedding_model: Name of the sentence transformer model
        """
        # Parse URL
        parsed = urlparse(weaviate_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 8080
        
        # Initialize Weaviate client
        self.client = weaviate.connect_to_local(
            host=host,
            port=port,
            grpc_port=50051
        )
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Get or create collection
        self.collection_name = "Document"
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        if not self.client.collections.exists(self.collection_name):
            # Create collection without vectorizer (we provide our own vectors)
            # Weaviate will infer vector dimensions from the first vector we add
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
        """
        Add documents to vector store.
        
        Args:
            file_id: Unique identifier for the file
            text_chunks: List of text chunks (dicts with 'text' and 'page_number' keys, or strings for backward compatibility)
            filename: Original filename of the document
        """
        collection = self.client.collections.get(self.collection_name)
        
        # Extract text from chunks (handle both dict and string formats)
        chunk_texts = []
        chunk_pages = []
        for chunk in text_chunks:
            if isinstance(chunk, dict):
                chunk_texts.append(chunk["text"])
                chunk_pages.append(chunk.get("page_number", 1))
            else:
                # Backward compatibility: treat as string
                chunk_texts.append(chunk)
                chunk_pages.append(1)
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunk_texts).tolist()
        
        # Get total chunk count
        chunk_count = len(chunk_texts)
        page_count = len(set(chunk_pages))
        
        # Prepare data objects
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
        """
        Check if a filename already exists in the vector store.
        
        Args:
            filename: Filename to check
        
        Returns:
            True if filename exists, False otherwise
        """
        if not filename:
            return False
        
        collection = self.client.collections.get(self.collection_name)
        
        # Query for documents with matching filename using Weaviate v4 filter syntax
        results = collection.query.fetch_objects(
            limit=1,
            filters=Filter.by_property("filename").equal(filename)
        )
        
        return len(results.objects) > 0
    
    def search(self, query: str, n_results: int = 5):
        """
        Search for similar documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
        
        Returns:
            Search results
        """
        collection = self.client.collections.get(self.collection_name)
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search
        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=n_results,
            return_metadata=["distance", "certainty"]
        )
        
        # Format results
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
    
    def get_all_documents(self):
        """
        Get a list of all unique documents with their metadata.
        Efficiently fetches only the first chunk (chunk_index=0) of each document.
        
        Returns:
            List of documents with file_id, filename, and chunk_count
        """
        collection = self.client.collections.get(self.collection_name)
        
        # Fetch only chunks with chunk_index=0 (first chunk of each document)
        # This gives us one chunk per document, much more efficient than fetching all chunks
        results = collection.query.fetch_objects(
            limit=10000,  # Adjust if you have more than 10k documents
            filters=Filter.by_property("chunk_index").equal(0)
        )
        
        # Extract document metadata from the first chunk of each document
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
        """
        Delete all chunks for a given document by file_id.
        
        Args:
            file_id: Unique identifier for the file to delete
        
        Returns:
            Number of chunks deleted
        """
        collection = self.client.collections.get(self.collection_name)
        
        # First, count how many chunks exist for this file_id
        results = collection.query.fetch_objects(
            limit=10000,  # Adjust if you have more than 10k chunks per document
            filters=Filter.by_property("file_id").equal(file_id)
        )
        
        chunk_count = len(results.objects)
        
        if chunk_count == 0:
            return 0
        
        # Delete all chunks using delete_by with filter
        # This is more efficient than deleting by UUIDs
        collection.data.delete_many(
            where=Filter.by_property("file_id").equal(file_id)
        )
        
        return chunk_count
    
    def get_chunks_by_range(self, file_id: str, chunk_indices: List[int]):
        """
        Get chunks by file_id and list of chunk indices.
        
        Args:
            file_id: Unique identifier for the file
            chunk_indices: List of chunk indices to retrieve
        
        Returns:
            List of chunk objects sorted by chunk_index
        """
        collection = self.client.collections.get(self.collection_name)
        
        # Weaviate doesn't support IN queries directly, so we'll fetch all chunks for the file
        # and filter in Python (or use multiple queries)
        # For efficiency, we'll fetch all chunks for the file_id and filter
        results = collection.query.fetch_objects(
            limit=10000,  # Adjust if you have more than 10k chunks per document
            filters=Filter.by_property("file_id").equal(file_id)
        )
        
        # Filter by chunk_indices and format
        chunks = []
        for obj in results.objects:
            chunk_idx = obj.properties["chunk_index"]
            if chunk_idx in chunk_indices:
                chunks.append({
                    "chunk_index": chunk_idx,
                    "text": obj.properties["text"],
                    "file_id": obj.properties["file_id"],
                    "filename": obj.properties["filename"],
                    "page_number": obj.properties.get("page_number", 1)
                })
        
        # Sort by chunk_index
        chunks.sort(key=lambda x: x["chunk_index"])
        return chunks
    
    def close(self):
        """Close the Weaviate connection."""
        if self.client:
            self.client.close()
