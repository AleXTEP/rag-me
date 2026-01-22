from elasticsearch import Elasticsearch
from typing import List, Dict, Optional
import uuid

class ElasticsearchStore:
    def __init__(self, elasticsearch_url: str):
        """
        Initialize Elasticsearch store for keyword search.
        
        Args:
            elasticsearch_url: URL to Elasticsearch instance (e.g., http://localhost:9200)
        """
        self.client = Elasticsearch(
            [elasticsearch_url],
            request_timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )
        self.index_name = "documents"
        self._ensure_index()
    
    def _ensure_index(self):
        """Create index with mapping if it doesn't exist."""
        if not self.client.indices.exists(index=self.index_name):
            # Define mapping for better search performance
            mapping = {
                "mappings": {
                    "properties": {
                        "file_id": {
                            "type": "keyword"
                        },
                        "filename": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword"
                                }
                            }
                        },
                        "chunk_index": {
                            "type": "integer"
                        },
                        "chunk_count": {
                            "type": "integer"
                        },
                        "page_number": {
                            "type": "integer"
                        },
                        "text": {
                            "type": "text",
                            "analyzer": "standard"
                        }
                    }
                }
            }
            # Elasticsearch 8.x uses mappings parameter directly
            self.client.indices.create(index=self.index_name, mappings=mapping["mappings"])
    
    def add_documents(self, file_id: str, text_chunks: List, filename: str = ""):
        """
        Add documents to Elasticsearch.
        
        Args:
            file_id: Unique identifier for the file
            text_chunks: List of text chunks (dicts with 'text' and 'page_number' keys, or strings for backward compatibility)
            filename: Original filename of the document
        """
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
        
        chunk_count = len(chunk_texts)
        page_count = len(set(chunk_pages))
        
        # Prepare bulk operations
        actions = []
        for i, (chunk_text, page_num) in enumerate(zip(chunk_texts, chunk_pages)):
            doc = {
                "_index": self.index_name,
                "_id": f"{file_id}_chunk_{i}",
                "_source": {
                    "file_id": file_id,
                    "filename": filename,
                    "chunk_index": i,
                    "chunk_count": chunk_count,
                    "page_number": page_num,
                    "page_count": page_count,
                    "text": chunk_text,
                }
            }
            actions.append(doc)
        
        # Bulk index documents
        if actions:
            from elasticsearch.helpers import bulk
            bulk(self.client, actions)
            # Refresh index to make documents searchable immediately
            self.client.indices.refresh(index=self.index_name)
    
    def search(self, query: str, n_results: int = 5, filename: Optional[str] = None):
        """
        Search for documents by keywords.
        
        Args:
            query: Search query (keywords)
            n_results: Number of results to return
            filename: Optional filename filter
        
        Returns:
            Search results
        """
        # Build search query
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["text^2", "filename"],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        }
                    ]
                }
            },
            "size": n_results,
            "_source": ["file_id", "filename", "chunk_index", "chunk_count", "page_number", "text"],
            "highlight": {
                "fields": {
                    "text": {
                        "fragment_size": 150,
                        "number_of_fragments": 1
                    }
                }
            }
        }
        
        # Add filename filter if provided
        if filename:
            search_body["query"]["bool"]["filter"] = [
                {"term": {"filename.keyword": filename}}
            ]
        
        # Execute search
        response = self.client.search(index=self.index_name, body=search_body)
        
        # Format results
        formatted_results = {
            "objects": []
        }
        
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            result = {
                "id": hit["_id"],
                "text": source["text"],
                "file_id": source["file_id"],
                "filename": source["filename"],
                "chunk_index": source["chunk_index"],
                "chunk_count": source["chunk_count"],
                "page_number": source.get("page_number", 1),
                "score": hit["_score"]
            }
            
            # Add highlighted text if available
            if "highlight" in hit and "text" in hit["highlight"]:
                result["highlight"] = hit["highlight"]["text"][0]
            
            formatted_results["objects"].append(result)
        
        return formatted_results
    
    def filename_exists(self, filename: str) -> bool:
        """
        Check if a filename already exists in Elasticsearch.
        
        Args:
            filename: Filename to check
        
        Returns:
            True if filename exists, False otherwise
        """
        if not filename:
            return False
        
        search_body = {
            "query": {
                "term": {
                    "filename.keyword": filename
                }
            },
            "size": 1
        }
        
        response = self.client.search(index=self.index_name, body=search_body)
        return response["hits"]["total"]["value"] > 0
    
    def get_all_documents(self):
        """
        Get a list of all unique documents with their metadata.
        
        Returns:
            List of documents with file_id, filename, and chunk_count
        """
        # Use aggregation to get unique documents
        search_body = {
            "size": 0,
            "aggs": {
                "unique_docs": {
                    "terms": {
                        "field": "file_id",
                        "size": 10000
                    },
                    "aggs": {
                        "first_chunk": {
                            "top_hits": {
                                "size": 1,
                                "sort": [{"chunk_index": {"order": "asc"}}],
                                "_source": ["filename", "chunk_count", "page_count"]
                            }
                        }
                    }
                }
            }
        }
        
        response = self.client.search(index=self.index_name, body=search_body)
        documents = []
        for bucket in response["aggregations"]["unique_docs"]["buckets"]:
            file_id = bucket["key"]
            first_chunk = bucket["first_chunk"]["hits"]["hits"][0]["_source"]
            print(first_chunk)
            documents.append({
                "file_id": file_id,
                "filename": first_chunk.get("filename", ""),
                "chunk_count": first_chunk.get("chunk_count", 0),
                "page_count": first_chunk.get("page_count", 0)
            })
        
        return documents
    
    def delete_document(self, file_id: str) -> int:
        """
        Delete all chunks for a given document by file_id.
        
        Args:
            file_id: Unique identifier for the file to delete
        
        Returns:
            Number of chunks deleted
        """
        # Use delete_by_query to delete all documents with matching file_id
        delete_body = {
            "query": {
                "term": {
                    "file_id": file_id
                }
            }
        }
        
        response = self.client.delete_by_query(
            index=self.index_name,
            body=delete_body,
            refresh=True  # Refresh index immediately after deletion
        )
        
        return response.get("deleted", 0)
    
    def get_chunks_by_range(self, file_id: str, chunk_indices: List[int]):
        """
        Get chunks by file_id and list of chunk indices.
        
        Args:
            file_id: Unique identifier for the file
            chunk_indices: List of chunk indices to retrieve
        
        Returns:
            List of chunk objects sorted by chunk_index
        """
        if not chunk_indices:
            return []
        
        # Build query to get chunks by file_id and chunk_index
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"file_id": file_id}},
                        {"terms": {"chunk_index": chunk_indices}}
                    ]
                }
            },
            "size": len(chunk_indices),
            "_source": ["file_id", "filename", "chunk_index", "chunk_count", "page_number", "text"],
            "sort": [{"chunk_index": {"order": "asc"}}]
        }
        
        response = self.client.search(index=self.index_name, body=search_body)
        
        # Format results
        chunks = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            chunks.append({
                "chunk_index": source["chunk_index"],
                "text": source["text"],
                "file_id": source["file_id"],
                "filename": source["filename"],
                "page_number": source.get("page_number", 1)
            })
        
        return chunks
    
    def close(self):
        """Close the Elasticsearch connection."""
        if self.client:
            self.client.close()

