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
                        "text": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "text_preview": {
                            "type": "text"
                        }
                    }
                }
            }
            # Elasticsearch 8.x uses mappings parameter directly
            self.client.indices.create(index=self.index_name, mappings=mapping["mappings"])
    
    def add_documents(self, file_id: str, text_chunks: List[str], filename: str = ""):
        """
        Add documents to Elasticsearch.
        
        Args:
            file_id: Unique identifier for the file
            text_chunks: List of text chunks to store
            filename: Original filename of the document
        """
        chunk_count = len(text_chunks)
        
        # Prepare bulk operations
        actions = []
        for i, chunk in enumerate(text_chunks):
            doc = {
                "_index": self.index_name,
                "_id": f"{file_id}_chunk_{i}",
                "_source": {
                    "file_id": file_id,
                    "filename": filename,
                    "chunk_index": i,
                    "chunk_count": chunk_count,
                    "text": chunk,
                    "text_preview": chunk[:200]
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
                                "fields": ["text^2", "text_preview", "filename"],
                                "type": "best_fields",
                                "fuzziness": "AUTO"
                            }
                        }
                    ]
                }
            },
            "size": n_results,
            "_source": ["file_id", "filename", "chunk_index", "chunk_count", "text", "text_preview"],
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
                                "_source": ["filename", "chunk_count"]
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
            documents.append({
                "file_id": file_id,
                "filename": first_chunk.get("filename", ""),
                "chunk_count": first_chunk.get("chunk_count", 0)
            })
        
        return documents
    
    def close(self):
        """Close the Elasticsearch connection."""
        if self.client:
            self.client.close()

