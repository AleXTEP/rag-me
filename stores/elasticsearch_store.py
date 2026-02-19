from typing import List, Optional

from elasticsearch import Elasticsearch

from stores.base import BaseStore


class ElasticsearchStore(BaseStore):
    def __init__(self, elasticsearch_url: str):
        self.client = Elasticsearch(
            [elasticsearch_url],
            request_timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )
        self.index_name = "documents"
        self._ensure_index()

    def _ensure_index(self):
        if not self.client.indices.exists(index=self.index_name):
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
            self.client.indices.create(index=self.index_name, mappings=mapping["mappings"])

    def add_documents(self, file_id: str, text_chunks: List, filename: str = ""):
        chunk_texts = []
        chunk_pages = []
        for chunk in text_chunks:
            if isinstance(chunk, dict):
                chunk_texts.append(chunk["text"])
                chunk_pages.append(chunk.get("page_number", 1))
            else:
                chunk_texts.append(chunk)
                chunk_pages.append(1)

        chunk_count = len(chunk_texts)
        page_count = len(set(chunk_pages))

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

        if actions:
            from elasticsearch.helpers import bulk
            bulk(self.client, actions)
            self.client.indices.refresh(index=self.index_name)

    def search(self, query: str, n_results: int = 5, filename: Optional[str] = None):
        bool_query = {
            "should": [
                {
                    "match_phrase": {
                        "text": {
                            "query": query.strip(),
                            "boost": 3
                        }
                    }
                },
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["text^2", "filename"],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                }
            ],
            "minimum_should_match": 1
        }

        search_body = {
            "query": {"bool": bool_query},
            "size": n_results,
            "_source": ["file_id", "filename", "chunk_index", "chunk_count", "page_number", "text"]
        }

        if filename:
            bool_query["filter"] = [
                {"term": {"filename.keyword": filename}}
            ]

        response = self.client.search(index=self.index_name, body=search_body)

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
            formatted_results["objects"].append(result)

        return formatted_results

    def filename_exists(self, filename: str) -> bool:
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
            documents.append({
                "file_id": file_id,
                "filename": first_chunk.get("filename", ""),
                "chunk_count": first_chunk.get("chunk_count", 0),
                "page_count": first_chunk.get("page_count", 0)
            })

        return documents

    def delete_document(self, file_id: str) -> int:
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
            refresh=True
        )

        return response.get("deleted", 0)

    def get_chunks_by_range(self, file_id: str, chunk_indices: List[int]):
        if not chunk_indices:
            return []

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
        if self.client:
            self.client.close()
