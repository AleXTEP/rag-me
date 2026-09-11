from pydantic import BaseModel, Field
from typing import Optional


class SearchRequest(BaseModel):
    q: str = Field(..., description="Search query")
    limit: Optional[int] = Field(5, ge=1, le=50, description="Number of results to return")
    use_hyde: Optional[bool] = Field(False, description="Use HyDE: generate a hypothetical answer and embed it instead of the raw query")
    rerank: Optional[bool] = Field(True, description="Apply cross-encoder reranking to the fused results")
    alpha: Optional[float] = Field(0.5, ge=0, le=1, description="Weaviate hybrid blend, used ONLY when Elasticsearch is disabled: 0 = pure keyword (BM25), 1 = pure vector. Ignored in RRF mode.")


class SearchRequestWithContext(BaseModel):
    q: str = Field(..., description="Search query")
    limit: Optional[int] = Field(5, ge=1, le=50, description="Number of results to return")
    context_chunks: Optional[int] = Field(3, ge=0, le=20, description="Number of chunks before and after each result to include")
    use_hyde: Optional[bool] = Field(False, description="Use HyDE: generate a hypothetical answer and embed it instead of the raw query")
    rerank: Optional[bool] = Field(True, description="Apply cross-encoder reranking to the fused results")
    alpha: Optional[float] = Field(0.5, ge=0, le=1, description="Weaviate hybrid blend, used ONLY when Elasticsearch is disabled: 0 = pure keyword (BM25), 1 = pure vector. Ignored in RRF mode.")
