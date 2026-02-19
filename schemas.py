from pydantic import BaseModel, Field
from typing import Optional


class SearchRequest(BaseModel):
    q: str = Field(..., description="Search query")
    limit: Optional[int] = Field(5, ge=1, le=50, description="Number of results to return")


class SearchRequestWithContext(BaseModel):
    q: str = Field(..., description="Search query")
    limit: Optional[int] = Field(5, ge=1, le=50, description="Number of results to return")
    context_chunks: Optional[int] = Field(3, ge=0, le=20, description="Number of chunks before and after each result to include")
