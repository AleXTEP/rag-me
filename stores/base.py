from abc import ABC, abstractmethod
from typing import List


class BaseStore(ABC):
    """Abstract base class defining the shared interface for all document stores."""

    @abstractmethod
    def add_documents(self, file_id: str, text_chunks: List, filename: str = ""):
        ...

    @abstractmethod
    def search(self, query: str, n_results: int = 5) -> dict:
        ...

    @abstractmethod
    def delete_document(self, file_id: str) -> int:
        ...

    @abstractmethod
    def get_chunks_by_range(self, file_id: str, chunk_indices: List[int]) -> list:
        ...

    @abstractmethod
    def get_all_documents(self) -> list:
        ...

    @abstractmethod
    def filename_exists(self, filename: str) -> bool:
        ...

    @abstractmethod
    def close(self):
        ...
