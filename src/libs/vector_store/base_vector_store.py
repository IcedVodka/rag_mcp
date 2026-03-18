#!/usr/bin/env python3
"""
Base Vector Store - Abstract Interface for Vector Databases

Defines the contract for vector storage implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any

from core.trace.trace_context import TraceContext


@dataclass
class VectorRecord:
    """A record to be stored in the vector database."""
    id: str
    vector: list[float]
    text: str
    metadata: dict[str, Any]


@dataclass
class QueryResult:
    """Result from a vector query."""
    id: str
    score: float  # Distance/similarity score
    text: str
    metadata: dict[str, Any]


class BaseVectorStore(ABC):
    """
    Abstract base class for vector store implementations.
    
    All vector stores (Chroma, Qdrant, etc.) must implement this interface.
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the vector store with configuration.
        
        Args:
            config: Provider-specific configuration
        """
        self.config = config
        self.collection_name = config.get("collection_name", "default")
    
    @abstractmethod
    def upsert(
        self,
        records: list[VectorRecord],
        collection: Optional[str] = None,
        trace: Optional[TraceContext] = None
    ) -> None:
        """
        Upsert vectors into the store.
        
        Args:
            records: List of vector records to upsert
            collection: Optional collection name (for multi-tenant stores)
            trace: Optional trace context
            
        Raises:
            VectorStoreError: On storage errors
        """
        pass
    
    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
        trace: Optional[TraceContext] = None
    ) -> list[QueryResult]:
        """
        Query for similar vectors.
        
        Args:
            vector: Query vector
            top_k: Number of results to return
            filters: Optional metadata filters
            trace: Optional trace context
            
        Returns:
            List of query results sorted by relevance
        """
        pass
    
    @abstractmethod
    def delete(
        self,
        ids: list[str],
        trace: Optional[TraceContext] = None
    ) -> None:
        """
        Delete vectors by ID.
        
        Args:
            ids: List of record IDs to delete
            trace: Optional trace context
        """
        pass
    
    @abstractmethod
    def get_by_ids(
        self,
        ids: list[str],
        trace: Optional[TraceContext] = None
    ) -> list[dict[str, Any]]:
        """
        Get records by their IDs.
        
        Args:
            ids: List of record IDs
            trace: Optional trace context
            
        Returns:
            List of records (id, text, metadata)
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Return the total number of vectors in the store."""
        pass


class VectorStoreError(Exception):
    """Base exception for vector store errors."""
    pass


class VectorStoreConnectionError(VectorStoreError):
    """Raised when connection to vector store fails."""
    pass


class VectorStoreNotFoundError(VectorStoreError):
    """Raised when queried data is not found."""
    pass
