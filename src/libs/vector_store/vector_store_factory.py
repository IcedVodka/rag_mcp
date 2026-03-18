#!/usr/bin/env python3
"""
Vector Store Factory - Provider-Agnostic Vector Store Creation

Creates appropriate vector store instances based on configuration.
"""

from typing import Optional, Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from .base_vector_store import BaseVectorStore, VectorRecord, QueryResult


class VectorStoreFactory:
    """
    Factory for creating vector store instances.
    
    Routes to appropriate provider implementation based on settings.
    """
    
    @staticmethod
    def create(settings: Settings) -> BaseVectorStore:
        """
        Create a vector store instance based on settings.
        
        Args:
            settings: Application settings
            
        Returns:
            Configured vector store instance
            
        Raises:
            ValueError: If provider is unknown
        """
        provider = settings.vector_store.provider
        config = getattr(settings.vector_store, provider, {})
        
        if provider == "chroma":
            from .chroma_store import ChromaStore
            return ChromaStore(config)
        elif provider == "qdrant":
            from .qdrant_store import QdrantStore
            return QdrantStore(config)
        else:
            raise ValueError(f"Unknown vector store provider: {provider}")


class VectorStoreProvider:
    """
    Wrapper for vector store with common functionality.
    
    Provides trace recording and batch operations.
    """
    
    def __init__(self, store: BaseVectorStore) -> None:
        self.store = store
    
    def upsert_with_trace(
        self,
        records: list[VectorRecord],
        trace: Optional[TraceContext] = None
    ) -> None:
        """
        Upsert with automatic trace recording.
        
        Args:
            records: Records to upsert
            trace: Optional trace context
        """
        if trace:
            trace.record_stage(
                name="vector_upsert",
                provider=self.store.__class__.__name__,
                details={
                    "collection": self.store.collection_name,
                    "record_count": len(records)
                }
            )
        
        return self.store.upsert(records, trace)
    
    def query_with_trace(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
        trace: Optional[TraceContext] = None
    ) -> list[QueryResult]:
        """
        Query with automatic trace recording.
        
        Args:
            vector: Query vector
            top_k: Number of results
            filters: Optional filters
            trace: Optional trace context
            
        Returns:
            Query results
        """
        results = self.store.query(vector, top_k, filters, trace)
        
        if trace:
            trace.record_stage(
                name="vector_query",
                provider=self.store.__class__.__name__,
                details={
                    "collection": self.store.collection_name,
                    "top_k": top_k,
                    "result_count": len(results),
                    "filters": filters
                }
            )
        
        return results
