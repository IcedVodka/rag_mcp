#!/usr/bin/env python3
"""
Dense Retriever - Semantic Search via Vector Similarity

Combines EmbeddingClient (query vectorization) with VectorStore (vector retrieval)
to perform semantic recall of relevant chunks.
"""

from typing import Optional, Any, List

from core.settings import Settings
from core.types import RetrievalResult
from core.trace.trace_context import TraceContext
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.base_vector_store import BaseVectorStore, QueryResult
from libs.vector_store.vector_store_factory import VectorStoreFactory


class DenseRetriever:
    """
    Dense retriever for semantic search using vector similarity.
    
    Orchestrates the retrieval pipeline:
    1. Convert query text to embedding vector using EmbeddingClient
    2. Query VectorStore for similar vectors
    3. Transform results to RetrievalResult format
    
    Supports dependency injection for testing via embedding_client and vector_store
    parameters in __init__.
    """
    
    def __init__(
        self,
        settings: Settings,
        embedding_client: Optional[BaseEmbedding] = None,
        vector_store: Optional[BaseVectorStore] = None
    ) -> None:
        """
        Initialize the dense retriever.
        
        Args:
            settings: Application settings for creating embedding and vector store
            embedding_client: Optional pre-configured embedding client (for testing)
            vector_store: Optional pre-configured vector store (for testing)
        """
        self.settings = settings
        
        # Use provided clients or create from settings
        self._embedding = embedding_client or EmbeddingFactory.create(settings)
        self._vector_store = vector_store or VectorStoreFactory.create(settings)
    
    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict[str, Any]] = None,
        trace: Optional[TraceContext] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks using dense vector similarity.
        
        Args:
            query: Query text to search for
            top_k: Number of results to return
            filters: Optional metadata filters for the vector store query
            trace: Optional trace context for observability
            
        Returns:
            List of RetrievalResult sorted by relevance (highest score first)
            
        Raises:
            EmbeddingError: If embedding generation fails
            VectorStoreError: If vector store query fails
        """
        # Step 1: Generate embedding for the query
        query_vectors = self._embedding.embed([query], trace=trace)
        if not query_vectors or not query_vectors[0]:
            return []
        query_vector = query_vectors[0]
        
        # Step 2: Query vector store for similar vectors
        query_results: List[QueryResult] = self._vector_store.query(
            vector=query_vector,
            top_k=top_k,
            filters=filters,
            trace=trace
        )
        
        # Step 3: Transform QueryResult to RetrievalResult
        retrieval_results: List[RetrievalResult] = []
        for result in query_results:
            retrieval_results.append(
                RetrievalResult(
                    chunk_id=result.id,
                    score=result.score,
                    text=result.text,
                    metadata=result.metadata
                )
            )
        
        return retrieval_results
