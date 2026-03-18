#!/usr/bin/env python3
"""
Sparse Retriever - BM25-based keyword retrieval.

Loads BM25 index from storage and retrieves chunks based on keyword queries.
Integrates with VectorStore to fetch full text and metadata for retrieved chunks.
"""

from typing import List, Optional, Any, Dict
from pathlib import Path

from core.types import RetrievalResult
from core.trace.trace_context import TraceContext
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.vector_store.base_vector_store import BaseVectorStore


class SparseRetriever:
    """
    Sparse retriever using BM25 for keyword-based retrieval.
    
    Retrieves relevant chunks by querying a pre-built BM25 index,
    then fetches full document content and metadata from the vector store.
    
    Attributes:
        settings: Configuration settings
        bm25_indexer: BM25Indexer instance (or None to create from settings)
        vector_store: BaseVectorStore instance for fetching chunk content
        
    Example:
        >>> retriever = SparseRetriever(settings, bm25_indexer, vector_store)
        >>> results = retriever.retrieve(["machine", "learning"], top_k=10)
    """
    
    def __init__(
        self,
        settings: Optional[Dict[str, Any]] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        vector_store: Optional[BaseVectorStore] = None
    ) -> None:
        """
        Initialize the sparse retriever.
        
        Args:
            settings: Configuration dictionary containing:
                - bm25_index_path: Path to BM25 index directory
                - bm25_k1: BM25 k1 parameter (default: 1.5)
                - bm25_b: BM25 b parameter (default: 0.75)
            bm25_indexer: Pre-configured BM25Indexer instance (optional)
            vector_store: VectorStore for fetching chunk content (required if querying)
            
        Raises:
            ValueError: If settings not provided and bm25_indexer is None
        """
        self.settings = settings or {}
        
        # Initialize BM25 indexer
        if bm25_indexer is not None:
            self.bm25_indexer = bm25_indexer
        else:
            index_path = self.settings.get("bm25_index_path", "data/db/bm25")
            k1 = self.settings.get("bm25_k1", 1.5)
            b = self.settings.get("bm25_b", 0.75)
            self.bm25_indexer = BM25Indexer(
                index_path=index_path,
                k1=k1,
                b=b
            )
        
        self.vector_store = vector_store
    
    def retrieve(
        self,
        keywords: List[str],
        top_k: int = 10,
        trace: Optional[TraceContext] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve chunks using BM25 keyword search.
        
        Flow:
        1. Query BM25 index with keywords to get (chunk_id, score) pairs
        2. Fetch full text and metadata from vector store using chunk_ids
        3. Merge scores with content to create RetrievalResult objects
        
        Args:
            keywords: List of query keywords from QueryProcessor
            top_k: Maximum number of results to return
            trace: Optional trace context for monitoring
            
        Returns:
            List of RetrievalResult sorted by score (descending)
            
        Raises:
            ValueError: If vector_store is not set
        """
        if trace:
            trace.record_stage(
                name="sparse_retrieval",
                method="bm25",
                provider="bm25_indexer",
                details={"keywords": keywords, "top_k": top_k}
            )
        
        # Step 1: Query BM25 index
        bm25_results = self.bm25_indexer.query(keywords, top_k=top_k)
        
        if not bm25_results:
            return []
        
        # Extract chunk_ids and scores
        chunk_ids = [chunk_id for chunk_id, _ in bm25_results]
        score_map = {chunk_id: score for chunk_id, score in bm25_results}
        
        if trace:
            trace.record_stage(
                name="bm25_query_complete",
                details={"results_found": len(chunk_ids)}
            )
        
        # Step 2: Fetch full records from vector store
        if self.vector_store is None:
            raise ValueError(
                "Vector store not initialized. "
                "Please provide vector_store parameter to SparseRetriever."
            )
        
        records = self.vector_store.get_by_ids(chunk_ids, trace=trace)
        
        if trace:
            trace.record_stage(
                name="vector_store_fetch_complete",
                details={"records_fetched": len(records)}
            )
        
        # Step 3: Merge scores with content to create RetrievalResult objects
        results: List[RetrievalResult] = []
        for record in records:
            chunk_id = record["id"]
            results.append(RetrievalResult(
                chunk_id=chunk_id,
                text=record["text"],
                score=score_map.get(chunk_id, 0.0),
                metadata=record.get("metadata", {})
            ))
        
        # Sort by score (descending) to maintain BM25 ranking
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the BM25 index.
        
        Returns:
            Dictionary with index statistics
        """
        return self.bm25_indexer.get_stats()
