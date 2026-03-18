#!/usr/bin/env python3
"""
Hybrid Search - Dense + Sparse + Fusion Orchestration

This module implements the HybridSearch class that orchestrates the complete
hybrid retrieval flow: Query Processing → Dense Retrieval + Sparse Retrieval
(in parallel) → RRF Fusion → Metadata Filtering → Top-K Results.

It serves as the main entry point for hybrid search in the RAG pipeline,
providing graceful degradation when either retrieval path fails.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, Protocol

from core.types import RetrievalResult
from core.settings import Settings
from core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)


class QueryProcessorProtocol(Protocol):
    """Protocol for query processor (D1)."""
    
    def process(self, query: str) -> dict[str, Any]:
        """
        Process query and extract keywords, filters, etc.
        
        Args:
            query: Raw user query
            
        Returns:
            Processed query dict with 'query', 'keywords', 'filters', etc.
        """
        ...


class DenseRetrieverProtocol(Protocol):
    """Protocol for dense retriever (D2)."""
    
    def retrieve(
        self, 
        query: str, 
        top_k: int, 
        filters: Optional[dict] = None,
        trace: Optional[TraceContext] = None
    ) -> list[RetrievalResult]:
        """
        Retrieve using dense vector similarity.
        
        Args:
            query: Query text
            top_k: Number of results to return
            filters: Optional metadata filters
            trace: Optional trace context
            
        Returns:
            List of retrieval results
        """
        ...


class SparseRetrieverProtocol(Protocol):
    """Protocol for sparse retriever (D3)."""
    
    def retrieve(
        self, 
        keywords: list[str], 
        top_k: int,
        trace: Optional[TraceContext] = None
    ) -> list[RetrievalResult]:
        """
        Retrieve using sparse (BM25) retrieval.
        
        Args:
            keywords: List of keywords to search for
            top_k: Number of results to return
            trace: Optional trace context
            
        Returns:
            List of retrieval results
        """
        ...


class FusionProtocol(Protocol):
    """Protocol for result fusion (D4)."""
    
    def fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int,
        trace: Optional[TraceContext] = None
    ) -> list[RetrievalResult]:
        """
        Fuse dense and sparse results.
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            top_k: Number of results to return
            trace: Optional trace context
            
        Returns:
            Fused and ranked results
        """
        ...


class HybridSearch:
    """
    Hybrid Search orchestrator for Dense + Sparse + Fusion retrieval.
    
    This class orchestrates the complete hybrid retrieval flow:
    1. Query processing (keyword extraction, filter parsing)
    2. Parallel dense and sparse retrieval
    3. Result fusion using RRF
    4. Metadata filtering (post-filter as safety net)
    5. Return Top-K results
    
    The class provides graceful degradation:
    - If one retrieval path fails, falls back to the other path
    - If fusion fails, returns dense results (if available)
    
    Attributes:
        settings: Application settings
        query_processor: Query processor component (D1)
        dense_retriever: Dense retriever component (D2)
        sparse_retriever: Sparse retriever component (D3)
        fusion: Fusion component (D4)
    """
    
    def __init__(
        self,
        settings: Settings,
        query_processor: Optional[QueryProcessorProtocol] = None,
        dense_retriever: Optional[DenseRetrieverProtocol] = None,
        sparse_retriever: Optional[SparseRetrieverProtocol] = None,
        fusion: Optional[FusionProtocol] = None
    ):
        """
        Initialize HybridSearch.
        
        Args:
            settings: Application settings containing retrieval configuration
            query_processor: Optional query processor (D1)
            dense_retriever: Optional dense retriever (D2)
            sparse_retriever: Optional sparse retriever (D3)
            fusion: Optional fusion component (D4)
        """
        self.settings = settings
        self.query_processor = query_processor
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion = fusion
        
        # Load retrieval settings from config
        retrieval_cfg = getattr(settings, 'retrieval', None)
        if retrieval_cfg:
            hybrid_cfg = getattr(retrieval_cfg, 'hybrid', {})
            self._top_k_dense = hybrid_cfg.get('top_k_dense', 20)
            self._top_k_sparse = hybrid_cfg.get('top_k_sparse', 20)
            self._fusion_k = hybrid_cfg.get('fusion_k', 60)  # RRF constant
        else:
            self._top_k_dense = 20
            self._top_k_sparse = 20
            self._fusion_k = 60
        
        logger.info(
            f"HybridSearch initialized: dense={dense_retriever is not None}, "
            f"sparse={sparse_retriever is not None}, fusion={fusion is not None}"
        )
    
    def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict] = None,
        trace: Optional[TraceContext] = None
    ) -> list[RetrievalResult]:
        """
        Execute hybrid search.
        
        Flow:
        1. Process query with QueryProcessor (extract keywords)
        2. Execute dense and sparse retrieval in parallel
        3. Fuse results using RRF
        4. Apply metadata filters
        5. Return Top-K results
        
        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {'collection': 'docs', 'doc_type': 'pdf'})
            trace: Optional trace context for observability
            
        Returns:
            List of retrieval results sorted by relevance
            
        Note:
            If dense or sparse retrieval fails, gracefully degrades to single-path results.
            If fusion fails, returns dense results if available.
        """
        if trace:
            trace.record_stage("hybrid_search_start", method="hybrid", details={"query": query, "top_k": top_k})
        
        # Step 1: Query Processing (D1)
        processed = self._process_query(query)
        keywords = processed.get('keywords', [])
        query_for_dense = processed.get('query', query)
        
        if trace:
            trace.record_stage("query_processing", details={"keywords": keywords, "processed_query": query_for_dense})
        
        # Step 2: Parallel Dense + Sparse Retrieval
        dense_results: list[RetrievalResult] = []
        sparse_results: list[RetrievalResult] = []
        dense_error: Optional[Exception] = None
        sparse_error: Optional[Exception] = None
        
        # Determine if we can run in parallel
        can_run_dense = self.dense_retriever is not None
        can_run_sparse = self.sparse_retriever is not None and keywords
        
        if can_run_dense and can_run_sparse:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=2) as executor:
                dense_future = executor.submit(
                    self._safe_dense_retrieve, query_for_dense, self._top_k_dense, filters, trace
                )
                sparse_future = executor.submit(
                    self._safe_sparse_retrieve, keywords, self._top_k_sparse, trace
                )
                
                # Collect results
                for future in as_completed([dense_future, sparse_future]):
                    try:
                        result_type, results, error = future.result()
                        if result_type == "dense":
                            dense_results = results
                            dense_error = error
                        else:
                            sparse_results = results
                            sparse_error = error
                    except Exception as e:
                        logger.error(f"Unexpected error in retrieval: {e}")
        else:
            # Sequential execution (one or both paths unavailable)
            if can_run_dense:
                _, dense_results, dense_error = self._safe_dense_retrieve(
                    query_for_dense, self._top_k_dense, filters, trace
                )
            if can_run_sparse:
                _, sparse_results, sparse_error = self._safe_sparse_retrieve(
                    keywords, self._top_k_sparse, trace
                )
        
        # Log retrieval stats
        logger.info(
            f"Retrieval complete: dense={len(dense_results)}" +
            (f" (error: {dense_error})" if dense_error else "") +
            f", sparse={len(sparse_results)}" +
            (f" (error: {sparse_error})" if sparse_error else "")
        )
        
        if trace:
            trace.record_stage(
                "retrieval",
                method="parallel",
                details={
                    "dense_count": len(dense_results),
                    "sparse_count": len(sparse_results),
                    "dense_error": str(dense_error) if dense_error else None,
                    "sparse_error": str(sparse_error) if sparse_error else None,
                }
            )
        
        # Step 3: Fusion (D4) with graceful degradation
        fused_results = self._fuse_results(dense_results, sparse_results, top_k * 2, trace)
        
        if trace:
            trace.record_stage("fusion", details={"fused_count": len(fused_results)})
        
        # Step 4: Metadata Filtering (post-filter as safety net)
        if filters:
            filtered_results = self._apply_metadata_filters(fused_results, filters)
            logger.info(f"Metadata filtering: {len(fused_results)} -> {len(filtered_results)}")
        else:
            filtered_results = fused_results
        
        # Step 5: Return Top-K
        final_results = filtered_results[:top_k]
        
        if trace:
            trace.record_stage(
                "hybrid_search_complete",
                details={
                    "final_count": len(final_results),
                    "filters_applied": filters is not None
                }
            )
        
        return final_results
    
    def _process_query(self, query: str) -> dict[str, Any]:
        """
        Process query using QueryProcessor if available.
        
        Args:
            query: Raw query string
            
        Returns:
            Processed query dict
        """
        if self.query_processor is None:
            # Fallback: simple keyword extraction (split by space)
            return {
                'query': query,
                'keywords': query.split(),
                'filters': {}
            }
        
        try:
            return self.query_processor.process(query)
        except Exception as e:
            logger.warning(f"QueryProcessor failed, using fallback: {e}")
            return {
                'query': query,
                'keywords': query.split(),
                'filters': {}
            }
    
    def _safe_dense_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict],
        trace: Optional[TraceContext]
    ) -> tuple[str, list[RetrievalResult], Optional[Exception]]:
        """
        Safely execute dense retrieval with error handling.
        
        Args:
            query: Query string
            top_k: Number of results
            filters: Optional filters
            trace: Optional trace context
            
        Returns:
            Tuple of (result_type, results, error)
        """
        if self.dense_retriever is None:
            return ("dense", [], None)
        
        try:
            results = self.dense_retriever.retrieve(query, top_k, filters, trace)
            return ("dense", results, None)
        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return ("dense", [], e)
    
    def _safe_sparse_retrieve(
        self,
        keywords: list[str],
        top_k: int,
        trace: Optional[TraceContext]
    ) -> tuple[str, list[RetrievalResult], Optional[Exception]]:
        """
        Safely execute sparse retrieval with error handling.
        
        Args:
            keywords: List of keywords
            top_k: Number of results
            trace: Optional trace context
            
        Returns:
            Tuple of (result_type, results, error)
        """
        if self.sparse_retriever is None:
            return ("sparse", [], None)
        
        try:
            results = self.sparse_retriever.retrieve(keywords, top_k, trace)
            return ("sparse", results, None)
        except Exception as e:
            logger.error(f"Sparse retrieval failed: {e}")
            return ("sparse", [], e)
    
    def _fuse_results(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int,
        trace: Optional[TraceContext]
    ) -> list[RetrievalResult]:
        """
        Fuse dense and sparse results with graceful degradation.
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            top_k: Number of results to return
            trace: Optional trace context
            
        Returns:
            Fused results or fallback to available path
        """
        # Handle edge cases
        if not dense_results and not sparse_results:
            return []
        
        if not dense_results:
            logger.info("No dense results, using sparse only")
            return sparse_results[:top_k]
        
        if not sparse_results:
            logger.info("No sparse results, using dense only")
            return dense_results[:top_k]
        
        # Try fusion
        if self.fusion is not None:
            try:
                return self.fusion.fuse(dense_results, sparse_results, top_k, trace)
            except Exception as e:
                logger.error(f"Fusion failed, falling back to dense: {e}")
                return dense_results[:top_k]
        else:
            # No fusion component - simple merge with dense first
            logger.warning("No fusion component, merging results (dense first)")
            seen_ids = set()
            merged = []
            
            for result in dense_results + sparse_results:
                if result.chunk_id not in seen_ids:
                    seen_ids.add(result.chunk_id)
                    merged.append(result)
            
            return merged[:top_k]
    
    def _apply_metadata_filters(
        self,
        candidates: list[RetrievalResult],
        filters: dict[str, Any]
    ) -> list[RetrievalResult]:
        """
        Apply metadata filters to candidate results.
        
        This is a post-filter fallback to ensure results match the filter criteria.
        Supports filtering by collection, doc_type, and other metadata fields.
        
        Filter logic: A result passes if its metadata contains ALL key-value pairs
        from the filter. For missing fields, uses "loose inclusion" (missing -> include)
        to avoid over-filtering.
        
        Args:
            candidates: List of retrieval results to filter
            filters: Dictionary of filter criteria (e.g., {'collection': 'docs'})
            
        Returns:
            Filtered list of results
        """
        if not filters:
            return candidates
        
        filtered = []
        
        for result in candidates:
            metadata = result.metadata or {}
            passes = True
            
            for key, expected_value in filters.items():
                actual_value = metadata.get(key)
                
                # Loose inclusion: if field is missing, include the result
                if actual_value is None:
                    continue
                
                # Strict match for non-None values
                if actual_value != expected_value:
                    passes = False
                    break
            
            if passes:
                filtered.append(result)
        
        return filtered
