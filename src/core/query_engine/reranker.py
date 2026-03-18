#!/usr/bin/env python3
"""
Core Reranker - Query Engine Reranking Orchestration

This module provides the CoreReranker class that orchestrates result reranking
in the query engine. It integrates with the libs.reranker backend and provides
automatic fallback to fusion ranking on failure or timeout.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from core.settings import Settings
from core.types import RetrievalResult
from core.trace.trace_context import TraceContext
from libs.reranker.reranker_factory import RerankerProvider, RerankerFactory
from libs.reranker.base_reranker import (
    RerankCandidate, RerankResult, BaseReranker, NoneReranker
)

logger = logging.getLogger(__name__)


@dataclass
class RerankResultInfo:
    """
    Result information from the reranking process.
    
    Attributes:
        results: List of reranked retrieval results
        fallback: Whether fallback was used due to reranker failure
        elapsed_ms: Time taken for reranking in milliseconds
        backend: Name of the reranker backend used
    """
    results: list[RetrievalResult]
    fallback: bool
    elapsed_ms: float
    backend: str


class CoreReranker:
    """
    Core layer reranker for orchestrating result reranking.
    
    This class serves as the bridge between the query engine and the
    libs.reranker backend. It handles:
    - Converting RetrievalResult to RerankCandidate
    - Calling the reranker with fallback support
    - Converting RerankResult back to RetrievalResult
    - Recording trace information
    
    Attributes:
        settings: Application settings
        reranker_provider: RerankerProvider instance (can be injected for testing)
    """
    
    def __init__(
        self,
        settings: Settings,
        reranker_provider: Optional[RerankerProvider] = None
    ) -> None:
        """
        Initialize CoreReranker.
        
        Args:
            settings: Application settings containing reranker configuration
            reranker_provider: Optional RerankerProvider for dependency injection.
                              If not provided, creates one from settings.
        """
        self.settings = settings
        
        if reranker_provider is not None:
            self._provider = reranker_provider
            self._backend_name = reranker_provider.reranker.__class__.__name__
        else:
            # Create reranker from factory
            reranker = RerankerFactory.create(settings)
            self._provider = RerankerProvider(reranker, fallback_on_error=True)
            self._backend_name = settings.reranker.backend
        
        logger.info(f"CoreReranker initialized with backend: {self._backend_name}")
    
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        trace: Optional[TraceContext] = None
    ) -> RerankResultInfo:
        """
        Rerank candidates based on query relevance.
        
        Flow:
        1. Convert RetrievalResult to RerankCandidate
        2. Call reranker with fallback support
        3. Convert RerankResult back to RetrievalResult
        4. Record trace information
        
        Args:
            query: Original query string
            candidates: List of retrieval results to rerank
            trace: Optional trace context for observability
            
        Returns:
            RerankResultInfo containing reranked results and metadata
        """
        start_time = time.time()
        
        # Handle empty candidates
        if not candidates:
            elapsed_ms = (time.time() - start_time) * 1000
            if trace:
                trace.record_stage(
                    name="rerank",
                    method="none",
                    provider=self._backend_name,
                    details={
                        "candidate_count": 0,
                        "fallback": False,
                        "elapsed_ms": elapsed_ms
                    }
                )
            return RerankResultInfo(
                results=[],
                fallback=False,
                elapsed_ms=elapsed_ms,
                backend=self._backend_name
            )
        
        # Step 1: Convert RetrievalResult to RerankCandidate
        rerank_candidates = self._to_rerank_candidates(candidates)
        
        # Step 2: Call reranker with fallback
        fallback_occurred = False
        try:
            rerank_results = self._provider.rerank_with_fallback(
                query, rerank_candidates, trace
            )
            
            # Check if fallback occurred by comparing backend class
            # If the result order matches original and scores are unchanged,
            # fallback likely occurred
            fallback_occurred = self._detect_fallback(rerank_results, rerank_candidates)
            
        except Exception as e:
            # This should not happen due to fallback, but handle just in case
            logger.error(f"Reranking failed completely: {e}")
            fallback_occurred = True
            # Return original candidates as fallback
            rerank_results = [
                RerankResult(
                    id=c.chunk_id,
                    text=c.text,
                    original_score=c.score,
                    rerank_score=c.score,
                    metadata=c.metadata.copy()
                )
                for c in candidates
            ]
        
        # Step 3: Convert RerankResult back to RetrievalResult
        final_results = self._to_retrieval_results(rerank_results)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Step 4: Record trace
        if trace:
            trace.record_stage(
                name="rerank",
                method=self._backend_name if not fallback_occurred else "fallback",
                provider=self._backend_name,
                details={
                    "candidate_count": len(candidates),
                    "result_count": len(final_results),
                    "fallback": fallback_occurred,
                    "elapsed_ms": elapsed_ms
                }
            )
        
        logger.info(
            f"Reranking complete: {len(candidates)} candidates -> "
            f"{len(final_results)} results, fallback={fallback_occurred}, "
            f"elapsed={elapsed_ms:.2f}ms"
        )
        
        return RerankResultInfo(
            results=final_results,
            fallback=fallback_occurred,
            elapsed_ms=elapsed_ms,
            backend=self._backend_name
        )
    
    def _to_rerank_candidates(
        self,
        results: list[RetrievalResult]
    ) -> list[RerankCandidate]:
        """
        Convert RetrievalResult to RerankCandidate.
        
        Args:
            results: List of retrieval results
            
        Returns:
            List of rerank candidates
        """
        return [
            RerankCandidate(
                id=r.chunk_id,
                text=r.text,
                score=r.score,
                metadata=r.metadata.copy()
            )
            for r in results
        ]
    
    def _to_retrieval_results(
        self,
        rerank_results: list[RerankResult]
    ) -> list[RetrievalResult]:
        """
        Convert RerankResult to RetrievalResult.
        
        Uses rerank_score as the new score for the result.
        
        Args:
            rerank_results: List of rerank results
            
        Returns:
            List of retrieval results with updated scores
        """
        return [
            RetrievalResult(
                chunk_id=r.id,
                text=r.text,
                score=r.rerank_score,  # Use rerank_score as the new score
                metadata=r.metadata.copy()
            )
            for r in rerank_results
        ]
    
    def _detect_fallback(
        self,
        rerank_results: list[RerankResult],
        original_candidates: list[RerankCandidate]
    ) -> bool:
        """
        Detect if fallback occurred by checking result characteristics.
        
        Fallback is detected when:
        - Results maintain original order (id sequence matches)
        - Rerank scores equal original scores
        
        Args:
            rerank_results: Results from reranking
            original_candidates: Original candidates
            
        Returns:
            True if fallback likely occurred
        """
        if len(rerank_results) != len(original_candidates):
            return False
        
        # Check if order and scores match original (indicates NoneReranker was used)
        for i, (result, candidate) in enumerate(zip(rerank_results, original_candidates)):
            if result.id != candidate.id:
                # Order changed, so actual reranking occurred
                return False
            if abs(result.rerank_score - candidate.score) > 1e-9:
                # Score changed, so actual reranking occurred
                return False
        
        # All results match original order and scores
        return True
