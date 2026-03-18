#!/usr/bin/env python3
"""
Reranker Factory - Strategy-Agnostic Reranker Creation

Creates appropriate reranking instances based on configuration.
"""

from typing import Optional, Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from .base_reranker import BaseReranker, NoneReranker, RerankCandidate, RerankResult


class RerankerFactory:
    """
    Factory for creating reranker instances.
    
    Routes to appropriate backend implementation based on settings.
    """
    
    @staticmethod
    def create(settings: Settings) -> BaseReranker:
        """
        Create a reranker instance based on settings.
        
        Args:
            settings: Application settings
            
        Returns:
            Configured reranker instance
            
        Raises:
            ValueError: If backend is unknown
        """
        backend = settings.reranker.backend
        config = getattr(settings.reranker, backend, {})
        
        if backend == "none":
            return NoneReranker(config)
        elif backend == "cross_encoder":
            from .cross_encoder_reranker import CrossEncoderReranker
            return CrossEncoderReranker(config)
        elif backend == "llm":
            from .llm_reranker import LLMReranker
            return LLMReranker(config)
        else:
            raise ValueError(f"Unknown reranker backend: {backend}")


class RerankerProvider:
    """
    Wrapper for reranker with common functionality.
    
    Provides error handling and fallback behavior.
    """
    
    def __init__(self, reranker: BaseReranker, fallback_on_error: bool = True) -> None:
        self.reranker = reranker
        self.fallback_on_error = fallback_on_error
        self.none_reranker = NoneReranker({})
    
    def rerank_with_fallback(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Optional[TraceContext] = None
    ) -> list[RerankResult]:
        """
        Rerank with automatic fallback on error.
        
        Args:
            query: Original query
            candidates: Candidates to rerank
            trace: Optional trace context
            
        Returns:
            Reranked results (or original order on failure)
        """
        try:
            results = self.reranker.rerank(query, candidates, trace)
            return results
        except Exception as e:
            if trace:
                trace.record_stage(
                    name="rerank",
                    method="fallback",
                    details={
                        "error": str(e),
                        "original_reranker": self.reranker.__class__.__name__,
                        "fallback": True
                    }
                )
            
            if self.fallback_on_error:
                # Fallback to NoneReranker (original order)
                return self.none_reranker.rerank(query, candidates, trace)
            raise
