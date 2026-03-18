#!/usr/bin/env python3
"""
Base Reranker - Abstract Interface for Result Reranking

Defines the contract for reranking implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any

from core.trace.trace_context import TraceContext


@dataclass
class RerankCandidate:
    """A candidate for reranking."""
    id: str
    text: str
    score: float  # Original score from retrieval
    metadata: dict[str, Any]


@dataclass
class RerankResult:
    """Result from reranking."""
    id: str
    text: str
    original_score: float
    rerank_score: float  # New score from reranker
    metadata: dict[str, Any]


class BaseReranker(ABC):
    """
    Abstract base class for reranking implementations.
    
    All rerankers (CrossEncoder, LLM, None) must implement this interface.
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the reranker with configuration.
        
        Args:
            config: Strategy-specific configuration
        """
        self.config = config
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Optional[TraceContext] = None
    ) -> list[RerankResult]:
        """
        Rerank candidates based on query relevance.
        
        Args:
            query: Original query
            candidates: Candidates to rerank
            trace: Optional trace context
            
        Returns:
            Reranked results sorted by rerank_score
        """
        pass
    
    @abstractmethod
    async def arerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Optional[TraceContext] = None
    ) -> list[RerankResult]:
        """Async version of rerank."""
        pass


class NoneReranker(BaseReranker):
    """
    No-op reranker that returns candidates in original order.
    
    Used when reranking is disabled.
    """
    
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Optional[TraceContext] = None
    ) -> list[RerankResult]:
        """Return candidates without modification."""
        if trace:
            trace.record_stage(
                name="rerank",
                method="NoneReranker",
                details={"candidate_count": len(candidates), "skipped": True}
            )
        
        return [
            RerankResult(
                id=c.id,
                text=c.text,
                original_score=c.score,
                rerank_score=c.score,  # Keep original score
                metadata=c.metadata
            )
            for c in candidates
        ]
    
    async def arerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Optional[TraceContext] = None
    ) -> list[RerankResult]:
        """Async version - same as sync."""
        return self.rerank(query, candidates, trace)


class RerankerError(Exception):
    """Base exception for reranker errors."""
    pass


class RerankerFallbackError(RerankerError):
    """Raised when reranker fails and falls back to original order."""
    pass
