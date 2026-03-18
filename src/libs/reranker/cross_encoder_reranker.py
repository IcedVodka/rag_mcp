#!/usr/bin/env python3
"""
Cross-Encoder Reranker - Cross-Encoder Model Reranking

Uses cross-encoder models for scoring query-document pairs.
"""

from typing import Optional, Any

from core.trace.trace_context import TraceContext
from .base_reranker import BaseReranker, RerankCandidate, RerankResult, RerankerFallbackError


class CrossEncoderReranker(BaseReranker):
    """Reranker using cross-encoder models."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.model = config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.device = config.get("device", "cpu")
        self.max_length = config.get("max_length", 512)
        self.batch_size = config.get("batch_size", 32)
        self._model = None
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(
                    self.model,
                    device=self.device,
                    max_length=self.max_length
                )
            except ImportError:
                raise RerankerFallbackError(
                    "sentence-transformers is required for cross-encoder reranking"
                )
        return self._model
    
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Optional[TraceContext] = None
    ) -> list[RerankResult]:
        """Rerank using cross-encoder."""
        if not candidates:
            return []
        
        if trace:
            trace.record_stage(
                name="rerank",
                method="CrossEncoderReranker",
                details={
                    "model": self.model,
                    "candidate_count": len(candidates)
                }
            )
        
        try:
            model = self._load_model()
            
            # Create query-candidate pairs
            pairs = [(query, c.text) for c in candidates]
            
            # Get scores in batches
            scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
            
            # Create results with new scores
            results = []
            for i, candidate in enumerate(candidates):
                results.append(RerankResult(
                    id=candidate.id,
                    text=candidate.text,
                    original_score=candidate.score,
                    rerank_score=float(scores[i]),
                    metadata=candidate.metadata
                ))
            
            # Sort by rerank_score descending
            results.sort(key=lambda x: x.rerank_score, reverse=True)
            return results
            
        except Exception as e:
            # Fallback to original order
            if trace:
                trace.record_stage(
                    name="rerank",
                    method="CrossEncoderReranker",
                    details={"error": str(e), "fallback": True}
                )
            
            return [
                RerankResult(
                    id=c.id,
                    text=c.text,
                    original_score=c.score,
                    rerank_score=c.score,
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
        """Async rerank."""
        import asyncio
        return await asyncio.to_thread(self.rerank, query, candidates, trace)
