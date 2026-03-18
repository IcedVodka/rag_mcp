#!/usr/bin/env python3
"""
Custom Evaluator - Custom RAG Metrics Implementation

Implements custom evaluation metrics (Hit Rate, MRR, etc.).
"""

from typing import Optional, Any

from core.trace.trace_context import TraceContext
from .base_evaluator import BaseEvaluator, EvaluationInput, EvaluationResult, EvaluatorError


class CustomEvaluator(BaseEvaluator):
    """
    Custom evaluator for RAG metrics.
    
    Implements:
    - Hit Rate: Was any relevant doc retrieved?
    - MRR (Mean Reciprocal Rank): 1 / rank of first relevant doc
    - NDCG: Normalized Discounted Cumulative Gain
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.metrics = config.get("metrics", ["hit_rate", "mrr"])
    
    def evaluate(
        self,
        inputs: list[EvaluationInput],
        trace: Optional[TraceContext] = None
    ) -> list[EvaluationResult]:
        """Evaluate multiple queries and aggregate results."""
        all_results = []
        for inp in inputs:
            all_results.extend(self.evaluate_single(inp, trace))
        
        # Aggregate by metric
        from collections import defaultdict
        scores_by_metric = defaultdict(list)
        
        for result in all_results:
            scores_by_metric[result.metric_name].append(result.score)
        
        # Calculate averages
        aggregated = []
        for metric, scores in scores_by_metric.items():
            avg_score = sum(scores) / len(scores) if scores else 0.0
            aggregated.append(EvaluationResult(
                metric_name=metric,
                score=avg_score,
                details={"query_count": len(inputs)}
            ))
        
        return aggregated
    
    def evaluate_single(
        self,
        input_data: EvaluationInput,
        trace: Optional[TraceContext] = None
    ) -> list[EvaluationResult]:
        """Evaluate a single query."""
        results = []
        
        if "hit_rate" in self.metrics:
            results.append(self._calculate_hit_rate(input_data))
        
        if "mrr" in self.metrics:
            results.append(self._calculate_mrr(input_data))
        
        return results
    
    def _calculate_hit_rate(self, inp: EvaluationInput) -> EvaluationResult:
        """Calculate hit rate."""
        hits = set(inp.retrieved_ids) & set(inp.golden_ids)
        hit = 1.0 if hits else 0.0
        
        return EvaluationResult(
            metric_name="hit_rate",
            score=hit,
            details={
                "hits": len(hits),
                "retrieved_count": len(inp.retrieved_ids),
                "golden_count": len(inp.golden_ids)
            }
        )
    
    def _calculate_mrr(self, inp: EvaluationInput) -> EvaluationResult:
        """Calculate Mean Reciprocal Rank."""
        for rank, doc_id in enumerate(inp.retrieved_ids, start=1):
            if doc_id in inp.golden_ids:
                return EvaluationResult(
                    metric_name="mrr",
                    score=1.0 / rank,
                    details={"first_hit_rank": rank}
                )
        
        # No hits
        return EvaluationResult(
            metric_name="mrr",
            score=0.0,
            details={"first_hit_rank": None}
        )
