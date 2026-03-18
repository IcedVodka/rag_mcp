#!/usr/bin/env python3
"""
Evaluator Factory - Provider-Agnostic Evaluator Creation

Creates appropriate evaluation instances based on configuration.
"""

from typing import Optional, Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from .base_evaluator import BaseEvaluator, EvaluationInput, EvaluationResult


class EvaluatorFactory:
    """
    Factory for creating evaluator instances.
    
    Routes to appropriate provider implementation based on settings.
    """
    
    @staticmethod
    def create(settings: Settings) -> BaseEvaluator:
        """
        Create an evaluator instance based on settings.
        
        Args:
            settings: Application settings
            
        Returns:
            Configured evaluator instance
            
        Raises:
            ValueError: If provider is unknown
        """
        provider = settings.evaluation.provider
        config = getattr(settings.evaluation, provider, {})
        
        if provider == "custom":
            from .custom_evaluator import CustomEvaluator
            return CustomEvaluator(config)
        elif provider == "ragas":
            from .ragas_evaluator import RagasEvaluator
            return RagasEvaluator(config)
        else:
            raise ValueError(f"Unknown evaluation provider: {provider}")


class EvaluatorProvider:
    """
    Wrapper for evaluator with common functionality.
    
    Provides trace recording and batch evaluation.
    """
    
    def __init__(self, evaluator: BaseEvaluator) -> None:
        self.evaluator = evaluator
    
    def evaluate_with_trace(
        self,
        inputs: list[EvaluationInput],
        trace: Optional[TraceContext] = None
    ) -> list[EvaluationResult]:
        """
        Evaluate with automatic trace recording.
        
        Args:
            inputs: Evaluation inputs
            trace: Optional trace context
            
        Returns:
            Evaluation results
        """
        if trace:
            trace.record_stage(
                name="evaluation",
                provider=self.evaluator.__class__.__name__,
                details={
                    "metrics": self.evaluator.metrics,
                    "query_count": len(inputs)
                }
            )
        
        return self.evaluator.evaluate(inputs, trace)
    
    def get_average_scores(
        self,
        results: list[list[EvaluationResult]]
    ) -> dict[str, float]:
        """
        Calculate average scores across multiple evaluations.
        
        Args:
            results: List of result lists (one per query)
            
        Returns:
            Dictionary mapping metric names to average scores
        """
        from collections import defaultdict
        
        scores_by_metric: dict[str, list[float]] = defaultdict(list)
        
        for result_list in results:
            for result in result_list:
                scores_by_metric[result.metric_name].append(result.score)
        
        averages = {}
        for metric, scores in scores_by_metric.items():
            averages[metric] = sum(scores) / len(scores) if scores else 0.0
        
        return averages
