#!/usr/bin/env python3
"""
Base Evaluator - Abstract Interface for RAG Evaluation

Defines the contract for evaluation metric implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any

from core.trace.trace_context import TraceContext


@dataclass
class EvaluationInput:
    """Input data for evaluation."""
    query: str
    retrieved_ids: list[str]  # IDs of retrieved chunks
    golden_ids: list[str]     # Ground truth relevant IDs
    generated_answer: Optional[str] = None
    reference_answer: Optional[str] = None
    retrieved_texts: Optional[list[str]] = None


@dataclass
class EvaluationResult:
    """Result from evaluation."""
    metric_name: str
    score: float  # Typically 0.0 - 1.0
    details: dict[str, Any]


class BaseEvaluator(ABC):
    """
    Abstract base class for evaluation implementations.
    
    All evaluators (Ragas, DeepEval, Custom) must implement this interface.
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the evaluator with configuration.
        
        Args:
            config: Provider-specific configuration
        """
        self.config = config
        self.metrics = config.get("metrics", [])
    
    @abstractmethod
    def evaluate(
        self,
        inputs: list[EvaluationInput],
        trace: Optional[TraceContext] = None
    ) -> list[EvaluationResult]:
        """
        Evaluate RAG performance on a set of queries.
        
        Args:
            inputs: List of evaluation inputs
            trace: Optional trace context
            
        Returns:
            List of evaluation results (one per metric)
        """
        pass
    
    @abstractmethod
    def evaluate_single(
        self,
        input_data: EvaluationInput,
        trace: Optional[TraceContext] = None
    ) -> list[EvaluationResult]:
        """
        Evaluate a single query.
        
        Args:
            input_data: Single evaluation input
            trace: Optional trace context
            
        Returns:
            List of evaluation results
        """
        pass


class EvaluatorError(Exception):
    """Base exception for evaluator errors."""
    pass


class EvaluatorConfigError(EvaluatorError):
    """Raised when evaluator configuration is invalid."""
    pass
