#!/usr/bin/env python3
"""
Evaluator Factory Tests

Tests for evaluator factory routing and provider creation.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add src to path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from core.settings import Settings, EvaluationSettings
from libs.evaluator.evaluator_factory import EvaluatorFactory, EvaluatorProvider
from libs.evaluator.base_evaluator import (
    BaseEvaluator, EvaluationInput, EvaluationResult, EvaluatorError
)


class FakeEvaluator(BaseEvaluator):
    """Fake evaluator for testing."""
    
    def evaluate(self, inputs, trace=None):
        # Return fake metrics
        return [
            EvaluationResult(
                metric_name="hit_rate",
                score=0.8,
                details={"hits": 4, "total": 5}
            ),
            EvaluationResult(
                metric_name="mrr",
                score=0.75,
                details={"reciprocal_rank": 0.75}
            )
        ]
    
    def evaluate_single(self, input_data, trace=None):
        return self.evaluate([input_data], trace)


class TestEvaluatorFactory:
    """Test evaluator factory routing logic."""
    
    def test_factory_routes_to_custom(self):
        """Test factory routes to custom provider."""
        settings = Mock(spec=Settings)
        settings.evaluation = Mock(spec=EvaluationSettings)
        settings.evaluation.provider = "custom"
        settings.evaluation.custom = {"metrics": ["hit_rate", "mrr"]}
        
        with patch.dict('sys.modules', {'libs.evaluator.custom_evaluator': Mock(CustomEvaluator=FakeEvaluator)}):
            evaluator = EvaluatorFactory.create(settings)
            assert isinstance(evaluator, FakeEvaluator)
    
    def test_factory_routes_to_ragas(self):
        """Test factory routes to ragas provider."""
        settings = Mock(spec=Settings)
        settings.evaluation = Mock(spec=EvaluationSettings)
        settings.evaluation.provider = "ragas"
        settings.evaluation.ragas = {"metrics": ["faithfulness", "relevancy"]}
        
        with patch.dict('sys.modules', {'libs.evaluator.ragas_evaluator': Mock(RagasEvaluator=FakeEvaluator)}):
            evaluator = EvaluatorFactory.create(settings)
            assert isinstance(evaluator, FakeEvaluator)
    
    def test_factory_raises_on_unknown_provider(self):
        """Test factory raises error for unknown provider."""
        settings = Mock(spec=Settings)
        settings.evaluation = Mock(spec=EvaluationSettings)
        settings.evaluation.provider = "unknown_provider"
        
        with pytest.raises(ValueError) as exc_info:
            EvaluatorFactory.create(settings)
        
        assert "unknown_provider" in str(exc_info.value).lower()


class TestFakeEvaluator:
    """Test fake evaluator behavior."""
    
    def test_evaluate_returns_results(self):
        """Test evaluate returns expected results."""
        evaluator = FakeEvaluator({"metrics": ["hit_rate"]})
        
        inputs = [
            EvaluationInput(
                query="test",
                retrieved_ids=["1", "2"],
                golden_ids=["1"]
            )
        ]
        
        results = evaluator.evaluate(inputs)
        
        assert len(results) == 2
        assert results[0].metric_name == "hit_rate"
        assert results[0].score == 0.8
    
    def test_evaluate_single(self):
        """Test evaluate_single returns results."""
        evaluator = FakeEvaluator({"metrics": ["hit_rate"]})
        
        input_data = EvaluationInput(
            query="test",
            retrieved_ids=["1", "2"],
            golden_ids=["1"]
        )
        
        results = evaluator.evaluate_single(input_data)
        
        assert len(results) == 2
        assert all(isinstance(r, EvaluationResult) for r in results)


class TestEvaluatorProvider:
    """Test evaluator provider wrapper."""
    
    def test_evaluate_with_trace(self):
        """Test evaluate with trace recording."""
        evaluator = FakeEvaluator({"metrics": ["hit_rate"]})
        provider = EvaluatorProvider(evaluator)
        
        from core.trace.trace_context import TraceContext
        trace = TraceContext(trace_type="test")
        
        inputs = [EvaluationInput(query="q", retrieved_ids=["1"], golden_ids=["1"])]
        results = provider.evaluate_with_trace(inputs, trace)
        
        assert len(results) == 2
        assert len(trace.stages) == 1
        assert trace.stages[0]["name"] == "evaluation"
    
    def test_get_average_scores(self):
        """Test calculating average scores."""
        evaluator = FakeEvaluator({})
        provider = EvaluatorProvider(evaluator)
        
        # Simulate results from 3 queries
        results = [
            [
                EvaluationResult("hit_rate", 0.8, {}),
                EvaluationResult("mrr", 0.7, {})
            ],
            [
                EvaluationResult("hit_rate", 1.0, {}),
                EvaluationResult("mrr", 0.8, {})
            ],
            [
                EvaluationResult("hit_rate", 0.6, {}),
                EvaluationResult("mrr", 0.6, {})
            ]
        ]
        
        averages = provider.get_average_scores(results)
        
        assert averages["hit_rate"] == pytest.approx(0.8, 0.01)  # (0.8 + 1.0 + 0.6) / 3
        assert averages["mrr"] == pytest.approx(0.7, 0.01)  # (0.7 + 0.8 + 0.6) / 3
    
    def test_get_average_scores_empty(self):
        """Test calculating average with empty results."""
        evaluator = FakeEvaluator({})
        provider = EvaluatorProvider(evaluator)
        
        averages = provider.get_average_scores([])
        
        assert averages == {}
