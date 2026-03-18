#!/usr/bin/env python3
"""
Unit tests for Core Reranker with Fallback.

Test scenarios:
1. Successful reranking with order change
2. Fallback when reranker fails
3. Empty candidates handling
4. Trace recording
5. Dependency injection
"""

import pytest
from unittest.mock import Mock, MagicMock

from core.query_engine.reranker import CoreReranker, RerankResultInfo
from core.types import RetrievalResult
from core.settings import (
    Settings, LLMSettings, EmbeddingSettings, VectorStoreSettings,
    SplitterSettings, RetrievalSettings, RerankerSettings,
    IngestionSettings, BM25Settings, EvaluationSettings,
    ObservabilitySettings, StorageSettings
)
from core.trace.trace_context import TraceContext
from libs.reranker.base_reranker import (
    RerankCandidate, RerankResult, BaseReranker, NoneReranker
)
from libs.reranker.reranker_factory import RerankerProvider


def create_mock_settings(backend: str = "cross_encoder") -> Settings:
    """Create a minimal settings object for testing."""
    return Settings(
        llm=LLMSettings(provider="test"),
        embedding=EmbeddingSettings(provider="test"),
        vector_store=VectorStoreSettings(provider="test"),
        splitter=SplitterSettings(strategy="recursive"),
        retrieval=RetrievalSettings(),
        reranker=RerankerSettings(
            backend=backend,
            cross_encoder={"model": "cross-encoder/ms-marco"},
            llm={"prompt_path": "config/prompts/rerank.txt"}
        ),
        ingestion=IngestionSettings(),
        bm25=BM25Settings(),
        evaluation=EvaluationSettings(provider="test"),
        observability=ObservabilitySettings(),
        storage=StorageSettings(),
    )


def create_retrieval_result(
    chunk_id: str,
    score: float = 0.5,
    text: str = "test text"
) -> RetrievalResult:
    """Helper to create a RetrievalResult."""
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text,
        metadata={"source": "test", "chunk_id": chunk_id}
    )


class FakeReranker(BaseReranker):
    """Fake reranker that reverses order and assigns new scores."""
    
    def rerank(self, query, candidates, trace=None):
        results = []
        for i, c in enumerate(reversed(candidates)):
            results.append(RerankResult(
                id=c.id,
                text=c.text,
                original_score=c.score,
                rerank_score=1.0 - (i * 0.1),
                metadata=c.metadata
            ))
        return results
    
    async def arerank(self, query, candidates, trace=None):
        return self.rerank(query, candidates, trace)


class ErrorReranker(BaseReranker):
    """Reranker that always fails."""
    
    def rerank(self, query, candidates, trace=None):
        raise Exception("Simulated reranker failure")
    
    async def arerank(self, query, candidates, trace=None):
        raise Exception("Simulated reranker failure")


class TestCoreRerankerBasic:
    """Test basic CoreReranker functionality."""
    
    def test_init_with_provider_injection(self):
        """Test initialization with injected RerankerProvider."""
        settings = create_mock_settings()
        fake_reranker = FakeReranker({})
        provider = RerankerProvider(fake_reranker, fallback_on_error=True)
        
        reranker = CoreReranker(settings, reranker_provider=provider)
        
        assert reranker._provider is provider
        assert reranker._backend_name == "FakeReranker"
    
    def test_init_without_provider(self):
        """Test initialization without provider (creates from settings)."""
        settings = create_mock_settings(backend="none")
        
        reranker = CoreReranker(settings)
        
        assert reranker._backend_name == "none"
        assert isinstance(reranker._provider.reranker, NoneReranker)


class TestCoreRerankerReranking:
    """Test reranking functionality."""
    
    def test_successful_reranking_changes_order(self):
        """Test that successful reranking changes result order."""
        settings = create_mock_settings()
        fake_reranker = FakeReranker({})
        provider = RerankerProvider(fake_reranker, fallback_on_error=True)
        reranker = CoreReranker(settings, reranker_provider=provider)
        
        candidates = [
            create_retrieval_result("1", score=0.9, text="text1"),
            create_retrieval_result("2", score=0.8, text="text2"),
            create_retrieval_result("3", score=0.7, text="text3"),
        ]
        
        result = reranker.rerank("test query", candidates)
        
        assert isinstance(result, RerankResultInfo)
        assert len(result.results) == 3
        # FakeReranker reverses order: 3, 2, 1
        assert result.results[0].chunk_id == "3"
        assert result.results[1].chunk_id == "2"
        assert result.results[2].chunk_id == "1"
        # Check scores are updated
        assert result.results[0].score == 1.0
        assert result.results[1].score == 0.9
        assert result.results[2].score == 0.8
    
    def test_fallback_on_reranker_error(self):
        """Test fallback when reranker fails."""
        settings = create_mock_settings()
        error_reranker = ErrorReranker({})
        provider = RerankerProvider(error_reranker, fallback_on_error=True)
        reranker = CoreReranker(settings, reranker_provider=provider)
        
        candidates = [
            create_retrieval_result("1", score=0.9, text="text1"),
            create_retrieval_result("2", score=0.8, text="text2"),
        ]
        
        result = reranker.rerank("test query", candidates)
        
        assert isinstance(result, RerankResultInfo)
        assert result.fallback is True
        assert len(result.results) == 2
        # Order should be preserved (fallback)
        assert result.results[0].chunk_id == "1"
        assert result.results[1].chunk_id == "2"
        # Scores should be preserved
        assert result.results[0].score == 0.9
        assert result.results[1].score == 0.8
    
    def test_empty_candidates(self):
        """Test handling of empty candidates."""
        settings = create_mock_settings()
        fake_reranker = FakeReranker({})
        provider = RerankerProvider(fake_reranker, fallback_on_error=True)
        reranker = CoreReranker(settings, reranker_provider=provider)
        
        result = reranker.rerank("test query", [])
        
        assert isinstance(result, RerankResultInfo)
        assert result.results == []
        assert result.fallback is False
        assert result.elapsed_ms >= 0
    
    def test_single_candidate(self):
        """Test with single candidate."""
        settings = create_mock_settings()
        fake_reranker = FakeReranker({})
        provider = RerankerProvider(fake_reranker, fallback_on_error=True)
        reranker = CoreReranker(settings, reranker_provider=provider)
        
        candidates = [create_retrieval_result("1", score=0.9, text="text1")]
        
        result = reranker.rerank("test query", candidates)
        
        assert len(result.results) == 1
        assert result.results[0].chunk_id == "1"
        assert result.fallback is False  # Single item, no real reordering


class TestCoreRerankerTrace:
    """Test trace recording functionality."""
    
    def test_trace_recorded_on_success(self):
        """Test that trace is recorded on successful reranking."""
        settings = create_mock_settings()
        fake_reranker = FakeReranker({})
        provider = RerankerProvider(fake_reranker, fallback_on_error=True)
        reranker = CoreReranker(settings, reranker_provider=provider)
        
        trace = TraceContext()
        candidates = [
            create_retrieval_result("1", score=0.9),
            create_retrieval_result("2", score=0.8),
        ]
        
        reranker.rerank("test query", candidates, trace=trace)
        
        # Check trace stages
        rerank_stages = [s for s in trace.stages if s["name"] == "rerank"]
        assert len(rerank_stages) == 1
        
        stage = rerank_stages[0]
        assert stage["provider"] == "FakeReranker"
        assert stage["details"]["candidate_count"] == 2
        assert stage["details"]["result_count"] == 2
        assert stage["details"]["fallback"] is False
        assert "elapsed_ms" in stage["details"]
    
    def test_trace_recorded_on_fallback(self):
        """Test that trace is recorded with fallback flag on error."""
        settings = create_mock_settings()
        error_reranker = ErrorReranker({})
        provider = RerankerProvider(error_reranker, fallback_on_error=True)
        reranker = CoreReranker(settings, reranker_provider=provider)
        
        trace = TraceContext()
        candidates = [
            create_retrieval_result("1", score=0.9),
            create_retrieval_result("2", score=0.8),
        ]
        
        reranker.rerank("test query", candidates, trace=trace)
        
        # Check trace stages - there may be multiple "rerank" stages
        # from RerankerProvider, NoneReranker, and CoreReranker
        rerank_stages = [s for s in trace.stages if s["name"] == "rerank"]
        assert len(rerank_stages) >= 1
        
        # Find the stage recorded by CoreReranker (has "fallback" in details and provider field)
        core_stage = None
        for stage in rerank_stages:
            if stage.get("details", {}).get("fallback") is True and "provider" in stage:
                core_stage = stage
                break
        
        assert core_stage is not None
        assert core_stage["method"] == "fallback"
        assert core_stage["details"]["fallback"] is True
    
    def test_trace_recorded_on_empty(self):
        """Test that trace is recorded for empty candidates."""
        settings = create_mock_settings()
        fake_reranker = FakeReranker({})
        provider = RerankerProvider(fake_reranker, fallback_on_error=True)
        reranker = CoreReranker(settings, reranker_provider=provider)
        
        trace = TraceContext()
        
        reranker.rerank("test query", [], trace=trace)
        
        rerank_stages = [s for s in trace.stages if s["name"] == "rerank"]
        assert len(rerank_stages) == 1
        
        stage = rerank_stages[0]
        assert stage["method"] == "none"
        assert stage["details"]["candidate_count"] == 0


class TestCoreRerankerResultConversion:
    """Test result conversion between types."""
    
    def test_retrieval_result_to_rerank_candidate(self):
        """Test conversion from RetrievalResult to RerankCandidate."""
        settings = create_mock_settings()
        fake_reranker = FakeReranker({})
        provider = RerankerProvider(fake_reranker, fallback_on_error=True)
        reranker = CoreReranker(settings, reranker_provider=provider)
        
        retrieval_results = [
            RetrievalResult(
                chunk_id="doc1_chunk1",
                score=0.95,
                text="test content",
                metadata={"key": "value", "num": 42}
            )
        ]
        
        candidates = reranker._to_rerank_candidates(retrieval_results)
        
        assert len(candidates) == 1
        assert candidates[0].id == "doc1_chunk1"
        assert candidates[0].score == 0.95
        assert candidates[0].text == "test content"
        assert candidates[0].metadata == {"key": "value", "num": 42}
    
    def test_rerank_result_to_retrieval_result(self):
        """Test conversion from RerankResult to RetrievalResult."""
        settings = create_mock_settings()
        fake_reranker = FakeReranker({})
        provider = RerankerProvider(fake_reranker, fallback_on_error=True)
        reranker = CoreReranker(settings, reranker_provider=provider)
        
        rerank_results = [
            RerankResult(
                id="chunk1",
                text="content",
                original_score=0.8,
                rerank_score=0.95,  # Improved score
                metadata={"source": "test"}
            )
        ]
        
        retrieval_results = reranker._to_retrieval_results(rerank_results)
        
        assert len(retrieval_results) == 1
        assert retrieval_results[0].chunk_id == "chunk1"
        assert retrieval_results[0].score == 0.95  # Uses rerank_score
        assert retrieval_results[0].text == "content"
        assert retrieval_results[0].metadata == {"source": "test"}
    
    def test_metadata_is_copied(self):
        """Test that metadata is copied, not referenced."""
        settings = create_mock_settings()
        fake_reranker = FakeReranker({})
        provider = RerankerProvider(fake_reranker, fallback_on_error=True)
        reranker = CoreReranker(settings, reranker_provider=provider)
        
        original_metadata = {"key": "value"}
        retrieval_results = [
            RetrievalResult(
                chunk_id="chunk1",
                score=0.8,
                text="text",
                metadata=original_metadata
            )
        ]
        
        candidates = reranker._to_rerank_candidates(retrieval_results)
        # Modify candidate metadata
        candidates[0].metadata["key"] = "modified"
        
        # Original should be unchanged
        assert original_metadata["key"] == "value"


class TestCoreRerankerFallbackDetection:
    """Test fallback detection logic."""
    
    def test_detect_fallback_true(self):
        """Test detection when fallback occurred."""
        settings = create_mock_settings()
        fake_reranker = FakeReranker({})
        provider = RerankerProvider(fake_reranker, fallback_on_error=True)
        reranker = CoreReranker(settings, reranker_provider=provider)
        
        # Results that match original order and scores
        rerank_results = [
            RerankResult(id="1", text="t1", original_score=0.9, rerank_score=0.9, metadata={}),
            RerankResult(id="2", text="t2", original_score=0.8, rerank_score=0.8, metadata={}),
        ]
        original_candidates = [
            RerankCandidate(id="1", text="t1", score=0.9, metadata={}),
            RerankCandidate(id="2", text="t2", score=0.8, metadata={}),
        ]
        
        is_fallback = reranker._detect_fallback(rerank_results, original_candidates)
        
        assert is_fallback is True
    
    def test_detect_fallback_false_different_order(self):
        """Test detection when order changed (no fallback)."""
        settings = create_mock_settings()
        fake_reranker = FakeReranker({})
        provider = RerankerProvider(fake_reranker, fallback_on_error=True)
        reranker = CoreReranker(settings, reranker_provider=provider)
        
        # Results in different order
        rerank_results = [
            RerankResult(id="2", text="t2", original_score=0.8, rerank_score=0.95, metadata={}),
            RerankResult(id="1", text="t1", original_score=0.9, rerank_score=0.85, metadata={}),
        ]
        original_candidates = [
            RerankCandidate(id="1", text="t1", score=0.9, metadata={}),
            RerankCandidate(id="2", text="t2", score=0.8, metadata={}),
        ]
        
        is_fallback = reranker._detect_fallback(rerank_results, original_candidates)
        
        assert is_fallback is False
    
    def test_detect_fallback_false_different_scores(self):
        """Test detection when scores changed (no fallback)."""
        settings = create_mock_settings()
        fake_reranker = FakeReranker({})
        provider = RerankerProvider(fake_reranker, fallback_on_error=True)
        reranker = CoreReranker(settings, reranker_provider=provider)
        
        # Same order but different scores
        rerank_results = [
            RerankResult(id="1", text="t1", original_score=0.9, rerank_score=0.95, metadata={}),
            RerankResult(id="2", text="t2", original_score=0.8, rerank_score=0.85, metadata={}),
        ]
        original_candidates = [
            RerankCandidate(id="1", text="t1", score=0.9, metadata={}),
            RerankCandidate(id="2", text="t2", score=0.8, metadata={}),
        ]
        
        is_fallback = reranker._detect_fallback(rerank_results, original_candidates)
        
        assert is_fallback is False


class TestCoreRerankerIntegration:
    """Integration-style tests with mock provider."""
    
    def test_mock_provider_injection(self):
        """Test using a completely mocked provider."""
        settings = create_mock_settings()
        
        # Create a mock provider
        mock_provider = Mock(spec=RerankerProvider)
        mock_provider.rerank_with_fallback.return_value = [
            RerankResult(id="2", text="t2", original_score=0.8, rerank_score=0.98, metadata={}),
            RerankResult(id="1", text="t1", original_score=0.9, rerank_score=0.85, metadata={}),
        ]
        mock_provider.reranker = Mock()
        mock_provider.reranker.__class__ = Mock
        mock_provider.reranker.__class__.__name__ = "MockReranker"
        
        reranker = CoreReranker(settings, reranker_provider=mock_provider)
        
        candidates = [
            create_retrieval_result("1", score=0.9),
            create_retrieval_result("2", score=0.8),
        ]
        
        result = reranker.rerank("query", candidates)
        
        # Verify mock was called
        mock_provider.rerank_with_fallback.assert_called_once()
        call_args = mock_provider.rerank_with_fallback.call_args
        assert call_args[0][0] == "query"  # First positional arg is query
        
        # Verify results
        assert result.results[0].chunk_id == "2"
        assert result.results[0].score == 0.98
    
    def test_timing_measurement(self):
        """Test that timing is properly measured."""
        settings = create_mock_settings()
        fake_reranker = FakeReranker({})
        provider = RerankerProvider(fake_reranker, fallback_on_error=True)
        reranker = CoreReranker(settings, reranker_provider=provider)
        
        candidates = [create_retrieval_result("1", score=0.9)]
        
        result = reranker.rerank("query", candidates)
        
        assert result.elapsed_ms >= 0
        # Should be very fast (less than 1 second = 1000ms)
        assert result.elapsed_ms < 1000


class TestCoreRerankerNoneBackend:
    """Test with NoneReranker backend."""
    
    def test_none_backend_preserves_order(self):
        """Test that none backend preserves original order."""
        settings = create_mock_settings(backend="none")
        reranker = CoreReranker(settings)
        
        candidates = [
            create_retrieval_result("1", score=0.9),
            create_retrieval_result("2", score=0.8),
            create_retrieval_result("3", score=0.7),
        ]
        
        result = reranker.rerank("query", candidates)
        
        assert result.fallback is True  # NoneReranker is detected as fallback
        assert [r.chunk_id for r in result.results] == ["1", "2", "3"]
        assert result.backend == "none"
