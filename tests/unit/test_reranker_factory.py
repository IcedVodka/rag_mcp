#!/usr/bin/env python3
"""
Reranker Factory Tests

Tests for reranker factory routing and backend creation.
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

from core.settings import Settings, RerankerSettings
from libs.reranker.reranker_factory import RerankerFactory, RerankerProvider
from libs.reranker.base_reranker import (
    BaseReranker, NoneReranker, RerankCandidate, RerankResult, RerankerError
)


class FakeReranker(BaseReranker):
    """Fake reranker for testing."""
    
    def rerank(self, query, candidates, trace=None):
        # Reverse the order and assign new scores
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
        raise RerankerError("Simulated failure")
    
    async def arerank(self, query, candidates, trace=None):
        raise RerankerError("Simulated failure")


class TestRerankerFactory:
    """Test reranker factory routing logic."""
    
    def test_factory_routes_to_none(self):
        """Test factory routes to none (no-op) backend."""
        settings = Mock(spec=Settings)
        settings.reranker = Mock(spec=RerankerSettings)
        settings.reranker.backend = "none"
        settings.reranker.none = {}
        
        reranker = RerankerFactory.create(settings)
        assert isinstance(reranker, NoneReranker)
    
    def test_factory_routes_to_cross_encoder(self):
        """Test factory routes to cross_encoder backend."""
        settings = Mock(spec=Settings)
        settings.reranker = Mock(spec=RerankerSettings)
        settings.reranker.backend = "cross_encoder"
        settings.reranker.cross_encoder = {"model": "cross-encoder/ms-marco"}
        
        with patch.dict('sys.modules', {'libs.reranker.cross_encoder_reranker': Mock(CrossEncoderReranker=FakeReranker)}):
            reranker = RerankerFactory.create(settings)
            assert isinstance(reranker, FakeReranker)
    
    def test_factory_routes_to_llm(self):
        """Test factory routes to llm backend."""
        settings = Mock(spec=Settings)
        settings.reranker = Mock(spec=RerankerSettings)
        settings.reranker.backend = "llm"
        settings.reranker.llm = {"prompt_path": "config/prompts/rerank.txt"}
        
        with patch.dict('sys.modules', {'libs.reranker.llm_reranker': Mock(LLMReranker=FakeReranker)}):
            reranker = RerankerFactory.create(settings)
            assert isinstance(reranker, FakeReranker)
    
    def test_factory_raises_on_unknown_backend(self):
        """Test factory raises error for unknown backend."""
        settings = Mock(spec=Settings)
        settings.reranker = Mock(spec=RerankerSettings)
        settings.reranker.backend = "unknown_backend"
        
        with pytest.raises(ValueError) as exc_info:
            RerankerFactory.create(settings)
        
        assert "unknown_backend" in str(exc_info.value).lower()


class TestNoneReranker:
    """Test NoneReranker (no-op) behavior."""
    
    def test_none_reranker_preserves_order(self):
        """Test NoneReranker preserves original order."""
        reranker = NoneReranker({})
        
        candidates = [
            RerankCandidate(id="1", text="text1", score=0.9, metadata={}),
            RerankCandidate(id="2", text="text2", score=0.8, metadata={}),
        ]
        
        results = reranker.rerank("query", candidates)
        
        assert len(results) == 2
        assert results[0].id == "1"
        assert results[1].id == "2"
        assert results[0].rerank_score == 0.9  # Same as original
    
    def test_none_reranker_empty_candidates(self):
        """Test NoneReranker with empty candidates."""
        reranker = NoneReranker({})
        
        results = reranker.rerank("query", [])
        
        assert results == []


class TestRerankerProvider:
    """Test reranker provider wrapper."""
    
    def test_reranker_provider_success(self):
        """Test provider with successful rerank."""
        reranker = FakeReranker({})
        provider = RerankerProvider(reranker, fallback_on_error=True)
        
        candidates = [
            RerankCandidate(id="1", text="text1", score=0.5, metadata={}),
            RerankCandidate(id="2", text="text2", score=0.6, metadata={}),
        ]
        
        results = provider.rerank_with_fallback("query", candidates)
        
        # FakeReranker reverses order
        assert results[0].id == "2"
        assert results[1].id == "1"
    
    def test_reranker_provider_fallback_on_error(self):
        """Test provider fallback on error."""
        error_reranker = ErrorReranker({})
        provider = RerankerProvider(error_reranker, fallback_on_error=True)
        
        candidates = [
            RerankCandidate(id="1", text="text1", score=0.9, metadata={}),
            RerankCandidate(id="2", text="text2", score=0.8, metadata={}),
        ]
        
        # Should not raise, should fallback
        results = provider.rerank_with_fallback("query", candidates)
        
        # Fallback preserves original order
        assert results[0].id == "1"
        assert results[1].id == "2"
    
    def test_reranker_provider_no_fallback_raises(self):
        """Test provider raises error when fallback disabled."""
        error_reranker = ErrorReranker({})
        provider = RerankerProvider(error_reranker, fallback_on_error=False)
        
        candidates = [RerankCandidate(id="1", text="text1", score=0.5, metadata={})]
        
        with pytest.raises(RerankerError):
            provider.rerank_with_fallback("query", candidates)
