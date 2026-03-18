#!/usr/bin/env python3
"""
Embedding Factory Tests

Tests for embedding factory routing and provider creation.
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

from core.settings import Settings, EmbeddingSettings
from libs.embedding.embedding_factory import EmbeddingFactory, EmbeddingProvider
from libs.embedding.base_embedding import BaseEmbedding


class FakeEmbedding(BaseEmbedding):
    """Fake embedding for testing."""
    
    def embed(self, texts, trace=None):
        # Return fake vectors with configured dimensions
        dim = self.dimensions or 8
        return [[0.1] * dim for _ in texts]
    
    async def aembed(self, texts, trace=None):
        return self.embed(texts, trace)


class TestEmbeddingFactory:
    """Test embedding factory routing logic."""
    
    def test_factory_routes_to_openai(self):
        """Test factory routes to OpenAI provider."""
        settings = Mock(spec=Settings)
        settings.embedding = Mock(spec=EmbeddingSettings)
        settings.embedding.provider = "openai"
        settings.embedding.openai = {"api_key": "test", "model": "text-embedding-3-small"}
        
        with patch.dict('sys.modules', {'libs.embedding.openai_embedding': Mock(OpenAIEmbedding=FakeEmbedding)}):
            embedding = EmbeddingFactory.create(settings)
            assert isinstance(embedding, FakeEmbedding)
    
    def test_factory_routes_to_dashscope(self):
        """Test factory routes to DashScope provider."""
        settings = Mock(spec=Settings)
        settings.embedding = Mock(spec=EmbeddingSettings)
        settings.embedding.provider = "dashscope"
        settings.embedding.dashscope = {"api_key": "test", "model": "text-embedding-v4"}
        
        with patch.dict('sys.modules', {'libs.embedding.dashscope_embedding': Mock(DashScopeEmbedding=FakeEmbedding)}):
            embedding = EmbeddingFactory.create(settings)
            assert isinstance(embedding, FakeEmbedding)
    
    def test_factory_raises_on_anthropic(self):
        """Test factory raises error for Anthropic (no embedding service)."""
        settings = Mock(spec=Settings)
        settings.embedding = Mock(spec=EmbeddingSettings)
        settings.embedding.provider = "anthropic"
        
        with pytest.raises(ValueError) as exc_info:
            EmbeddingFactory.create(settings)
        
        assert "anthropic" in str(exc_info.value).lower()
        assert "embedding" in str(exc_info.value).lower()
    
    def test_factory_routes_to_litellm(self):
        """Test factory routes to LiteLLM provider."""
        settings = Mock(spec=Settings)
        settings.embedding = Mock(spec=EmbeddingSettings)
        settings.embedding.provider = "litellm"
        settings.embedding.litellm = {"api_key": "test", "model": "text-embedding-3-small"}
        
        with patch.dict('sys.modules', {'libs.embedding.litellm_embedding': Mock(LiteLLMEmbedding=FakeEmbedding)}):
            embedding = EmbeddingFactory.create(settings)
            assert isinstance(embedding, FakeEmbedding)
    
    def test_factory_raises_on_unknown_provider(self):
        """Test factory raises error for unknown provider."""
        settings = Mock(spec=Settings)
        settings.embedding = Mock(spec=EmbeddingSettings)
        settings.embedding.provider = "unknown_provider"
        
        with pytest.raises(ValueError) as exc_info:
            EmbeddingFactory.create(settings)
        
        assert "unknown_provider" in str(exc_info.value).lower()


class TestEmbeddingProvider:
    """Test embedding provider wrapper."""
    
    def test_embed_single(self):
        """Test embed_single convenience method."""
        embedding = FakeEmbedding({"model": "fake", "dimensions": 10})
        
        result = embedding.embed_single("test text")
        
        assert len(result) == 10
        assert all(v == 0.1 for v in result)
    
    def test_embed_single_empty_config(self):
        """Test embed_single with no dimensions config."""
        embedding = FakeEmbedding({"model": "fake"})
        
        result = embedding.embed_single("test text")
        
        assert len(result) == 8  # Default from FakeEmbedding
    
    def test_provider_embed_with_trace(self):
        """Test provider wrapper with trace."""
        embedding = FakeEmbedding({"model": "fake", "dimensions": 10, "batch_size": 5})
        provider = EmbeddingProvider(embedding)
        
        from core.trace.trace_context import TraceContext
        trace = TraceContext(trace_type="test")
        
        results = provider.embed_with_trace(["text1", "text2"], trace=trace)
        
        assert len(results) == 2
        assert len(trace.stages) == 1
        assert trace.stages[0]["name"] == "embedding"
        assert trace.stages[0]["details"]["text_count"] == 2
    
    def test_provider_embed_batches(self):
        """Test batch embedding."""
        embedding = FakeEmbedding({"model": "fake", "dimensions": 4, "batch_size": 3})
        provider = EmbeddingProvider(embedding)
        
        texts = ["text1", "text2", "text3", "text4", "text5"]
        results = provider.embed_batches(texts)
        
        assert len(results) == 5
        assert all(len(v) == 4 for v in results)
    
    def test_provider_embed_batches_empty(self):
        """Test batch embedding with empty list."""
        embedding = FakeEmbedding({"model": "fake", "batch_size": 3})
        provider = EmbeddingProvider(embedding)
        
        results = provider.embed_batches([])
        
        assert results == []
    
    def test_provider_embed_batches_single_batch(self):
        """Test batch embedding when all fits in one batch."""
        embedding = FakeEmbedding({"model": "fake", "dimensions": 4, "batch_size": 10})
        provider = EmbeddingProvider(embedding)
        
        texts = ["text1", "text2"]
        results = provider.embed_batches(texts)
        
        assert len(results) == 2
