#!/usr/bin/env python3
"""
Dense Retriever Unit Tests

Tests for DenseRetriever using mocked EmbeddingClient and VectorStore.
No real embedding API calls are made.
"""

import sys
from pathlib import Path
from typing import Optional
from unittest.mock import Mock, MagicMock

import pytest

# Add src to path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from core.settings import Settings, EmbeddingSettings, VectorStoreSettings
from core.types import RetrievalResult
from core.trace.trace_context import TraceContext
from core.query_engine.dense_retriever import DenseRetriever
from libs.vector_store.base_vector_store import QueryResult


class MockEmbeddingClient:
    """Mock embedding client for testing."""
    
    def __init__(self, dimension: int = 3):
        self.dimension = dimension
        self.embed_call_count = 0
        self.last_texts = None
    
    def embed(self, texts: list[str], trace=None) -> list[list[float]]:
        """Return deterministic mock embeddings."""
        self.embed_call_count += 1
        self.last_texts = texts
        # Return deterministic vectors based on text hash
        results = []
        for text in texts:
            # Simple deterministic vector generation
            base = hash(text) % 1000 / 1000.0
            vector = [base + i * 0.1 for i in range(self.dimension)]
            results.append(vector)
        return results
    
    async def aembed(self, texts: list[str], trace=None) -> list[list[float]]:
        """Async version returns same as embed."""
        return self.embed(texts, trace)


class MockVectorStore:
    """Mock vector store for testing."""
    
    def __init__(self):
        self.query_call_count = 0
        self.last_query_vector = None
        self.last_top_k = None
        self.last_filters = None
        self.mock_results = []
    
    def set_mock_results(self, results: list[QueryResult]) -> None:
        """Set the mock results to return from query."""
        self.mock_results = results
    
    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: Optional[dict] = None,
        trace=None
    ) -> list[QueryResult]:
        """Return mock query results."""
        self.query_call_count += 1
        self.last_query_vector = vector
        self.last_top_k = top_k
        self.last_filters = filters
        return self.mock_results[:top_k]
    
    def upsert(self, records, trace=None) -> None:
        pass
    
    def delete(self, ids, trace=None) -> None:
        pass
    
    def get_by_ids(self, ids, trace=None) -> list[dict]:
        return []
    
    def count(self) -> int:
        return 0


class TestDenseRetriever:
    """Test DenseRetriever functionality."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock(spec=Settings)
        settings.embedding = Mock(spec=EmbeddingSettings)
        settings.embedding.provider = "mock"
        settings.vector_store = Mock(spec=VectorStoreSettings)
        settings.vector_store.provider = "mock"
        return settings
    
    @pytest.fixture
    def mock_embedding(self):
        """Create mock embedding client."""
        return MockEmbeddingClient(dimension=3)
    
    @pytest.fixture
    def mock_store(self):
        """Create mock vector store."""
        return MockVectorStore()
    
    def test_retrieve_basic(self, mock_settings, mock_embedding, mock_store):
        """Test basic retrieval flow."""
        # Setup mock results
        mock_results = [
            QueryResult(id="chunk1", score=0.95, text="text1", metadata={"doc_id": "doc1"}),
            QueryResult(id="chunk2", score=0.85, text="text2", metadata={"doc_id": "doc2"}),
        ]
        mock_store.set_mock_results(mock_results)
        
        # Create retriever with injected dependencies
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=mock_embedding,
            vector_store=mock_store
        )
        
        # Execute retrieval
        results = retriever.retrieve(query="test query", top_k=2)
        
        # Verify embedding was called
        assert mock_embedding.embed_call_count == 1
        assert mock_embedding.last_texts == ["test query"]
        
        # Verify vector store was called
        assert mock_store.query_call_count == 1
        assert mock_store.last_top_k == 2
        assert mock_store.last_query_vector is not None
        
        # Verify results
        assert len(results) == 2
        assert isinstance(results[0], RetrievalResult)
        assert results[0].chunk_id == "chunk1"
        assert results[0].score == 0.95
        assert results[0].text == "text1"
        assert results[0].metadata == {"doc_id": "doc1"}
        
        assert results[1].chunk_id == "chunk2"
        assert results[1].score == 0.85
        assert results[1].text == "text2"
        assert results[1].metadata == {"doc_id": "doc2"}
    
    def test_retrieve_with_filters(self, mock_settings, mock_embedding, mock_store):
        """Test retrieval with metadata filters."""
        mock_results = [
            QueryResult(id="chunk1", score=0.95, text="filtered text", metadata={"category": "A"}),
        ]
        mock_store.set_mock_results(mock_results)
        
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=mock_embedding,
            vector_store=mock_store
        )
        
        filters = {"category": "A"}
        results = retriever.retrieve(query="test", top_k=1, filters=filters)
        
        # Verify filters were passed
        assert mock_store.last_filters == filters
        assert len(results) == 1
        assert results[0].text == "filtered text"
    
    def test_retrieve_with_trace(self, mock_settings, mock_embedding, mock_store):
        """Test retrieval with trace context."""
        mock_results = [
            QueryResult(id="chunk1", score=0.9, text="text1", metadata={}),
        ]
        mock_store.set_mock_results(mock_results)
        
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=mock_embedding,
            vector_store=mock_store
        )
        
        trace = TraceContext(trace_type="query")
        results = retriever.retrieve(query="test", top_k=1, trace=trace)
        
        # Verify trace was passed to dependencies (they would record stages)
        assert len(results) == 1
    
    def test_retrieve_empty_results(self, mock_settings, mock_embedding, mock_store):
        """Test retrieval with no results."""
        mock_store.set_mock_results([])
        
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=mock_embedding,
            vector_store=mock_store
        )
        
        results = retriever.retrieve(query="test", top_k=5)
        
        assert len(results) == 0
        assert mock_embedding.embed_call_count == 1
        assert mock_store.query_call_count == 1
    
    def test_retrieve_respects_top_k(self, mock_settings, mock_embedding, mock_store):
        """Test that top_k parameter is respected."""
        mock_results = [
            QueryResult(id=f"chunk{i}", score=0.9 - i*0.1, text=f"text{i}", metadata={})
            for i in range(10)
        ]
        mock_store.set_mock_results(mock_results)
        
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=mock_embedding,
            vector_store=mock_store
        )
        
        results = retriever.retrieve(query="test", top_k=3)
        
        assert len(results) == 3
        assert mock_store.last_top_k == 3
    
    def test_retrieve_embedding_to_store_vector_flow(self, mock_settings, mock_embedding, mock_store):
        """Test that embedding output flows correctly to vector store input."""
        mock_results = [
            QueryResult(id="chunk1", score=0.9, text="text1", metadata={}),
        ]
        mock_store.set_mock_results(mock_results)
        
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=mock_embedding,
            vector_store=mock_store
        )
        
        retriever.retrieve(query="test query", top_k=1)
        
        # Verify the vector passed to store matches embedding output
        expected_vector = mock_embedding.embed(["test query"])[0]
        assert mock_store.last_query_vector == expected_vector
    
    def test_retrieve_complex_metadata(self, mock_settings, mock_embedding, mock_store):
        """Test retrieval with complex metadata structures."""
        mock_results = [
            QueryResult(
                id="chunk1",
                score=0.95,
                text="text with metadata",
                metadata={
                    "doc_id": "doc1",
                    "source": "/path/to/doc.pdf",
                    "nested": {"key": "value"},
                    "tags": ["tag1", "tag2"]
                }
            ),
        ]
        mock_store.set_mock_results(mock_results)
        
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=mock_embedding,
            vector_store=mock_store
        )
        
        results = retriever.retrieve(query="test", top_k=1)
        
        assert len(results) == 1
        assert results[0].metadata["doc_id"] == "doc1"
        assert results[0].metadata["nested"]["key"] == "value"
        assert results[0].metadata["tags"] == ["tag1", "tag2"]


class TestDenseRetrieverRetrievalResult:
    """Test RetrievalResult transformation from QueryResult."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock(spec=Settings)
        settings.embedding = Mock(spec=EmbeddingSettings)
        settings.embedding.provider = "mock"
        settings.vector_store = Mock(spec=VectorStoreSettings)
        settings.vector_store.provider = "mock"
        return settings
    
    def test_query_result_to_retrieval_result_mapping(self, mock_settings):
        """Test that QueryResult fields map correctly to RetrievalResult."""
        mock_embedding = MockEmbeddingClient()
        mock_store = MockVectorStore()
        
        mock_results = [
            QueryResult(
                id="test_chunk_123",
                score=0.987,
                text="This is the retrieved text content.",
                metadata={"source": "test.pdf", "page": 5}
            ),
        ]
        mock_store.set_mock_results(mock_results)
        
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=mock_embedding,
            vector_store=mock_store
        )
        
        results = retriever.retrieve(query="test", top_k=1)
        
        assert len(results) == 1
        result = results[0]
        
        # Verify field mapping
        assert result.chunk_id == "test_chunk_123"  # QueryResult.id -> RetrievalResult.chunk_id
        assert result.score == 0.987
        assert result.text == "This is the retrieved text content."
        assert result.metadata == {"source": "test.pdf", "page": 5}
    
    def test_retrieval_result_serialization(self, mock_settings):
        """Test that RetrievalResult can be serialized to dict."""
        mock_embedding = MockEmbeddingClient()
        mock_store = MockVectorStore()
        
        mock_results = [
            QueryResult(
                id="chunk1",
                score=0.9,
                text="test text",
                metadata={"key": "value"}
            ),
        ]
        mock_store.set_mock_results(mock_results)
        
        retriever = DenseRetriever(
            settings=mock_settings,
            embedding_client=mock_embedding,
            vector_store=mock_store
        )
        
        results = retriever.retrieve(query="test", top_k=1)
        
        # Test serialization
        data = results[0].to_dict()
        assert data["chunk_id"] == "chunk1"
        assert data["score"] == 0.9
        assert data["text"] == "test text"
        assert data["metadata"] == {"key": "value"}
        
        # Test deserialization
        restored = RetrievalResult.from_dict(data)
        assert restored.chunk_id == "chunk1"
        assert restored.score == 0.9
        assert restored.text == "test text"
        assert restored.metadata == {"key": "value"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
