#!/usr/bin/env python3
"""
Sparse Retriever Unit Tests

Tests the SparseRetriever class using mock BM25Indexer and VectorStore.
No real index data is required.
"""

from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock, Mock
import pytest

from core.query_engine.sparse_retriever import SparseRetriever
from core.types import RetrievalResult
from core.trace.trace_context import TraceContext


class MockVectorStore:
    """Mock vector store for testing."""
    
    def __init__(self, records: Optional[Dict[str, Dict[str, Any]]] = None):
        self.records = records or {}
    
    def get_by_ids(
        self,
        ids: List[str],
        trace: Optional[TraceContext] = None
    ) -> List[Dict[str, Any]]:
        """Mock get_by_ids method."""
        result = []
        for id_ in ids:
            if id_ in self.records:
                record = self.records[id_].copy()
                record["id"] = id_
                result.append(record)
        return result


class MockBM25Indexer:
    """Mock BM25 indexer for testing."""
    
    def __init__(self, query_results: Optional[Dict[str, List[tuple]]] = None):
        self.query_results = query_results or {}
        self._stats = {"N": 0, "avgdl": 0.0, "num_terms": 0}
    
    def query(
        self,
        keywords: List[str],
        top_k: int = 10,
        case_sensitive: bool = False
    ) -> List[tuple]:
        """Mock query method."""
        key = " ".join(keywords).lower()
        results = self.query_results.get(key, [])
        return results[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """Mock get_stats method."""
        return self._stats.copy()


class TestSparseRetrieverInit:
    """Test SparseRetriever initialization."""
    
    def test_init_with_mocks(self):
        """Test initialization with mock dependencies."""
        mock_indexer = MockBM25Indexer()
        mock_store = MockVectorStore()
        
        retriever = SparseRetriever(
            bm25_indexer=mock_indexer,
            vector_store=mock_store
        )
        
        assert retriever.bm25_indexer is mock_indexer
        assert retriever.vector_store is mock_store
    
    def test_init_with_settings(self):
        """Test initialization with settings only."""
        settings = {
            "bm25_index_path": "/tmp/test_bm25",
            "bm25_k1": 2.0,
            "bm25_b": 0.5
        }
        
        retriever = SparseRetriever(settings=settings)
        
        assert retriever.settings == settings
        assert retriever.bm25_indexer is not None
        assert retriever.vector_store is None
    
    def test_init_default_settings(self):
        """Test initialization with default settings."""
        retriever = SparseRetriever()
        
        assert retriever.settings == {}
        assert retriever.bm25_indexer is not None
        assert retriever.vector_store is None


class TestSparseRetrieverRetrieve:
    """Test sparse retrieval functionality."""
    
    def test_retrieve_single_result(self):
        """Test retrieving a single result."""
        # Setup mock data
        mock_indexer = MockBM25Indexer({
            "hello": [("chunk_1", 1.5)]
        })
        mock_store = MockVectorStore({
            "chunk_1": {"text": "Hello world", "metadata": {"source": "doc1.txt"}}
        })
        
        retriever = SparseRetriever(
            bm25_indexer=mock_indexer,
            vector_store=mock_store
        )
        
        # Execute
        results = retriever.retrieve(["hello"], top_k=10)
        
        # Verify
        assert len(results) == 1
        assert results[0].chunk_id == "chunk_1"
        assert results[0].text == "Hello world"
        assert results[0].score == 1.5
        assert results[0].metadata["source"] == "doc1.txt"
    
    def test_retrieve_multiple_results(self):
        """Test retrieving multiple results sorted by score."""
        # Setup mock data - scores intentionally unsorted
        mock_indexer = MockBM25Indexer({
            "machine learning": [
                ("chunk_2", 0.8),
                ("chunk_1", 1.2),
                ("chunk_3", 0.5)
            ]
        })
        mock_store = MockVectorStore({
            "chunk_1": {"text": "Machine learning basics", "metadata": {"page": 1}},
            "chunk_2": {"text": "Deep learning advanced", "metadata": {"page": 2}},
            "chunk_3": {"text": "AI overview", "metadata": {"page": 3}}
        })
        
        retriever = SparseRetriever(
            bm25_indexer=mock_indexer,
            vector_store=mock_store
        )
        
        # Execute
        results = retriever.retrieve(["machine", "learning"], top_k=10)
        
        # Verify - should be sorted by score descending
        assert len(results) == 3
        assert results[0].chunk_id == "chunk_1"
        assert results[0].score == 1.2
        assert results[1].chunk_id == "chunk_2"
        assert results[1].score == 0.8
        assert results[2].chunk_id == "chunk_3"
        assert results[2].score == 0.5
    
    def test_retrieve_empty_keywords(self):
        """Test retrieving with empty keywords."""
        mock_indexer = MockBM25Indexer()
        mock_store = MockVectorStore()
        
        retriever = SparseRetriever(
            bm25_indexer=mock_indexer,
            vector_store=mock_store
        )
        
        results = retriever.retrieve([])
        
        assert results == []
    
    def test_retrieve_no_matches(self):
        """Test retrieving when no matches found."""
        mock_indexer = MockBM25Indexer()
        mock_store = MockVectorStore()
        
        retriever = SparseRetriever(
            bm25_indexer=mock_indexer,
            vector_store=mock_store
        )
        
        results = retriever.retrieve(["nonexistent"])
        
        assert results == []
    
    def test_retrieve_top_k_limit(self):
        """Test that top_k limits the number of results."""
        mock_indexer = MockBM25Indexer({
            "test": [
                ("chunk_1", 1.0),
                ("chunk_2", 0.9),
                ("chunk_3", 0.8),
                ("chunk_4", 0.7),
                ("chunk_5", 0.6)
            ]
        })
        mock_store = MockVectorStore({
            f"chunk_{i}": {"text": f"Text {i}", "metadata": {}}
            for i in range(1, 6)
        })
        
        retriever = SparseRetriever(
            bm25_indexer=mock_indexer,
            vector_store=mock_store
        )
        
        results = retriever.retrieve(["test"], top_k=3)
        
        assert len(results) == 3
    
    def test_retrieve_missing_vector_store(self):
        """Test that retrieve raises error when vector_store is None."""
        mock_indexer = MockBM25Indexer({
            "test": [("chunk_1", 1.0)]
        })
        
        retriever = SparseRetriever(bm25_indexer=mock_indexer)
        
        with pytest.raises(ValueError, match="Vector store not initialized"):
            retriever.retrieve(["test"])
    
    def test_retrieve_with_trace(self):
        """Test retrieval with trace context."""
        mock_indexer = MockBM25Indexer({
            "hello": [("chunk_1", 1.5)]
        })
        mock_store = MockVectorStore({
            "chunk_1": {"text": "Hello world", "metadata": {}}
        })
        
        retriever = SparseRetriever(
            bm25_indexer=mock_indexer,
            vector_store=mock_store
        )
        
        trace = TraceContext()
        results = retriever.retrieve(["hello"], top_k=5, trace=trace)
        
        # Verify trace recorded stages
        assert len(trace.stages) >= 1
        assert trace.stages[0]["name"] == "sparse_retrieval"
        assert trace.stages[0]["details"]["keywords"] == ["hello"]
        assert trace.stages[0]["details"]["top_k"] == 5
    
    def test_retrieve_result_type(self):
        """Test that retrieve returns RetrievalResult objects."""
        mock_indexer = MockBM25Indexer({
            "test": [("chunk_1", 1.0)]
        })
        mock_store = MockVectorStore({
            "chunk_1": {"text": "Test text", "metadata": {"key": "value"}}
        })
        
        retriever = SparseRetriever(
            bm25_indexer=mock_indexer,
            vector_store=mock_store
        )
        
        results = retriever.retrieve(["test"])
        
        assert len(results) == 1
        assert isinstance(results[0], RetrievalResult)
        assert results[0].chunk_id == "chunk_1"
        assert results[0].text == "Test text"
        assert results[0].metadata == {"key": "value"}
    
    def test_retrieve_handles_missing_metadata(self):
        """Test that retrieve handles records without metadata."""
        mock_indexer = MockBM25Indexer({
            "test": [("chunk_1", 1.0)]
        })
        mock_store = MockVectorStore({
            "chunk_1": {"text": "Test text"}  # No metadata
        })
        
        retriever = SparseRetriever(
            bm25_indexer=mock_indexer,
            vector_store=mock_store
        )
        
        results = retriever.retrieve(["test"])
        
        assert len(results) == 1
        assert results[0].metadata == {}


class TestSparseRetrieverStats:
    """Test statistics methods."""
    
    def test_get_index_stats(self):
        """Test getting index statistics."""
        mock_indexer = MockBM25Indexer()
        mock_indexer._stats = {
            "N": 100,
            "avgdl": 150.5,
            "num_terms": 5000
        }
        
        retriever = SparseRetriever(bm25_indexer=mock_indexer)
        stats = retriever.get_index_stats()
        
        assert stats["N"] == 100
        assert stats["avgdl"] == 150.5
        assert stats["num_terms"] == 5000


class TestSparseRetrieverIntegration:
    """Integration-style tests with more realistic mocks."""
    
    def test_full_retrieval_workflow(self):
        """Test complete retrieval workflow."""
        # Setup realistic mock data
        bm25_data = {
            "machine learning": [
                ("doc_001_chunk_0", 2.35),
                ("doc_001_chunk_3", 1.87),
                ("doc_002_chunk_1", 1.52)
            ]
        }
        
        store_data = {
            "doc_001_chunk_0": {
                "text": "Machine learning is a subset of artificial intelligence.",
                "metadata": {
                    "source": "intro_to_ml.pdf",
                    "page": 1,
                    "chunk_index": 0
                }
            },
            "doc_001_chunk_3": {
                "text": "Supervised learning is the most common type of machine learning.",
                "metadata": {
                    "source": "intro_to_ml.pdf",
                    "page": 2,
                    "chunk_index": 3
                }
            },
            "doc_002_chunk_1": {
                "text": "Deep learning uses neural networks with multiple layers.",
                "metadata": {
                    "source": "deep_learning.pdf",
                    "page": 1,
                    "chunk_index": 1
                }
            }
        }
        
        mock_indexer = MockBM25Indexer(bm25_data)
        mock_store = MockVectorStore(store_data)
        
        retriever = SparseRetriever(
            bm25_indexer=mock_indexer,
            vector_store=mock_store
        )
        
        # Execute retrieval
        results = retriever.retrieve(["machine", "learning"], top_k=3)
        
        # Verify
        assert len(results) == 3
        
        # Highest score first
        assert results[0].chunk_id == "doc_001_chunk_0"
        assert results[0].score == 2.35
        assert "machine learning" in results[0].text.lower()
        
        # Verify metadata preserved
        assert results[1].metadata["source"] == "intro_to_ml.pdf"
        assert results[2].metadata["source"] == "deep_learning.pdf"
    
    def test_retrieve_with_partial_store_match(self):
        """Test when BM25 returns chunk_ids but some are not in vector store."""
        mock_indexer = MockBM25Indexer({
            "test": [
                ("chunk_1", 1.0),
                ("chunk_2", 0.8),  # Not in store
                ("chunk_3", 0.6)
            ]
        })
        mock_store = MockVectorStore({
            "chunk_1": {"text": "Text 1", "metadata": {}},
            "chunk_3": {"text": "Text 3", "metadata": {}}
            # chunk_2 is missing
        })
        
        retriever = SparseRetriever(
            bm25_indexer=mock_indexer,
            vector_store=mock_store
        )
        
        results = retriever.retrieve(["test"])
        
        # Only return chunks that exist in store
        assert len(results) == 2
        assert {r.chunk_id for r in results} == {"chunk_1", "chunk_3"}
