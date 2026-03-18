#!/usr/bin/env python3
"""
Integration Tests for Hybrid Search (D5)

Tests the HybridSearch orchestrator with mocked D1-D4 components.
Covers normal hybrid retrieval, filtering, and degradation scenarios.
"""

import pytest
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import Mock, MagicMock

from src.core.query_engine.hybrid_search import HybridSearch
from src.core.types import RetrievalResult
from src.core.settings import Settings, LLMSettings, EmbeddingSettings, VectorStoreSettings
from src.core.settings import SplitterSettings, RetrievalSettings, RerankerSettings
from src.core.settings import IngestionSettings, BM25Settings, EvaluationSettings
from src.core.settings import ObservabilitySettings, StorageSettings
from src.core.trace.trace_context import TraceContext


# =============================================================================
# Mock Components for D1-D4
# =============================================================================

@dataclass
class MockProcessedQuery:
    """Mock processed query result."""
    query: str
    keywords: list[str]
    filters: dict[str, Any]


class MockQueryProcessor:
    """Mock Query Processor (D1)."""
    
    def __init__(self, keywords_map: Optional[dict[str, list[str]]] = None):
        self.keywords_map = keywords_map or {}
    
    def process(self, query: str) -> dict[str, Any]:
        """Extract keywords from query."""
        keywords = self.keywords_map.get(query, query.lower().split())
        return {
            'query': query,
            'keywords': keywords,
            'filters': {}
        }


class MockDenseRetriever:
    """Mock Dense Retriever (D2)."""
    
    def __init__(
        self, 
        results_map: Optional[dict[str, list[RetrievalResult]]] = None,
        should_fail: bool = False
    ):
        self.results_map = results_map or {}
        self.should_fail = should_fail
        self.call_count = 0
    
    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict] = None,
        trace: Optional[TraceContext] = None
    ) -> list[RetrievalResult]:
        """Mock dense retrieval."""
        self.call_count += 1
        
        if self.should_fail:
            raise RuntimeError("Dense retriever failed")
        
        if trace:
            trace.record_stage("dense_retrieval", method="mock_dense", details={"query": query})
        
        results = self.results_map.get(query, [])
        
        # Apply filters if provided
        if filters:
            results = [
                r for r in results 
                if all(r.metadata.get(k) == v for k, v in filters.items() if r.metadata.get(k) is not None)
            ]
        
        return results[:top_k]


class MockSparseRetriever:
    """Mock Sparse Retriever (D3)."""
    
    def __init__(
        self,
        results_map: Optional[dict[str, list[RetrievalResult]]] = None,
        should_fail: bool = False
    ):
        self.results_map = results_map or {}
        self.should_fail = should_fail
        self.call_count = 0
    
    def retrieve(
        self,
        keywords: list[str],
        top_k: int,
        trace: Optional[TraceContext] = None
    ) -> list[RetrievalResult]:
        """Mock sparse retrieval."""
        self.call_count += 1
        
        if self.should_fail:
            raise RuntimeError("Sparse retriever failed")
        
        if trace:
            trace.record_stage("sparse_retrieval", method="mock_sparse", details={"keywords": keywords})
        
        # Use first keyword to lookup results
        key = keywords[0] if keywords else ""
        return self.results_map.get(key, [])[:top_k]


class MockRRFFusion:
    """Mock RRF Fusion (D4)."""
    
    def __init__(self, k: int = 60, should_fail: bool = False):
        self.k = k
        self.should_fail = should_fail
        self.call_count = 0
    
    def fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int,
        trace: Optional[TraceContext] = None
    ) -> list[RetrievalResult]:
        """Mock RRF fusion."""
        self.call_count += 1
        
        if self.should_fail:
            raise RuntimeError("Fusion failed")
        
        if trace:
            trace.record_stage("fusion", method="rrf", details={"k": self.k})
        
        # Simple RRF implementation for testing
        scores: dict[str, float] = {}
        result_map: dict[str, RetrievalResult] = {}
        
        # Score dense results by rank
        for rank, result in enumerate(dense_results, 1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0) + 1.0 / (self.k + rank)
            result_map[result.chunk_id] = result
        
        # Score sparse results by rank
        for rank, result in enumerate(sparse_results, 1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0) + 1.0 / (self.k + rank)
            result_map[result.chunk_id] = result
        
        # Sort by score descending
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Return top_k with updated scores
        fused = []
        for chunk_id in sorted_ids[:top_k]:
            original = result_map[chunk_id]
            fused.append(RetrievalResult(
                chunk_id=chunk_id,
                score=scores[chunk_id],
                text=original.text,
                metadata=original.metadata
            ))
        
        return fused


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_settings():
    """Create minimal settings for testing."""
    return Settings(
        llm=LLMSettings(provider="mock"),
        embedding=EmbeddingSettings(provider="mock"),
        vector_store=VectorStoreSettings(provider="mock"),
        splitter=SplitterSettings(strategy="recursive"),
        retrieval=RetrievalSettings(
            dense={},
            sparse={},
            hybrid={"top_k_dense": 10, "top_k_sparse": 10, "fusion_k": 60}
        ),
        reranker=RerankerSettings(backend="none"),
        ingestion=IngestionSettings(),
        bm25=BM25Settings(),
        evaluation=EvaluationSettings(provider="mock"),
        observability=ObservabilitySettings(),
        storage=StorageSettings()
    )


@pytest.fixture
def sample_results():
    """Create sample retrieval results for testing."""
    return {
        "chunk1": RetrievalResult(
            chunk_id="chunk1",
            score=0.9,
            text="Python is a programming language",
            metadata={"collection": "docs", "doc_type": "pdf", "source": "python_intro.pdf"}
        ),
        "chunk2": RetrievalResult(
            chunk_id="chunk2",
            score=0.85,
            text="Java is also a programming language",
            metadata={"collection": "docs", "doc_type": "pdf", "source": "java_intro.pdf"}
        ),
        "chunk3": RetrievalResult(
            chunk_id="chunk3",
            score=0.8,
            text="Python is used for data science",
            metadata={"collection": "tutorials", "doc_type": "md", "source": "data_science.md"}
        ),
        "chunk4": RetrievalResult(
            chunk_id="chunk4",
            score=0.75,
            text="Learning Python basics",
            metadata={"collection": "docs", "doc_type": "html", "source": "basics.html"}
        ),
    }


# =============================================================================
# Test Cases
# =============================================================================

class TestHybridSearchNormal:
    """Test normal hybrid search scenarios."""
    
    def test_basic_hybrid_search(self, mock_settings, sample_results):
        """Test basic hybrid search with both paths returning results."""
        # Setup
        query = "python programming"
        
        dense_results = [sample_results["chunk1"], sample_results["chunk2"]]
        sparse_results = [sample_results["chunk3"], sample_results["chunk4"]]
        
        query_processor = MockQueryProcessor({query: ["python", "programming"]})
        dense_retriever = MockDenseRetriever({query: dense_results})
        sparse_retriever = MockSparseRetriever({"python": sparse_results})
        fusion = MockRRFFusion()
        
        hybrid = HybridSearch(
            settings=mock_settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion
        )
        
        # Execute
        results = hybrid.search(query, top_k=3)
        
        # Verify
        assert len(results) <= 3
        assert fusion.call_count == 1
        assert dense_retriever.call_count == 1
        assert sparse_retriever.call_count == 1
    
    def test_hybrid_search_returns_top_k(self, mock_settings, sample_results):
        """Test that search returns exactly top_k results."""
        query = "python"
        
        # Create many results
        many_results = list(sample_results.values()) * 3  # 12 results
        
        query_processor = MockQueryProcessor()
        dense_retriever = MockDenseRetriever({query: many_results})
        sparse_retriever = MockSparseRetriever({"python": []})
        fusion = MockRRFFusion()
        
        hybrid = HybridSearch(
            settings=mock_settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion
        )
        
        results = hybrid.search(query, top_k=5)
        
        assert len(results) == 5
    
    def test_hybrid_search_with_trace(self, mock_settings, sample_results):
        """Test that trace is recorded properly."""
        query = "python"
        trace = TraceContext(trace_type="query")
        
        dense_results = [sample_results["chunk1"]]
        sparse_results = [sample_results["chunk3"]]
        
        query_processor = MockQueryProcessor()
        dense_retriever = MockDenseRetriever({query: dense_results})
        sparse_retriever = MockSparseRetriever({"python": sparse_results})
        fusion = MockRRFFusion()
        
        hybrid = HybridSearch(
            settings=mock_settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion
        )
        
        results = hybrid.search(query, top_k=2, trace=trace)
        
        # Verify trace was recorded
        stage_names = [s["name"] for s in trace.stages]
        assert "hybrid_search_start" in stage_names
        assert "query_processing" in stage_names
        assert "fusion" in stage_names
        assert "hybrid_search_complete" in stage_names


class TestHybridSearchFiltering:
    """Test metadata filtering scenarios."""
    
    def test_filter_by_collection(self, mock_settings, sample_results):
        """Test filtering results by collection."""
        query = "python"
        
        # All chunks available
        all_results = list(sample_results.values())
        
        query_processor = MockQueryProcessor()
        dense_retriever = MockDenseRetriever({query: all_results})
        sparse_retriever = MockSparseRetriever({"python": []})
        fusion = MockRRFFusion()
        
        hybrid = HybridSearch(
            settings=mock_settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion
        )
        
        # Filter by collection
        results = hybrid.search(query, top_k=10, filters={"collection": "docs"})
        
        # Should only return results from "docs" collection
        assert len(results) == 3  # chunk1, chunk2, chunk4
        for r in results:
            assert r.metadata["collection"] == "docs"
    
    def test_filter_by_doc_type(self, mock_settings, sample_results):
        """Test filtering results by doc_type."""
        query = "python"
        
        all_results = list(sample_results.values())
        
        query_processor = MockQueryProcessor()
        dense_retriever = MockDenseRetriever({query: all_results})
        sparse_retriever = MockSparseRetriever({"python": []})
        fusion = MockRRFFusion()
        
        hybrid = HybridSearch(
            settings=mock_settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion
        )
        
        # Filter by doc_type
        results = hybrid.search(query, top_k=10, filters={"doc_type": "pdf"})
        
        # Should only return PDF results
        assert len(results) == 2  # chunk1, chunk2
        for r in results:
            assert r.metadata["doc_type"] == "pdf"
    
    def test_filter_with_multiple_criteria(self, mock_settings, sample_results):
        """Test filtering with multiple criteria."""
        query = "python"
        
        all_results = list(sample_results.values())
        
        query_processor = MockQueryProcessor()
        dense_retriever = MockDenseRetriever({query: all_results})
        sparse_retriever = MockSparseRetriever({"python": []})
        fusion = MockRRFFusion()
        
        hybrid = HybridSearch(
            settings=mock_settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion
        )
        
        # Filter by both collection and doc_type
        results = hybrid.search(
            query, 
            top_k=10, 
            filters={"collection": "docs", "doc_type": "pdf"}
        )
        
        # Should only return results matching both criteria
        assert len(results) == 2  # chunk1, chunk2
        for r in results:
            assert r.metadata["collection"] == "docs"
            assert r.metadata["doc_type"] == "pdf"
    
    def test_filter_missing_field_includes_result(self, mock_settings):
        """Test that results with missing filter fields are included (loose inclusion)."""
        query = "python"
        
        # Result without the filter field
        result_without_field = RetrievalResult(
            chunk_id="chunk_no_field",
            score=0.9,
            text="Some text",
            metadata={"source": "test.txt"}  # No 'collection' field
        )
        
        query_processor = MockQueryProcessor()
        dense_retriever = MockDenseRetriever({query: [result_without_field]})
        sparse_retriever = MockSparseRetriever({"python": []})
        fusion = MockRRFFusion()
        
        hybrid = HybridSearch(
            settings=mock_settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion
        )
        
        # Filter by collection - should include result without field
        results = hybrid.search(query, top_k=10, filters={"collection": "docs"})
        
        # Result should be included because field is missing (loose inclusion)
        assert len(results) == 1
        assert results[0].chunk_id == "chunk_no_field"


class TestHybridSearchDegradation:
    """Test graceful degradation scenarios."""
    
    def test_dense_fails_fallback_to_sparse(self, mock_settings, sample_results):
        """Test fallback to sparse when dense fails."""
        query = "python"
        
        sparse_results = [sample_results["chunk3"], sample_results["chunk4"]]
        
        query_processor = MockQueryProcessor()
        dense_retriever = MockDenseRetriever(should_fail=True)
        sparse_retriever = MockSparseRetriever({"python": sparse_results})
        fusion = MockRRFFusion()
        
        hybrid = HybridSearch(
            settings=mock_settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion
        )
        
        # Execute - should not raise
        results = hybrid.search(query, top_k=2)
        
        # Should return sparse results
        assert len(results) == 2
        assert results[0].chunk_id == "chunk3"
    
    def test_sparse_fails_fallback_to_dense(self, mock_settings, sample_results):
        """Test fallback to dense when sparse fails."""
        query = "python"
        
        dense_results = [sample_results["chunk1"], sample_results["chunk2"]]
        
        query_processor = MockQueryProcessor()
        dense_retriever = MockDenseRetriever({query: dense_results})
        sparse_retriever = MockSparseRetriever(should_fail=True)
        fusion = MockRRFFusion()
        
        hybrid = HybridSearch(
            settings=mock_settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion
        )
        
        # Execute - should not raise
        results = hybrid.search(query, top_k=2)
        
        # Should return dense results
        assert len(results) == 2
        assert results[0].chunk_id == "chunk1"
    
    def test_fusion_falls_back_to_dense(self, mock_settings, sample_results):
        """Test fallback to dense when fusion fails."""
        query = "python"
        
        dense_results = [sample_results["chunk1"]]
        sparse_results = [sample_results["chunk3"]]
        
        query_processor = MockQueryProcessor()
        dense_retriever = MockDenseRetriever({query: dense_results})
        sparse_retriever = MockSparseRetriever({"python": sparse_results})
        fusion = MockRRFFusion(should_fail=True)
        
        hybrid = HybridSearch(
            settings=mock_settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion
        )
        
        # Execute - should not raise
        results = hybrid.search(query, top_k=2)
        
        # Should return dense results when fusion fails
        assert len(results) == 1
        assert results[0].chunk_id == "chunk1"
    
    def test_both_paths_fail_returns_empty(self, mock_settings):
        """Test empty result when both paths fail."""
        query = "python"
        
        query_processor = MockQueryProcessor()
        dense_retriever = MockDenseRetriever(should_fail=True)
        sparse_retriever = MockSparseRetriever(should_fail=True)
        fusion = MockRRFFusion()
        
        hybrid = HybridSearch(
            settings=mock_settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion
        )
        
        # Execute - should not raise, return empty
        results = hybrid.search(query, top_k=2)
        
        assert len(results) == 0
    
    def test_no_sparse_results_uses_dense(self, mock_settings, sample_results):
        """Test using only dense when sparse returns empty."""
        query = "python"
        
        dense_results = [sample_results["chunk1"], sample_results["chunk2"]]
        
        query_processor = MockQueryProcessor()
        dense_retriever = MockDenseRetriever({query: dense_results})
        sparse_retriever = MockSparseRetriever({"python": []})  # Empty results
        fusion = MockRRFFusion()
        
        hybrid = HybridSearch(
            settings=mock_settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion
        )
        
        results = hybrid.search(query, top_k=2)
        
        # Should use dense results directly (no fusion needed)
        assert len(results) == 2
        assert fusion.call_count == 0  # Fusion not called when one path empty


class TestHybridSearchEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_query(self, mock_settings):
        """Test handling of empty query."""
        query_processor = MockQueryProcessor()
        dense_retriever = MockDenseRetriever()
        sparse_retriever = MockSparseRetriever()
        fusion = MockRRFFusion()
        
        hybrid = HybridSearch(
            settings=mock_settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion
        )
        
        # Execute with empty query
        results = hybrid.search("", top_k=5)
        
        assert len(results) == 0
    
    def test_no_components_configured(self, mock_settings):
        """Test search with no retrievers configured."""
        hybrid = HybridSearch(settings=mock_settings)
        
        results = hybrid.search("test", top_k=5)
        
        assert len(results) == 0
    
    def test_no_keywords_extracted(self, mock_settings, sample_results):
        """Test when query processor returns no keywords."""
        query = "test"
        
        # Query processor returns empty keywords
        query_processor = MockQueryProcessor({query: []})
        dense_retriever = MockDenseRetriever({query: [sample_results["chunk1"]]})
        sparse_retriever = MockSparseRetriever()  # Won't be called with empty keywords
        fusion = MockRRFFusion()
        
        hybrid = HybridSearch(
            settings=mock_settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion
        )
        
        results = hybrid.search(query, top_k=5)
        
        # Should only use dense (sparse skipped with no keywords)
        assert len(results) == 1
        assert sparse_retriever.call_count == 0
    
    def test_query_processor_failure_uses_fallback(self, mock_settings, sample_results):
        """Test fallback when query processor fails."""
        query = "python programming"
        
        # Query processor that raises
        query_processor = Mock()
        query_processor.process = Mock(side_effect=Exception("Processor failed"))
        
        dense_retriever = MockDenseRetriever({query: [sample_results["chunk1"]]})
        sparse_retriever = MockSparseRetriever()
        fusion = MockRRFFusion()
        
        hybrid = HybridSearch(
            settings=mock_settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion
        )
        
        # Execute - should not raise
        results = hybrid.search(query, top_k=5)
        
        # Should use fallback keyword extraction (split by space)
        assert len(results) == 1
        # Sparse retriever should be called with split keywords
        assert sparse_retriever.call_count == 1
    
    def test_duplicate_chunk_ids_in_fusion(self, mock_settings):
        """Test handling of duplicate chunk IDs from both paths."""
        query = "python"
        
        # Same result from both paths
        common_result = RetrievalResult(
            chunk_id="common",
            score=0.9,
            text="Common text",
            metadata={}
        )
        
        query_processor = MockQueryProcessor()
        dense_retriever = MockDenseRetriever({query: [common_result]})
        sparse_retriever = MockSparseRetriever({"python": [common_result]})
        fusion = MockRRFFusion()
        
        hybrid = HybridSearch(
            settings=mock_settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion
        )
        
        results = hybrid.search(query, top_k=5)
        
        # Should deduplicate
        assert len(results) == 1
        assert results[0].chunk_id == "common"
        # Score should be combined from both paths
        assert results[0].score > 0.03  # 1/60 + 1/60 ≈ 0.033
    
    def test_top_k_larger_than_results(self, mock_settings, sample_results):
        """Test when top_k is larger than available results."""
        query = "python"
        
        query_processor = MockQueryProcessor()
        dense_retriever = MockDenseRetriever({query: [sample_results["chunk1"]]})
        sparse_retriever = MockSparseRetriever()
        fusion = MockRRFFusion()
        
        hybrid = HybridSearch(
            settings=mock_settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion
        )
        
        results = hybrid.search(query, top_k=100)
        
        # Should return all available results
        assert len(results) == 1


class TestHybridSearchResultFormat:
    """Test result format and content."""
    
    def test_result_contains_chunk_text(self, mock_settings, sample_results):
        """Test that results contain chunk text."""
        query = "python"
        
        query_processor = MockQueryProcessor()
        dense_retriever = MockDenseRetriever({query: [sample_results["chunk1"]]})
        sparse_retriever = MockSparseRetriever()
        fusion = MockRRFFusion()
        
        hybrid = HybridSearch(
            settings=mock_settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion
        )
        
        results = hybrid.search(query, top_k=5)
        
        assert len(results) > 0
        assert results[0].text == "Python is a programming language"
    
    def test_result_contains_metadata(self, mock_settings, sample_results):
        """Test that results contain metadata."""
        query = "python"
        
        query_processor = MockQueryProcessor()
        dense_retriever = MockDenseRetriever({query: [sample_results["chunk1"]]})
        sparse_retriever = MockSparseRetriever()
        fusion = MockRRFFusion()
        
        hybrid = HybridSearch(
            settings=mock_settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion
        )
        
        results = hybrid.search(query, top_k=5)
        
        assert len(results) > 0
        assert results[0].metadata["collection"] == "docs"
        assert results[0].metadata["doc_type"] == "pdf"
    
    def test_result_contains_score(self, mock_settings, sample_results):
        """Test that results contain relevance score."""
        query = "python"
        
        query_processor = MockQueryProcessor()
        dense_retriever = MockDenseRetriever({query: [sample_results["chunk1"]]})
        sparse_retriever = MockSparseRetriever()
        fusion = MockRRFFusion()
        
        hybrid = HybridSearch(
            settings=mock_settings,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion
        )
        
        results = hybrid.search(query, top_k=5)
        
        assert len(results) > 0
        assert isinstance(results[0].score, float)
        assert results[0].score > 0


# =============================================================================
# Integration with Real Components (for when D1-D4 are implemented)
# =============================================================================

@pytest.mark.skip(reason="Run after D1-D4 are implemented")
class TestHybridSearchRealIntegration:
    """Integration tests with real D1-D4 components."""
    
    def test_full_pipeline_with_real_components(self):
        """Test full pipeline when all components are available."""
        # This test should be enabled after D1-D4 are implemented
        pass
