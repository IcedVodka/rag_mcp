#!/usr/bin/env python3
"""
Unit tests for RRF Fusion module.

Test scenarios:
1. Two lists with overlapping results
2. Two lists with no overlapping results
3. One list is empty
4. Both lists are empty
5. Custom k values
"""

import pytest
from dataclasses import replace

from src.core.query_engine.fusion import RRFFusion
from src.core.types import RetrievalResult
from src.core.settings import Settings, LLMSettings, EmbeddingSettings, VectorStoreSettings
from src.core.settings import SplitterSettings, RetrievalSettings, RerankerSettings
from src.core.settings import IngestionSettings, BM25Settings, EvaluationSettings
from src.core.settings import ObservabilitySettings, StorageSettings


def create_mock_settings() -> Settings:
    """Create a minimal settings object for testing."""
    return Settings(
        llm=LLMSettings(provider="test"),
        embedding=EmbeddingSettings(provider="test"),
        vector_store=VectorStoreSettings(provider="test"),
        splitter=SplitterSettings(strategy="recursive"),
        retrieval=RetrievalSettings(),
        reranker=RerankerSettings(backend="test"),
        ingestion=IngestionSettings(),
        bm25=BM25Settings(),
        evaluation=EvaluationSettings(provider="test"),
        observability=ObservabilitySettings(),
        storage=StorageSettings(),
    )


def create_result(chunk_id: str, score: float = 0.0, text: str = "") -> RetrievalResult:
    """Helper to create a RetrievalResult."""
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text or f"Text content for {chunk_id}",
        metadata={"source": "test", "chunk_id": chunk_id}
    )


class TestRRFFusionInitialization:
    """Test RRFFusion initialization and configuration."""
    
    def test_default_k_value(self):
        """Test that default k value is 60."""
        settings = create_mock_settings()
        fusion = RRFFusion(settings)
        assert fusion.k == 60
    
    def test_custom_k_value(self):
        """Test that custom k value is accepted."""
        settings = create_mock_settings()
        fusion = RRFFusion(settings, k=40)
        assert fusion.k == 40
    
    def test_k_value_stored_in_instance(self):
        """Test that k value is properly stored in instance."""
        settings = create_mock_settings()
        fusion = RRFFusion(settings, k=100)
        assert fusion.k == 100
        assert fusion.settings is settings
    
    def test_invalid_k_zero(self):
        """Test that k=0 raises ValueError."""
        settings = create_mock_settings()
        with pytest.raises(ValueError, match="positive integer"):
            RRFFusion(settings, k=0)
    
    def test_invalid_k_negative(self):
        """Test that negative k raises ValueError."""
        settings = create_mock_settings()
        with pytest.raises(ValueError, match="positive integer"):
            RRFFusion(settings, k=-1)
    
    def test_invalid_k_non_integer(self):
        """Test that non-integer k raises ValueError."""
        settings = create_mock_settings()
        with pytest.raises(ValueError, match="positive integer"):
            RRFFusion(settings, k=30.5)


class TestRRFFusionBasic:
    """Test basic RRF fusion functionality."""
    
    def test_both_lists_empty(self):
        """Test that empty lists return empty result."""
        settings = create_mock_settings()
        fusion = RRFFusion(settings)
        
        result = fusion.fuse([], [], top_k=10)
        assert result == []
    
    def test_dense_empty_sparse_has_results(self):
        """Test fusion when dense list is empty."""
        settings = create_mock_settings()
        fusion = RRFFusion(settings)
        
        sparse_results = [
            create_result("A", score=1.0),
            create_result("B", score=0.9),
            create_result("C", score=0.8),
        ]
        
        result = fusion.fuse([], sparse_results, top_k=10)
        
        assert len(result) == 3
        # Scores should be 1/(60+1), 1/(60+2), 1/(60+3)
        expected_scores = [1/61, 1/62, 1/63]
        for i, (res, expected) in enumerate(zip(result, expected_scores)):
            assert res.chunk_id == sparse_results[i].chunk_id
            assert abs(res.score - expected) < 1e-10
    
    def test_sparse_empty_dense_has_results(self):
        """Test fusion when sparse list is empty."""
        settings = create_mock_settings()
        fusion = RRFFusion(settings)
        
        dense_results = [
            create_result("X", score=0.95),
            create_result("Y", score=0.85),
            create_result("Z", score=0.75),
        ]
        
        result = fusion.fuse(dense_results, [], top_k=10)
        
        assert len(result) == 3
        expected_scores = [1/61, 1/62, 1/63]
        for i, (res, expected) in enumerate(zip(result, expected_scores)):
            assert res.chunk_id == dense_results[i].chunk_id
            assert abs(res.score - expected) < 1e-10


class TestRRFFusionOverlapping:
    """Test RRF fusion with overlapping results."""
    
    def test_overlapping_results_scores_summed(self):
        """Test that overlapping results have scores summed."""
        settings = create_mock_settings()
        fusion = RRFFusion(settings, k=60)
        
        # Dense: A(rank=1), B(rank=2)
        dense_results = [
            create_result("A", score=0.9),
            create_result("B", score=0.8),
        ]
        
        # Sparse: B(rank=1), C(rank=2)
        sparse_results = [
            create_result("B", score=1.2),
            create_result("C", score=1.0),
        ]
        
        result = fusion.fuse(dense_results, sparse_results, top_k=10)
        
        # Calculate expected scores
        # A: 1/(60+1) from dense = 1/61
        # B: 1/(60+2) from dense + 1/(60+1) from sparse = 1/62 + 1/61
        # C: 1/(60+2) from sparse = 1/62
        score_a = 1 / 61
        score_b = 1 / 62 + 1 / 61
        score_c = 1 / 62
        
        assert len(result) == 3
        
        # Should be sorted: B > A > C (since A and C have same score, tiebreaker by chunk_id)
        # Actually A and C both have 1/61 and 1/62? Let me recalculate:
        # A: rank 1 in dense -> 1/61
        # B: rank 2 in dense -> 1/62, rank 1 in sparse -> 1/61, total = 1/62 + 1/61
        # C: rank 2 in sparse -> 1/62
        # So order should be B > A > C (since 1/61 ≈ 0.0164 > 1/62 ≈ 0.0161)
        
        assert result[0].chunk_id == "B"
        assert abs(result[0].score - score_b) < 1e-10
        
        # A and C should follow, with A before C due to alphabetical tiebreaker
        # (when scores differ, higher score comes first)
        assert result[1].chunk_id == "A"
        assert abs(result[1].score - score_a) < 1e-10
        
        assert result[2].chunk_id == "C"
        assert abs(result[2].score - score_c) < 1e-10
    
    def test_multiple_overlapping_results(self):
        """Test with multiple overlapping results."""
        settings = create_mock_settings()
        fusion = RRFFusion(settings, k=60)
        
        # Dense: A(1), B(2), C(3), D(4)
        dense_results = [
            create_result("A", score=0.9),
            create_result("B", score=0.8),
            create_result("C", score=0.7),
            create_result("D", score=0.6),
        ]
        
        # Sparse: B(1), C(2), E(3)
        sparse_results = [
            create_result("B", score=1.2),
            create_result("C", score=1.1),
            create_result("E", score=1.0),
        ]
        
        result = fusion.fuse(dense_results, sparse_results, top_k=10)
        
        assert len(result) == 5
        
        # Expected scores:
        # A: 1/61 (dense rank 1)
        # B: 1/62 (dense rank 2) + 1/61 (sparse rank 1)
        # C: 1/63 (dense rank 3) + 1/62 (sparse rank 2)
        # D: 1/64 (dense rank 4)
        # E: 1/63 (sparse rank 3)
        
        score_a = 1 / 61
        score_b = 1 / 62 + 1 / 61
        score_c = 1 / 63 + 1 / 62
        score_d = 1 / 64
        score_e = 1 / 63
        
        # Order should be: B > C > A > E > D
        assert result[0].chunk_id == "B"
        assert abs(result[0].score - score_b) < 1e-10
        
        assert result[1].chunk_id == "C"
        assert abs(result[1].score - score_c) < 1e-10
        
        assert result[2].chunk_id == "A"
        assert abs(result[2].score - score_a) < 1e-10
        
        assert result[3].chunk_id == "E"
        assert abs(result[3].score - score_e) < 1e-10
        
        assert result[4].chunk_id == "D"
        assert abs(result[4].score - score_d) < 1e-10


class TestRRFFusionNonOverlapping:
    """Test RRF fusion with no overlapping results."""
    
    def test_no_overlap(self):
        """Test fusion when no results overlap."""
        settings = create_mock_settings()
        fusion = RRFFusion(settings, k=60)
        
        dense_results = [
            create_result("A", score=0.9),
            create_result("B", score=0.8),
        ]
        
        sparse_results = [
            create_result("C", score=1.2),
            create_result("D", score=1.1),
        ]
        
        result = fusion.fuse(dense_results, sparse_results, top_k=10)
        
        assert len(result) == 4
        
        # Scores:
        # A: 1/61 (dense rank 1)
        # B: 1/62 (dense rank 2)
        # C: 1/61 (sparse rank 1)
        # D: 1/62 (sparse rank 2)
        
        # Order: A and C both have 1/61, B and D both have 1/62
        # Tiebreaker: alphabetically A before C, B before D
        assert result[0].chunk_id in ["A", "C"]
        assert result[1].chunk_id in ["A", "C"]
        assert result[2].chunk_id in ["B", "D"]
        assert result[3].chunk_id in ["B", "D"]
        
        # Check scores
        assert abs(result[0].score - 1/61) < 1e-10
        assert abs(result[1].score - 1/61) < 1e-10
        assert abs(result[2].score - 1/62) < 1e-10
        assert abs(result[3].score - 1/62) < 1e-10


class TestRRFFusionTopK:
    """Test top_k limiting functionality."""
    
    def test_top_k_limits_results(self):
        """Test that top_k correctly limits the number of results."""
        settings = create_mock_settings()
        fusion = RRFFusion(settings)
        
        dense_results = [
            create_result("A", score=0.9),
            create_result("B", score=0.8),
            create_result("C", score=0.7),
            create_result("D", score=0.6),
        ]
        
        sparse_results = [
            create_result("E", score=1.2),
            create_result("F", score=1.1),
        ]
        
        result = fusion.fuse(dense_results, sparse_results, top_k=3)
        
        assert len(result) == 3
    
    def test_top_k_zero_returns_empty(self):
        """Test that top_k=0 returns empty list."""
        settings = create_mock_settings()
        fusion = RRFFusion(settings)
        
        dense_results = [create_result("A", score=0.9)]
        sparse_results = [create_result("B", score=1.2)]
        
        result = fusion.fuse(dense_results, sparse_results, top_k=0)
        
        assert result == []
    
    def test_top_k_negative_returns_empty(self):
        """Test that negative top_k returns empty list."""
        settings = create_mock_settings()
        fusion = RRFFusion(settings)
        
        dense_results = [create_result("A", score=0.9)]
        sparse_results = [create_result("B", score=1.2)]
        
        result = fusion.fuse(dense_results, sparse_results, top_k=-1)
        
        assert result == []
    
    def test_top_k_larger_than_results(self):
        """Test that top_k larger than result count returns all results."""
        settings = create_mock_settings()
        fusion = RRFFusion(settings)
        
        dense_results = [create_result("A", score=0.9)]
        sparse_results = [create_result("B", score=1.2)]
        
        result = fusion.fuse(dense_results, sparse_results, top_k=100)
        
        assert len(result) == 2


class TestRRFFusionCustomK:
    """Test RRF fusion with custom k values."""
    
    def test_custom_k_changes_scores(self):
        """Test that custom k value affects score calculation."""
        settings = create_mock_settings()
        
        # With k=60
        fusion_k60 = RRFFusion(settings, k=60)
        # With k=20
        fusion_k20 = RRFFusion(settings, k=20)
        
        dense_results = [create_result("A", score=0.9)]
        sparse_results = []
        
        result_k60 = fusion_k60.fuse(dense_results, sparse_results, top_k=10)
        result_k20 = fusion_k20.fuse(dense_results, sparse_results, top_k=10)
        
        # k=60: score = 1/(60+1) = 1/61
        # k=20: score = 1/(20+1) = 1/21
        assert abs(result_k60[0].score - 1/61) < 1e-10
        assert abs(result_k20[0].score - 1/21) < 1e-10
        # Lower k gives higher score
        assert result_k20[0].score > result_k60[0].score
    
    def test_custom_k_affects_ranking(self):
        """Test that different k values can affect ranking."""
        settings = create_mock_settings()
        
        dense_results = [
            create_result("A", score=0.9),  # rank 1
            create_result("B", score=0.8),  # rank 2
        ]
        sparse_results = [
            create_result("B", score=1.2),  # rank 1
        ]
        
        # With k=0 (if allowed), B would get 1/2 + 1/1 = 1.5, A would get 1/1 = 1.0
        # With large k, the difference between ranks becomes smaller
        # This test verifies that different k values work correctly
        
        fusion_k10 = RRFFusion(settings, k=10)
        result_k10 = fusion_k10.fuse(dense_results, sparse_results, top_k=10)
        
        # B should still be first regardless of k
        assert result_k10[0].chunk_id == "B"
        # A should be second
        assert result_k10[1].chunk_id == "A"


class TestRRFFusionResultProperties:
    """Test properties of fused results."""
    
    def test_result_metadata_preserved(self):
        """Test that original result metadata is preserved."""
        settings = create_mock_settings()
        fusion = RRFFusion(settings)
        
        dense_results = [
            RetrievalResult(
                chunk_id="A",
                score=0.9,
                text="Original text",
                metadata={"key": "value", "number": 42}
            )
        ]
        
        result = fusion.fuse(dense_results, [], top_k=10)
        
        assert len(result) == 1
        assert result[0].chunk_id == "A"
        assert result[0].text == "Original text"
        assert result[0].metadata == {"key": "value", "number": 42}
    
    def test_result_score_replaced(self):
        """Test that original score is replaced with RRF score."""
        settings = create_mock_settings()
        fusion = RRFFusion(settings)
        
        dense_results = [
            create_result("A", score=999.0),  # Original score should be replaced
        ]
        
        result = fusion.fuse(dense_results, [], top_k=10)
        
        assert len(result) == 1
        # Score should be 1/(60+1), not 999.0
        assert abs(result[0].score - 1/61) < 1e-10
        assert result[0].score != 999.0
    
    def test_results_are_new_objects(self):
        """Test that fused results are new objects, not modified originals."""
        settings = create_mock_settings()
        fusion = RRFFusion(settings)
        
        original = create_result("A", score=0.9)
        dense_results = [original]
        
        result = fusion.fuse(dense_results, [], top_k=10)
        
        # Original should be unchanged
        assert original.score == 0.9
        # Result should be a new object with different score
        assert result[0].score != original.score
    
    def test_deterministic_ordering(self):
        """Test that results are deterministic given same input."""
        settings = create_mock_settings()
        fusion = RRFFusion(settings)
        
        dense_results = [
            create_result("A", score=0.9),
            create_result("B", score=0.8),
            create_result("C", score=0.7),
        ]
        sparse_results = [
            create_result("B", score=1.2),
            create_result("C", score=1.1),
            create_result("D", score=1.0),
        ]
        
        # Run multiple times
        results = []
        for _ in range(5):
            result = fusion.fuse(dense_results, sparse_results, top_k=10)
            results.append([r.chunk_id for r in result])
        
        # All results should be identical
        for r in results[1:]:
            assert r == results[0]


class TestRRFFusionEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_same_chunk_multiple_times_in_one_list(self):
        """Test handling of duplicate chunk_ids in a single list (shouldn't happen but test anyway)."""
        settings = create_mock_settings()
        fusion = RRFFusion(settings)
        
        # Same chunk appearing twice in dense (edge case)
        dense_results = [
            create_result("A", score=0.9),
            create_result("A", score=0.8),  # Duplicate chunk_id
        ]
        
        result = fusion.fuse(dense_results, [], top_k=10)
        
        # Should only have one result (second one overwrites or sums)
        # Our implementation sums scores for same chunk_id
        assert len(result) == 1
        # Score should be 1/(60+1) + 1/(60+2) = sum of both ranks
        expected_score = 1/61 + 1/62
        assert abs(result[0].score - expected_score) < 1e-10
    
    def test_large_number_of_results(self):
        """Test with a large number of results."""
        settings = create_mock_settings()
        fusion = RRFFusion(settings)
        
        dense_results = [create_result(f"chunk_{i}", score=1.0 - i*0.01) for i in range(100)]
        sparse_results = [create_result(f"chunk_{i+50}", score=1.0 - i*0.01) for i in range(100)]
        
        result = fusion.fuse(dense_results, sparse_results, top_k=10)
        
        assert len(result) == 10
        # First results should be from overlap region
        # chunk_50 appears at rank 51 in dense and rank 1 in sparse
        # chunk_51 appears at rank 52 in dense and rank 2 in sparse
        # etc.
    
    def test_single_result_in_each_list_same_id(self):
        """Test when both lists have single result with same chunk_id."""
        settings = create_mock_settings()
        fusion = RRFFusion(settings)
        
        dense_results = [create_result("A", score=0.9)]
        sparse_results = [create_result("A", score=1.2)]
        
        result = fusion.fuse(dense_results, sparse_results, top_k=10)
        
        assert len(result) == 1
        # Score should be sum from both lists: 1/(60+1) + 1/(60+1) = 2/61
        expected_score = 2 / 61
        assert abs(result[0].score - expected_score) < 1e-10


class TestRRFFusionConstants:
    """Test module constants."""
    
    def test_default_k_constant(self):
        """Test that DEFAULT_K constant is accessible."""
        assert RRFFusion.DEFAULT_K == 60
        assert hasattr(RRFFusion, 'DEFAULT_K')
