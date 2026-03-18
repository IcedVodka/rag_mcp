#!/usr/bin/env python3
"""
Sparse Encoder Unit Tests

Tests for SparseEncoder BM25 term statistics generation.
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

# Add src to path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from core.types import Chunk, ChunkRecord
from core.trace.trace_context import TraceContext
from ingestion.embedding.sparse_encoder import SparseEncoder


class TestSparseEncoderInit:
    """Test SparseEncoder initialization."""
    
    def test_default_initialization(self):
        """Test encoder initializes with default settings."""
        encoder = SparseEncoder()
        
        assert encoder.tokenizer == "jieba"
        assert encoder.use_stopwords is True
        assert len(encoder.stopwords) > 0
        assert encoder._jieba is not None
    
    def test_simple_tokenizer_initialization(self):
        """Test encoder initializes with simple tokenizer."""
        encoder = SparseEncoder(tokenizer="simple")
        
        assert encoder.tokenizer == "simple"
        assert encoder._jieba is None
    
    def test_stopwords_disabled(self):
        """Test encoder with stopwords disabled."""
        encoder = SparseEncoder(stopwords=False)
        
        assert encoder.use_stopwords is False
        assert len(encoder.stopwords) == 0
    
    def test_custom_stopwords(self):
        """Test encoder with custom stopwords."""
        custom = {"custom_word", "another_word"}
        encoder = SparseEncoder(custom_stopwords=custom)
        
        assert "custom_word" in encoder.stopwords
        assert "another_word" in encoder.stopwords
        assert "the" in encoder.stopwords  # Default stopwords still present
    
    def test_settings_override(self):
        """Test that settings can override defaults."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.sparse_encoder = {
            "tokenizer": "simple",
            "stopwords": False
        }
        
        encoder = SparseEncoder(settings=mock_settings)
        
        assert encoder.tokenizer == "simple"
        assert encoder.use_stopwords is False


class TestSparseEncoderTokenization:
    """Test tokenization functionality."""
    
    def test_simple_tokenizer_english(self):
        """Test simple tokenizer with English text."""
        encoder = SparseEncoder(tokenizer="simple", stopwords=False)
        
        text = "Hello world! This is a test."
        tokens = encoder._tokenize(text)
        
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        # All should be lowercase
        assert all(t.islower() or not t.isalpha() for t in tokens)
    
    def test_simple_tokenizer_removes_punctuation(self):
        """Test simple tokenizer removes punctuation."""
        encoder = SparseEncoder(tokenizer="simple", stopwords=False)
        
        text = "Hello, world! How are you?"
        tokens = encoder._tokenize(text)
        
        assert "," not in tokens
        assert "!" not in tokens
        assert "?" not in tokens
        assert "hello" in tokens
        assert "world" in tokens
    
    def test_jieba_tokenizer_chinese(self):
        """Test jieba tokenizer with Chinese text."""
        encoder = SparseEncoder(tokenizer="jieba", stopwords=False)
        
        text = "自然语言处理是人工智能的重要领域"
        tokens = encoder._tokenize(text)
        
        # Should produce multiple tokens
        assert len(tokens) > 0
        # All tokens should be non-empty
        assert all(len(t) > 0 for t in tokens)
    
    def test_jieba_tokenizer_mixed(self):
        """Test jieba tokenizer with mixed Chinese and English."""
        encoder = SparseEncoder(tokenizer="jieba", stopwords=False)
        
        text = "Hello世界，这是一个test"
        tokens = encoder._tokenize(text)
        
        assert len(tokens) > 0
    
    def test_tokenize_empty_text(self):
        """Test tokenization of empty text."""
        encoder = SparseEncoder()
        
        assert encoder._tokenize("") == []
        assert encoder._tokenize("   ") == []
        assert encoder._tokenize(None) == []  # type: ignore
    
    def test_tokenize_single_word(self):
        """Test tokenization of single word."""
        encoder = SparseEncoder(tokenizer="simple", stopwords=False)
        
        tokens = encoder._tokenize("hello")
        assert tokens == ["hello"]


class TestSparseEncoderTermStats:
    """Test term statistics computation."""
    
    def test_compute_term_stats_basic(self):
        """Test basic term frequency computation."""
        encoder = SparseEncoder(tokenizer="simple", stopwords=False)
        
        tokens = ["hello", "world", "hello"]
        stats = encoder._compute_term_stats(tokens)
        
        assert stats["hello"] == 2
        assert stats["world"] == 1
    
    def test_compute_term_stats_with_stopwords(self):
        """Test term stats with stopword filtering."""
        encoder = SparseEncoder(tokenizer="simple", stopwords=True)
        
        tokens = ["the", "hello", "the", "world", "the"]
        stats = encoder._compute_term_stats(tokens)
        
        # "the" should be filtered out
        assert "the" not in stats
        assert stats["hello"] == 1
        assert stats["world"] == 1
    
    def test_compute_term_stats_empty(self):
        """Test term stats with empty tokens."""
        encoder = SparseEncoder()
        
        stats = encoder._compute_term_stats([])
        assert stats == {}
    
    def test_compute_term_stats_for_text_basic(self):
        """Test complete term stats computation for text."""
        encoder = SparseEncoder(tokenizer="simple", stopwords=False)
        
        text = "hello world hello"
        result = encoder._compute_term_stats_for_text(text)
        
        assert result["terms"] == ["hello", "world"]
        assert result["tf"] == [2, 1]
        assert result["doc_length"] == 3
    
    def test_compute_term_stats_for_text_empty(self):
        """Test term stats for empty text."""
        encoder = SparseEncoder()
        
        result = encoder._compute_term_stats_for_text("")
        
        assert result["terms"] == []
        assert result["tf"] == []
        assert result["doc_length"] == 0
    
    def test_compute_term_stats_doc_length_with_stopwords(self):
        """Test doc_length calculation with stopword filtering."""
        encoder = SparseEncoder(tokenizer="simple", stopwords=True)
        
        # "the" is a stopword
        text = "the hello the world"
        result = encoder._compute_term_stats_for_text(text)
        
        # doc_length should only count non-stopwords
        assert result["doc_length"] == 2
        assert "hello" in result["terms"]
        assert "world" in result["terms"]
        assert "the" not in result["terms"]


class TestSparseEncoderEncode:
    """Test main encode functionality."""
    
    def test_encode_output_count_matches_input(self):
        """Test encoder returns same number of records as input chunks."""
        encoder = SparseEncoder(tokenizer="simple", stopwords=False)
        
        chunks = [
            Chunk(id="c1", text="First chunk", metadata={"doc_id": "d1"}),
            Chunk(id="c2", text="Second chunk", metadata={"doc_id": "d1"}),
            Chunk(id="c3", text="Third chunk", metadata={"doc_id": "d2"}),
        ]
        
        records = encoder.encode(chunks)
        
        assert len(records) == len(chunks)
        assert all(isinstance(r, ChunkRecord) for r in records)
    
    def test_encode_preserves_chunk_metadata(self):
        """Test encoder preserves chunk metadata in records."""
        encoder = SparseEncoder(tokenizer="simple", stopwords=False)
        
        chunks = [
            Chunk(
                id="c1",
                text="Test content",
                metadata={"custom_key": "custom_value", "page": 1},
                start_offset=0,
                end_offset=12,
                source_ref="doc1"
            ),
        ]
        
        records = encoder.encode(chunks)
        
        assert len(records) == 1
        record = records[0]
        assert record.id == "c1"
        assert record.text == "Test content"
        assert record.metadata["custom_key"] == "custom_value"
        assert record.metadata["page"] == 1
        assert record.metadata["start_offset"] == 0
        assert record.metadata["end_offset"] == 12
        assert record.metadata["source_ref"] == "doc1"
    
    def test_encode_sparse_vector_structure(self):
        """Test sparse vector has correct structure."""
        encoder = SparseEncoder(tokenizer="simple", stopwords=False)
        
        chunks = [Chunk(id="c1", text="hello world hello")]
        records = encoder.encode(chunks)
        
        sparse_vector = records[0].sparse_vector
        assert sparse_vector is not None
        assert "terms" in sparse_vector
        assert "tf" in sparse_vector
        assert "doc_length" in sparse_vector
        assert isinstance(sparse_vector["terms"], list)
        assert isinstance(sparse_vector["tf"], list)
        assert isinstance(sparse_vector["doc_length"], int)
    
    def test_encode_dense_vector_is_none(self):
        """Test DenseEncoder sets dense_vector to None."""
        encoder = SparseEncoder(tokenizer="simple", stopwords=False)
        
        chunks = [Chunk(id="c1", text="Test")]
        records = encoder.encode(chunks)
        
        assert len(records) == 1
        assert records[0].dense_vector is None
    
    def test_encode_empty_input_returns_empty_list(self):
        """Test encoder returns empty list for empty input."""
        encoder = SparseEncoder()
        
        records = encoder.encode([])
        
        assert records == []
    
    def test_encode_batch_empty_input_returns_empty_list(self):
        """Test encode_batch returns empty list for empty input."""
        encoder = SparseEncoder()
        
        records = encoder.encode_batch([])
        
        assert records == []
    
    def test_encode_empty_text_chunk(self):
        """Test encoding chunk with empty text."""
        encoder = SparseEncoder()
        
        chunks = [Chunk(id="c1", text="")]
        records = encoder.encode(chunks)
        
        assert len(records) == 1
        assert records[0].sparse_vector["terms"] == []
        assert records[0].sparse_vector["tf"] == []
        assert records[0].sparse_vector["doc_length"] == 0
    
    def test_encode_whitespace_only_text(self):
        """Test encoding chunk with whitespace-only text."""
        encoder = SparseEncoder()
        
        chunks = [Chunk(id="c1", text="   \n\t  ")]
        records = encoder.encode(chunks)
        
        assert len(records) == 1
        assert records[0].sparse_vector["doc_length"] == 0
    
    def test_output_order_matches_input(self):
        """Test output records maintain same order as input chunks."""
        encoder = SparseEncoder(tokenizer="simple", stopwords=False)
        
        chunks = [
            Chunk(id="first", text="First content"),
            Chunk(id="second", text="Second content"),
            Chunk(id="third", text="Third content"),
        ]
        
        records = encoder.encode(chunks)
        
        assert [r.id for r in records] == ["first", "second", "third"]
        assert [r.text for r in records] == ["First content", "Second content", "Third content"]


class TestSparseEncoderChinese:
    """Test Chinese text processing."""
    
    def test_chinese_tokenization(self):
        """Test Chinese text tokenization with jieba."""
        encoder = SparseEncoder(tokenizer="jieba", stopwords=False)
        
        text = "自然语言处理是人工智能的重要领域"
        tokens = encoder._tokenize(text)
        
        # Should tokenize into meaningful terms
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
    
    def test_chinese_with_stopwords(self):
        """Test Chinese text with Chinese stopwords."""
        encoder = SparseEncoder(tokenizer="jieba", stopwords=True)
        
        # "的" and "是" are Chinese stopwords
        text = "这书是我的"
        result = encoder._compute_term_stats_for_text(text)
        
        # "的" should be filtered
        assert "的" not in result["terms"]
    
    def test_chinese_english_mixed(self):
        """Test mixed Chinese and English text."""
        encoder = SparseEncoder(tokenizer="jieba", stopwords=False)
        
        text = "Hello世界，这是一个RAG系统test"
        result = encoder._compute_term_stats_for_text(text)
        
        assert result["doc_length"] > 0
        assert len(result["terms"]) > 0


class TestSparseEncoderTracing:
    """Test trace context integration."""
    
    def test_encode_records_trace(self):
        """Test trace recording during encoding."""
        encoder = SparseEncoder(tokenizer="simple", stopwords=True)
        
        chunks = [
            Chunk(id="c1", text="Chunk 1"),
            Chunk(id="c2", text="Chunk 2"),
        ]
        
        trace = TraceContext(trace_type="test")
        records = encoder.encode(chunks, trace=trace)
        
        # Check trace was recorded
        assert len(trace.stages) == 1
        stage = trace.stages[0]
        assert stage["name"] == "sparse_encoding"
        assert stage["method"] == "tokenize_and_count"
        assert "tokenizer" in stage["details"]
        assert stage["details"]["chunk_count"] == 2
        assert "stopwords_enabled" in stage["details"]
    
    def test_encode_no_trace_does_not_fail(self):
        """Test encoding works without trace context."""
        encoder = SparseEncoder(tokenizer="simple")
        
        chunks = [Chunk(id="c1", text="Test")]
        
        # Should not raise
        records = encoder.encode(chunks, trace=None)
        
        assert len(records) == 1
        assert records[0].sparse_vector is not None


class TestSparseEncoderStopwords:
    """Test stopword filtering functionality."""
    
    def test_english_stopwords_filtered(self):
        """Test English stopwords are filtered."""
        encoder = SparseEncoder(tokenizer="simple", stopwords=True)
        
        text = "the quick brown fox jumps over the lazy dog"
        result = encoder._compute_term_stats_for_text(text)
        
        # Common stopwords should be filtered
        assert "the" not in result["terms"]
        assert "over" not in result["terms"]
        # Content words should remain
        assert "quick" in result["terms"]
        assert "brown" in result["terms"]
        assert "fox" in result["terms"]
    
    def test_stopwords_disabled(self):
        """Test stopwords are kept when disabled."""
        encoder = SparseEncoder(tokenizer="simple", stopwords=False)
        
        text = "the quick brown fox"
        result = encoder._compute_term_stats_for_text(text)
        
        # All words including stopwords should be present
        assert "the" in result["terms"]
        assert "quick" in result["terms"]
    
    def test_chinese_stopwords_filtered(self):
        """Test Chinese stopwords are filtered."""
        encoder = SparseEncoder(tokenizer="jieba", stopwords=True)
        
        # "的", "是", "我" are Chinese stopwords
        text = "我的书是红色的"
        result = encoder._compute_term_stats_for_text(text)
        
        assert "的" not in result["terms"]
        assert "是" not in result["terms"]


class TestSparseEncoderGetStats:
    """Test get_stats method."""
    
    def test_get_stats_structure(self):
        """Test get_stats returns expected structure."""
        encoder = SparseEncoder(tokenizer="jieba", stopwords=True)
        
        stats = encoder.get_stats()
        
        assert "tokenizer" in stats
        assert "use_stopwords" in stats
        assert "stopword_count" in stats
        assert "jieba_available" in stats
        assert stats["tokenizer"] == "jieba"
        assert stats["use_stopwords"] is True
        assert stats["jieba_available"] is True
    
    def test_get_stats_simple_tokenizer(self):
        """Test get_stats with simple tokenizer."""
        encoder = SparseEncoder(tokenizer="simple", stopwords=False)
        
        stats = encoder.get_stats()
        
        assert stats["tokenizer"] == "simple"
        assert stats["use_stopwords"] is False
        assert stats["jieba_available"] is False


class TestSparseEncoderEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_chunk_encoding(self):
        """Test encoding a single chunk."""
        encoder = SparseEncoder(tokenizer="simple", stopwords=False)
        
        chunks = [Chunk(id="single", text="Only one chunk")]
        
        records = encoder.encode(chunks)
        
        assert len(records) == 1
        assert records[0].id == "single"
        assert records[0].sparse_vector is not None
    
    def test_chunk_with_empty_metadata(self):
        """Test encoding chunk with empty metadata."""
        encoder = SparseEncoder(tokenizer="simple")
        
        chunks = [Chunk(id="c1", text="Test")]  # Empty metadata by default
        
        records = encoder.encode(chunks)
        
        assert records[0].metadata is not None
        assert "start_offset" in records[0].metadata
    
    def test_large_batch_processing(self):
        """Test encoding large number of chunks."""
        encoder = SparseEncoder(tokenizer="simple", stopwords=False)
        
        chunks = [Chunk(id=f"c{i}", text=f"Text {i} content here") for i in range(100)]
        
        records = encoder.encode(chunks)
        
        assert len(records) == 100
        assert all(r.sparse_vector is not None for r in records)
    
    def test_repeated_terms(self):
        """Test handling of repeated terms."""
        encoder = SparseEncoder(tokenizer="simple", stopwords=False)
        
        text = "foo foo foo bar bar baz"
        result = encoder._compute_term_stats_for_text(text)
        
        # Check tf values
        tf_dict = dict(zip(result["terms"], result["tf"]))
        assert tf_dict["foo"] == 3
        assert tf_dict["bar"] == 2
        assert tf_dict["baz"] == 1
        assert result["doc_length"] == 6
    
    def test_special_characters(self):
        """Test handling of special characters."""
        encoder = SparseEncoder(tokenizer="simple", stopwords=False)
        
        text = "hello@world #test $100 50%"
        tokens = encoder._tokenize(text)
        
        # Special chars should be removed or normalized
        assert "@" not in tokens
        assert "#" not in tokens
        assert "$" not in tokens
        assert "%" not in tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
