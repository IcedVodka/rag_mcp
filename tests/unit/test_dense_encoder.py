#!/usr/bin/env python3
"""
Dense Encoder Unit Tests

Tests for DenseEncoder batch encoding functionality.
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
from ingestion.embedding.dense_encoder import DenseEncoder


class FakeEmbedding:
    """Fake embedding client for testing."""
    
    def __init__(self, dimensions: int = 8, batch_size: int = 3):
        self.config = {"model": "fake-embedding", "dimensions": dimensions, "batch_size": batch_size}
        self.model = "fake-embedding"
        self.dimensions = dimensions
        self.batch_size = batch_size
    
    def embed(self, texts, trace=None):
        """Return fake embeddings with configured dimensions."""
        return [[0.1 + i * 0.01] * self.dimensions for i in range(len(texts))]
    
    async def aembed(self, texts, trace=None):
        """Async version of embed."""
        return self.embed(texts, trace)


class TestDenseEncoder:
    """Test DenseEncoder functionality."""
    
    def test_encode_output_count_matches_input(self):
        """Test encoder returns same number of records as input chunks."""
        fake_embedding = FakeEmbedding(dimensions=8)
        encoder = DenseEncoder(settings=Mock(), embedding_client=fake_embedding)
        
        chunks = [
            Chunk(id="c1", text="First chunk", metadata={"doc_id": "d1"}),
            Chunk(id="c2", text="Second chunk", metadata={"doc_id": "d1"}),
            Chunk(id="c3", text="Third chunk", metadata={"doc_id": "d2"}),
        ]
        
        records = encoder.encode(chunks)
        
        assert len(records) == len(chunks)
        assert all(isinstance(r, ChunkRecord) for r in records)
    
    def test_encode_vector_dimensions_consistent(self):
        """Test all generated vectors have consistent dimensions."""
        fake_embedding = FakeEmbedding(dimensions=16)
        encoder = DenseEncoder(settings=Mock(), embedding_client=fake_embedding)
        
        chunks = [
            Chunk(id="c1", text="Chunk one"),
            Chunk(id="c2", text="Chunk two"),
            Chunk(id="c3", text="Chunk three"),
        ]
        
        records = encoder.encode(chunks)
        
        # All vectors should have the same dimension
        dimensions = [len(r.dense_vector) for r in records]
        assert all(d == 16 for d in dimensions)
        assert len(set(dimensions)) == 1  # All same dimension
    
    def test_encode_preserves_chunk_metadata(self):
        """Test encoder preserves chunk metadata in records."""
        fake_embedding = FakeEmbedding(dimensions=4)
        encoder = DenseEncoder(settings=Mock(), embedding_client=fake_embedding)
        
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
    
    def test_encode_batch_processing(self):
        """Test batch processing with FakeEmbedding batch_size configuration."""
        fake_embedding = FakeEmbedding(dimensions=4, batch_size=2)
        encoder = DenseEncoder(settings=Mock(), embedding_client=fake_embedding)
        
        # Create more chunks than batch size
        chunks = [
            Chunk(id=f"c{i}", text=f"Chunk {i}")
            for i in range(5)
        ]
        
        records = encoder.encode(chunks)
        
        assert len(records) == 5
        # Verify all records have vectors
        assert all(r.dense_vector is not None for r in records)
        assert all(len(r.dense_vector) == 4 for r in records)
    
    def test_encode_empty_input_returns_empty_list(self):
        """Test encoder returns empty list for empty input."""
        fake_embedding = FakeEmbedding()
        encoder = DenseEncoder(settings=Mock(), embedding_client=fake_embedding)
        
        records = encoder.encode([])
        
        assert records == []
    
    def test_encode_batch_empty_input_returns_empty_list(self):
        """Test encode_batch returns empty list for empty input."""
        fake_embedding = FakeEmbedding()
        encoder = DenseEncoder(settings=Mock(), embedding_client=fake_embedding)
        
        records = encoder.encode_batch([])
        
        assert records == []
    
    def test_encode_records_trace(self):
        """Test trace recording during encoding."""
        fake_embedding = FakeEmbedding(dimensions=8, batch_size=10)
        encoder = DenseEncoder(settings=Mock(), embedding_client=fake_embedding)
        
        chunks = [
            Chunk(id="c1", text="Chunk 1"),
            Chunk(id="c2", text="Chunk 2"),
        ]
        
        trace = TraceContext(trace_type="test")
        records = encoder.encode(chunks, trace=trace)
        
        # Check trace was recorded
        assert len(trace.stages) == 1
        stage = trace.stages[0]
        assert stage["name"] == "dense_encoding"
        assert stage["method"] == "batch_encode"
        assert stage["provider"] == "FakeEmbedding"
        assert stage["details"]["chunk_count"] == 2
        assert stage["details"]["model"] == "fake-embedding"
        assert stage["details"]["dimensions"] == 8
    
    def test_encode_batch_records_trace(self):
        """Test trace recording during batch encoding."""
        fake_embedding = FakeEmbedding(dimensions=8)
        encoder = DenseEncoder(settings=Mock(), embedding_client=fake_embedding)
        
        chunks = [Chunk(id="c1", text="Chunk 1")]
        
        trace = TraceContext(trace_type="test")
        records = encoder.encode_batch(chunks, trace=trace)
        
        # Check trace was recorded
        assert len(trace.stages) == 1
        assert trace.stages[0]["name"] == "dense_encoding"
    
    def test_encode_no_trace_does_not_fail(self):
        """Test encoding works without trace context."""
        fake_embedding = FakeEmbedding(dimensions=4)
        encoder = DenseEncoder(settings=Mock(), embedding_client=fake_embedding)
        
        chunks = [Chunk(id="c1", text="Test")]
        
        # Should not raise
        records = encoder.encode(chunks, trace=None)
        
        assert len(records) == 1
        assert records[0].dense_vector is not None
    
    def test_sparse_vector_is_none(self):
        """Test DenseEncoder sets sparse_vector to None."""
        fake_embedding = FakeEmbedding(dimensions=4)
        encoder = DenseEncoder(settings=Mock(), embedding_client=fake_embedding)
        
        chunks = [Chunk(id="c1", text="Test")]
        
        records = encoder.encode(chunks)
        
        assert len(records) == 1
        assert records[0].sparse_vector is None
    
    def test_output_order_matches_input(self):
        """Test output records maintain same order as input chunks."""
        fake_embedding = FakeEmbedding(dimensions=4)
        encoder = DenseEncoder(settings=Mock(), embedding_client=fake_embedding)
        
        chunks = [
            Chunk(id="first", text="First"),
            Chunk(id="second", text="Second"),
            Chunk(id="third", text="Third"),
        ]
        
        records = encoder.encode(chunks)
        
        assert [r.id for r in records] == ["first", "second", "third"]
        assert [r.text for r in records] == ["First", "Second", "Third"]


class TestDenseEncoderWithFactory:
    """Test DenseEncoder with EmbeddingFactory integration."""
    
    def test_encoder_uses_provided_embedding_client(self):
        """Test encoder uses provided embedding client instead of creating one."""
        fake_embedding = FakeEmbedding(dimensions=16)
        encoder = DenseEncoder(settings=Mock(), embedding_client=fake_embedding)
        
        assert encoder.embedding_client is fake_embedding
        assert encoder.embedding_client.dimensions == 16
    
    def test_encoder_creates_embedding_via_factory(self):
        """Test encoder creates embedding client via factory when not provided."""
        from unittest.mock import patch, MagicMock
        
        mock_embedding = FakeEmbedding(dimensions=32)
        
        with patch('ingestion.embedding.dense_encoder.EmbeddingFactory') as mock_factory:
            mock_factory.create.return_value = mock_embedding
            
            mock_settings = Mock()
            encoder = DenseEncoder(settings=mock_settings)
            
            mock_factory.create.assert_called_once_with(mock_settings)
            assert encoder.embedding_client is mock_embedding


class TestDenseEncoderEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_chunk_encoding(self):
        """Test encoding a single chunk."""
        fake_embedding = FakeEmbedding(dimensions=8)
        encoder = DenseEncoder(settings=Mock(), embedding_client=fake_embedding)
        
        chunks = [Chunk(id="single", text="Only one chunk")]
        
        records = encoder.encode(chunks)
        
        assert len(records) == 1
        assert records[0].id == "single"
        assert len(records[0].dense_vector) == 8
    
    def test_chunk_with_empty_metadata(self):
        """Test encoding chunk with empty metadata."""
        fake_embedding = FakeEmbedding(dimensions=4)
        encoder = DenseEncoder(settings=Mock(), embedding_client=fake_embedding)
        
        chunks = [Chunk(id="c1", text="Test")]  # Empty metadata by default
        
        records = encoder.encode(chunks)
        
        assert records[0].metadata is not None
        assert "start_offset" in records[0].metadata
    
    def test_large_batch_processing(self):
        """Test encoding large number of chunks."""
        fake_embedding = FakeEmbedding(dimensions=4, batch_size=10)
        encoder = DenseEncoder(settings=Mock(), embedding_client=fake_embedding)
        
        chunks = [Chunk(id=f"c{i}", text=f"Text {i}") for i in range(100)]
        
        records = encoder.encode(chunks)
        
        assert len(records) == 100
        assert all(len(r.dense_vector) == 4 for r in records)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
