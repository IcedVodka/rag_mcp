#!/usr/bin/env python3
"""
Batch Processor Unit Tests

Tests for BatchProcessor batch orchestration and encoding coordination.
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
from ingestion.embedding.batch_processor import BatchProcessor


class FakeDenseEncoder:
    """Fake dense encoder for testing."""
    
    def __init__(self, dimensions: int = 8):
        self.dimensions = dimensions
    
    def encode_batch(self, chunks, trace=None):
        """Return fake dense embeddings."""
        records = []
        for chunk in chunks:
            record = ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata={
                    **chunk.metadata,
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                    "source_ref": chunk.source_ref,
                },
                dense_vector=[0.1 + i * 0.01 for i in range(self.dimensions)],
                sparse_vector=None
            )
            records.append(record)
        return records


class FakeSparseEncoder:
    """Fake sparse encoder for testing."""
    
    def __init__(self, tokenizer: str = "simple"):
        self.tokenizer = tokenizer
    
    def encode_batch(self, chunks, trace=None):
        """Return fake sparse embeddings."""
        records = []
        for chunk in chunks:
            terms = chunk.text.lower().split()[:3] if chunk.text else []
            record = ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata={
                    **chunk.metadata,
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                    "source_ref": chunk.source_ref,
                },
                dense_vector=None,
                sparse_vector={
                    "terms": terms,
                    "tf": [1] * len(terms),
                    "doc_length": len(terms)
                } if terms else {"terms": [], "tf": [], "doc_length": 0}
            )
            records.append(record)
        return records


class TestBatchProcessorInit:
    """Test BatchProcessor initialization."""
    
    def test_default_initialization(self):
        """Test processor initializes with default settings."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = True
        
        fake_dense = FakeDenseEncoder()
        fake_sparse = FakeSparseEncoder()
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=fake_dense,
            sparse_encoder=fake_sparse
        )
        
        assert processor.batch_size == 32
        assert processor.encode_dense is True
        assert processor.encode_sparse is True
        assert processor.dense_encoder is fake_dense
        assert processor.sparse_encoder is fake_sparse
    
    def test_custom_batch_size(self):
        """Test processor with custom batch size."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 16
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = True
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=FakeDenseEncoder(),
            sparse_encoder=FakeSparseEncoder()
        )
        
        assert processor.batch_size == 16
    
    def test_encoder_injection(self):
        """Test that encoders can be injected."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = True
        
        fake_dense = FakeDenseEncoder(dimensions=16)
        fake_sparse = FakeSparseEncoder(tokenizer="jieba")
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=fake_dense,
            sparse_encoder=fake_sparse
        )
        
        assert processor.dense_encoder is fake_dense
        assert processor.dense_encoder.dimensions == 16
        assert processor.sparse_encoder is fake_sparse
        assert processor.sparse_encoder.tokenizer == "jieba"
    
    def test_encode_dense_disabled(self):
        """Test processor with dense encoding disabled."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = False
        mock_settings.ingestion.encode_sparse = True
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=None,
            sparse_encoder=FakeSparseEncoder()
        )
        
        assert processor.encode_dense is False
        assert processor.encode_sparse is True
        assert processor.dense_encoder is None
        assert processor.sparse_encoder is not None
    
    def test_encode_sparse_disabled(self):
        """Test processor with sparse encoding disabled."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = False
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=FakeDenseEncoder(),
            sparse_encoder=None
        )
        
        assert processor.encode_dense is True
        assert processor.encode_sparse is False
        assert processor.dense_encoder is not None
        assert processor.sparse_encoder is None
    
    def test_both_encoders_disabled(self):
        """Test processor with both encoders disabled."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = False
        mock_settings.ingestion.encode_sparse = False
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=None,
            sparse_encoder=None
        )
        
        assert processor.encode_dense is False
        assert processor.encode_sparse is False
        assert processor.dense_encoder is None
        assert processor.sparse_encoder is None


class TestBatchProcessorSplitBatches:
    """Test batch splitting functionality."""
    
    def test_split_batches_exact_multiple(self):
        """Test batch splitting when chunks count is exact multiple."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = True
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=FakeDenseEncoder(),
            sparse_encoder=FakeSparseEncoder()
        )
        
        chunks = [Chunk(id=f"c{i}", text=f"Text {i}") for i in range(6)]
        batches = processor._split_batches(chunks, batch_size=2)
        
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 2
    
    def test_split_batches_remainder(self):
        """Test batch splitting with remainder (5 chunks, batch_size=2 -> 3 batches)."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = True
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=FakeDenseEncoder(),
            sparse_encoder=FakeSparseEncoder()
        )
        
        chunks = [Chunk(id=f"c{i}", text=f"Text {i}") for i in range(5)]
        batches = processor._split_batches(chunks, batch_size=2)
        
        assert len(batches) == 3
        assert len(batches[0]) == 2  # c0, c1
        assert len(batches[1]) == 2  # c2, c3
        assert len(batches[2]) == 1  # c4
    
    def test_split_batches_order_stable(self):
        """Test that batch splitting maintains order."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = True
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=FakeDenseEncoder(),
            sparse_encoder=FakeSparseEncoder()
        )
        
        chunks = [
            Chunk(id="first", text="First"),
            Chunk(id="second", text="Second"),
            Chunk(id="third", text="Third"),
            Chunk(id="fourth", text="Fourth"),
            Chunk(id="fifth", text="Fifth"),
        ]
        batches = processor._split_batches(chunks, batch_size=2)
        
        # Check first batch order
        assert [c.id for c in batches[0]] == ["first", "second"]
        # Check second batch order
        assert [c.id for c in batches[1]] == ["third", "fourth"]
        # Check third batch order
        assert [c.id for c in batches[2]] == ["fifth"]
    
    def test_split_batches_empty_input(self):
        """Test batch splitting with empty input."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = True
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=FakeDenseEncoder(),
            sparse_encoder=FakeSparseEncoder()
        )
        
        batches = processor._split_batches([], batch_size=2)
        
        assert batches == []
    
    def test_split_batches_single_chunk(self):
        """Test batch splitting with single chunk."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = True
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=FakeDenseEncoder(),
            sparse_encoder=FakeSparseEncoder()
        )
        
        chunks = [Chunk(id="single", text="Only one")]
        batches = processor._split_batches(chunks, batch_size=2)
        
        assert len(batches) == 1
        assert len(batches[0]) == 1
        assert batches[0][0].id == "single"
    
    def test_split_batches_batch_size_larger_than_input(self):
        """Test batch splitting when batch_size > chunk count."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = True
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=FakeDenseEncoder(),
            sparse_encoder=FakeSparseEncoder()
        )
        
        chunks = [Chunk(id=f"c{i}", text=f"Text {i}") for i in range(3)]
        batches = processor._split_batches(chunks, batch_size=10)
        
        assert len(batches) == 1
        assert len(batches[0]) == 3


class TestBatchProcessorProcess:
    """Test main process functionality."""
    
    def test_process_returns_both_dense_and_sparse(self):
        """Test process returns both dense and sparse records."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = True
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=FakeDenseEncoder(dimensions=8),
            sparse_encoder=FakeSparseEncoder()
        )
        
        chunks = [
            Chunk(id="c1", text="First chunk"),
            Chunk(id="c2", text="Second chunk"),
        ]
        
        dense_records, sparse_records = processor.process(chunks)
        
        assert len(dense_records) == 2
        assert len(sparse_records) == 2
        
        # Check dense vectors
        assert all(r.dense_vector is not None for r in dense_records)
        assert all(len(r.dense_vector) == 8 for r in dense_records)
        
        # Check sparse vectors
        assert all(r.sparse_vector is not None for r in sparse_records)
    
    def test_process_order_aligned(self):
        """Test dense and sparse records are aligned by index."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = True
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=FakeDenseEncoder(),
            sparse_encoder=FakeSparseEncoder()
        )
        
        chunks = [
            Chunk(id="chunk_a", text="First content"),
            Chunk(id="chunk_b", text="Second content"),
            Chunk(id="chunk_c", text="Third content"),
        ]
        
        dense_records, sparse_records = processor.process(chunks)
        
        # Check alignment - same index should have same chunk id
        assert len(dense_records) == len(sparse_records)
        for i in range(len(dense_records)):
            assert dense_records[i].id == sparse_records[i].id
            assert dense_records[i].text == sparse_records[i].text
    
    def test_process_with_batches(self):
        """Test processing with multiple batches."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 2
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = True
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=FakeDenseEncoder(),
            sparse_encoder=FakeSparseEncoder()
        )
        
        chunks = [Chunk(id=f"c{i}", text=f"Text {i}") for i in range(5)]
        
        dense_records, sparse_records = processor.process(chunks)
        
        assert len(dense_records) == 5
        assert len(sparse_records) == 5
        # Order should be preserved across batches
        assert [r.id for r in dense_records] == ["c0", "c1", "c2", "c3", "c4"]
        assert [r.id for r in sparse_records] == ["c0", "c1", "c2", "c3", "c4"]
    
    def test_process_dense_only(self):
        """Test processing with only dense encoding enabled."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = False
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=FakeDenseEncoder(),
            sparse_encoder=None
        )
        
        chunks = [Chunk(id="c1", text="Test")]
        
        dense_records, sparse_records = processor.process(chunks)
        
        assert len(dense_records) == 1
        assert len(sparse_records) == 0
        assert dense_records[0].dense_vector is not None
    
    def test_process_sparse_only(self):
        """Test processing with only sparse encoding enabled."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = False
        mock_settings.ingestion.encode_sparse = True
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=None,
            sparse_encoder=FakeSparseEncoder()
        )
        
        chunks = [Chunk(id="c1", text="Test")]
        
        dense_records, sparse_records = processor.process(chunks)
        
        assert len(dense_records) == 0
        assert len(sparse_records) == 1
        assert sparse_records[0].sparse_vector is not None
    
    def test_process_both_disabled(self):
        """Test processing with both encoders disabled."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = False
        mock_settings.ingestion.encode_sparse = False
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=None,
            sparse_encoder=None
        )
        
        chunks = [Chunk(id="c1", text="Test")]
        
        dense_records, sparse_records = processor.process(chunks)
        
        assert len(dense_records) == 0
        assert len(sparse_records) == 0
    
    def test_process_empty_input(self):
        """Test processing with empty input."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = True
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=FakeDenseEncoder(),
            sparse_encoder=FakeSparseEncoder()
        )
        
        dense_records, sparse_records = processor.process([])
        
        assert dense_records == []
        assert sparse_records == []


class TestBatchProcessorTrace:
    """Test trace recording functionality."""
    
    def test_trace_records_dense_batch_timing(self):
        """Test trace records dense batch processing timing."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 2
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = False
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=FakeDenseEncoder(),
            sparse_encoder=None
        )
        
        chunks = [Chunk(id=f"c{i}", text=f"Text {i}") for i in range(3)]
        trace = TraceContext(trace_type="ingestion")
        
        processor.process(chunks, trace=trace)
        
        # Should have 2 batch stages (2 batches: [c0,c1], [c2])
        dense_stages = [s for s in trace.stages if s["name"] == "dense_batch_processing"]
        assert len(dense_stages) == 2
        
        # Check first batch details
        assert dense_stages[0]["details"]["batch_index"] == 0
        assert dense_stages[0]["details"]["batch_size"] == 2
        assert "elapsed_ms" in dense_stages[0]["details"]
        
        # Check second batch details
        assert dense_stages[1]["details"]["batch_index"] == 1
        assert dense_stages[1]["details"]["batch_size"] == 1
    
    def test_trace_records_sparse_batch_timing(self):
        """Test trace records sparse batch processing timing."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 2
        mock_settings.ingestion.encode_dense = False
        mock_settings.ingestion.encode_sparse = True
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=None,
            sparse_encoder=FakeSparseEncoder()
        )
        
        chunks = [Chunk(id=f"c{i}", text=f"Text {i}") for i in range(3)]
        trace = TraceContext(trace_type="ingestion")
        
        processor.process(chunks, trace=trace)
        
        # Should have 2 batch stages
        sparse_stages = [s for s in trace.stages if s["name"] == "sparse_batch_processing"]
        assert len(sparse_stages) == 2
        
        # Check batch details
        assert sparse_stages[0]["details"]["batch_index"] == 0
        assert sparse_stages[0]["details"]["batch_size"] == 2
        assert "elapsed_ms" in sparse_stages[0]["details"]
    
    def test_trace_records_both_encoders(self):
        """Test trace records both dense and sparse batches."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 2
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = True
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=FakeDenseEncoder(),
            sparse_encoder=FakeSparseEncoder()
        )
        
        chunks = [Chunk(id=f"c{i}", text=f"Text {i}") for i in range(3)]
        trace = TraceContext(trace_type="ingestion")
        
        processor.process(chunks, trace=trace)
        
        # Should have 2 dense + 2 sparse = 4 batch stages
        dense_stages = [s for s in trace.stages if s["name"] == "dense_batch_processing"]
        sparse_stages = [s for s in trace.stages if s["name"] == "sparse_batch_processing"]
        
        assert len(dense_stages) == 2
        assert len(sparse_stages) == 2
    
    def test_process_without_trace(self):
        """Test processing works without trace context."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = True
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=FakeDenseEncoder(),
            sparse_encoder=FakeSparseEncoder()
        )
        
        chunks = [Chunk(id="c1", text="Test")]
        
        # Should not raise
        dense_records, sparse_records = processor.process(chunks, trace=None)
        
        assert len(dense_records) == 1
        assert len(sparse_records) == 1


class TestBatchProcessorProcessBatchDense:
    """Test _process_batch_dense method."""
    
    def test_process_batch_dense_basic(self):
        """Test processing a single batch through dense encoder."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = False
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=FakeDenseEncoder(dimensions=4),
            sparse_encoder=None
        )
        
        batch = [
            Chunk(id="c1", text="First"),
            Chunk(id="c2", text="Second"),
        ]
        
        records = processor._process_batch_dense(batch)
        
        assert len(records) == 2
        assert all(r.dense_vector is not None for r in records)
        assert all(len(r.dense_vector) == 4 for r in records)
        assert records[0].id == "c1"
        assert records[1].id == "c2"
    
    def test_process_batch_dense_empty(self):
        """Test processing empty batch."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = False
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=FakeDenseEncoder(),
            sparse_encoder=None
        )
        
        records = processor._process_batch_dense([])
        
        assert records == []
    
    def test_process_batch_dense_without_encoder_raises(self):
        """Test that processing without encoder raises error."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = False  # Disabled so no encoder created
        mock_settings.ingestion.encode_sparse = False
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=None,  # No encoder
            sparse_encoder=None
        )
        
        batch = [Chunk(id="c1", text="Test")]
        
        with pytest.raises(RuntimeError, match="Dense encoder not initialized"):
            processor._process_batch_dense(batch)


class TestBatchProcessorProcessBatchSparse:
    """Test _process_batch_sparse method."""
    
    def test_process_batch_sparse_basic(self):
        """Test processing a single batch through sparse encoder."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = False
        mock_settings.ingestion.encode_sparse = True
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=None,
            sparse_encoder=FakeSparseEncoder()
        )
        
        batch = [
            Chunk(id="c1", text="hello world"),
            Chunk(id="c2", text="foo bar"),
        ]
        
        records = processor._process_batch_sparse(batch)
        
        assert len(records) == 2
        assert all(r.sparse_vector is not None for r in records)
        assert records[0].id == "c1"
        assert records[1].id == "c2"
    
    def test_process_batch_sparse_empty(self):
        """Test processing empty batch."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = False
        mock_settings.ingestion.encode_sparse = True
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=None,
            sparse_encoder=FakeSparseEncoder()
        )
        
        records = processor._process_batch_sparse([])
        
        assert records == []
    
    def test_process_batch_sparse_without_encoder_raises(self):
        """Test that processing without encoder raises error."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = False
        mock_settings.ingestion.encode_sparse = False  # Disabled so no encoder created
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=None,
            sparse_encoder=None  # No encoder
        )
        
        batch = [Chunk(id="c1", text="Test")]
        
        with pytest.raises(RuntimeError, match="Sparse encoder not initialized"):
            processor._process_batch_sparse(batch)


class TestBatchProcessorEdgeCases:
    """Test edge cases."""
    
    def test_single_chunk_both_encoders(self):
        """Test processing single chunk with both encoders."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = True
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=FakeDenseEncoder(dimensions=8),
            sparse_encoder=FakeSparseEncoder()
        )
        
        chunks = [Chunk(id="single", text="Only one chunk")]
        
        dense_records, sparse_records = processor.process(chunks)
        
        assert len(dense_records) == 1
        assert len(sparse_records) == 1
        assert dense_records[0].id == "single"
        assert sparse_records[0].id == "single"
        assert dense_records[0].dense_vector is not None
        assert sparse_records[0].sparse_vector is not None
    
    def test_large_batch_processing(self):
        """Test processing large number of chunks."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 10
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = True
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=FakeDenseEncoder(dimensions=4),
            sparse_encoder=FakeSparseEncoder()
        )
        
        chunks = [Chunk(id=f"c{i}", text=f"Text {i}") for i in range(100)]
        
        dense_records, sparse_records = processor.process(chunks)
        
        assert len(dense_records) == 100
        assert len(sparse_records) == 100
        # Verify order preserved
        assert [r.id for r in dense_records] == [f"c{i}" for i in range(100)]
    
    def test_preserves_chunk_metadata(self):
        """Test that processor preserves chunk metadata."""
        mock_settings = Mock()
        mock_settings.ingestion = Mock()
        mock_settings.ingestion.batch_size = 32
        mock_settings.ingestion.encode_dense = True
        mock_settings.ingestion.encode_sparse = True
        
        processor = BatchProcessor(
            settings=mock_settings,
            dense_encoder=FakeDenseEncoder(),
            sparse_encoder=FakeSparseEncoder()
        )
        
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
        
        dense_records, sparse_records = processor.process(chunks)
        
        # Check dense record metadata
        assert dense_records[0].metadata["custom_key"] == "custom_value"
        assert dense_records[0].metadata["page"] == 1
        assert dense_records[0].metadata["start_offset"] == 0
        assert dense_records[0].metadata["end_offset"] == 12
        assert dense_records[0].metadata["source_ref"] == "doc1"
        
        # Check sparse record metadata
        assert sparse_records[0].metadata["custom_key"] == "custom_value"
        assert sparse_records[0].metadata["page"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
