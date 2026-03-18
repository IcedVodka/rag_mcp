#!/usr/bin/env python3
"""
Vector Upserter Idempotency Tests

Tests the idempotent behavior of VectorUpserter including chunk ID generation
and vector store operations.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest

from ingestion.storage.vector_upserter import VectorUpserter
from core.types import ChunkRecord


class TestChunkIdGeneration:
    """Test stable chunk ID generation."""
    
    def test_same_input_same_id(self):
        """Test that same input always produces same ID."""
        id1 = VectorUpserter.generate_chunk_id("doc.pdf", 0, "abc12345")
        id2 = VectorUpserter.generate_chunk_id("doc.pdf", 0, "abc12345")
        assert id1 == id2
    
    def test_different_input_different_id(self):
        """Test that different inputs produce different IDs."""
        id1 = VectorUpserter.generate_chunk_id("doc1.pdf", 0, "abc12345")
        id2 = VectorUpserter.generate_chunk_id("doc2.pdf", 0, "abc12345")
        id3 = VectorUpserter.generate_chunk_id("doc1.pdf", 1, "abc12345")
        id4 = VectorUpserter.generate_chunk_id("doc1.pdf", 0, "xyz98765")
        
        assert id1 != id2
        assert id1 != id3
        assert id1 != id4
    
    def test_id_length(self):
        """Test that generated ID is 16 characters."""
        chunk_id = VectorUpserter.generate_chunk_id("doc.pdf", 0, "abc12345")
        assert len(chunk_id) == 16
    
    def test_id_is_hex(self):
        """Test that generated ID is valid hexadecimal."""
        chunk_id = VectorUpserter.generate_chunk_id("doc.pdf", 0, "abc12345")
        int(chunk_id, 16)  # Should not raise


class TestVectorUpserterInit:
    """Test VectorUpserter initialization."""
    
    def test_init_with_mock_store(self):
        """Test initialization with mock vector store."""
        mock_store = Mock()
        upserter = VectorUpserter(Mock(), vector_store=mock_store)
        assert upserter.vector_store is mock_store


class TestVectorUpsert:
    """Test vector upsert operations."""
    
    def test_upsert_single_record(self):
        """Test upserting a single record."""
        mock_store = Mock()
        upserter = VectorUpserter(Mock(), vector_store=mock_store)
        
        record = ChunkRecord(
            id="",
            text="hello world",
            dense_vector=[0.1, 0.2, 0.3],
            metadata={"source_path": "doc.pdf", "chunk_index": 0}
        )
        
        result = upserter.upsert([record])
        
        assert result["upserted_count"] == 1
        assert len(result["chunk_ids"]) == 1
        assert mock_store.upsert.called
    
    def test_upsert_generates_id(self):
        """Test that upsert generates ID for records without one."""
        mock_store = Mock()
        upserter = VectorUpserter(Mock(), vector_store=mock_store)
        
        record = ChunkRecord(
            id="",
            text="hello world",
            dense_vector=[0.1, 0.2, 0.3],
            metadata={"source_path": "doc.pdf", "chunk_index": 0}
        )
        
        upserter.upsert([record])
        
        assert len(record.id) == 16  # Generated ID is 16 chars
    
    def test_upsert_preserves_existing_id(self):
        """Test that existing IDs are preserved."""
        mock_store = Mock()
        upserter = VectorUpserter(Mock(), vector_store=mock_store)
        
        record = ChunkRecord(
            id="existing-id-1234567890abcdef",
            text="hello world",
            dense_vector=[0.1, 0.2, 0.3],
            metadata={"source_path": "doc.pdf", "chunk_index": 0}
        )
        
        upserter.upsert([record])
        
        # ID should not change if already set and long enough
        assert record.id == "existing-id-1234567890abcdef"
    
    def test_upsert_multiple_records(self):
        """Test upserting multiple records."""
        mock_store = Mock()
        upserter = VectorUpserter(Mock(), vector_store=mock_store)
        
        records = [
            ChunkRecord(
                id="",
                text=f"chunk {i}",
                dense_vector=[0.1, 0.2, 0.3],
                metadata={"source_path": "doc.pdf", "chunk_index": i}
            )
            for i in range(5)
        ]
        
        result = upserter.upsert(records)
        
        assert result["upserted_count"] == 5
        assert len(result["chunk_ids"]) == 5
    
    def test_missing_dense_vector_raises(self):
        """Test that missing dense_vector raises ValueError."""
        mock_store = Mock()
        upserter = VectorUpserter(Mock(), vector_store=mock_store)
        
        record = ChunkRecord(
            id="c1",
            text="hello",
            dense_vector=None,  # type: ignore
            metadata={}
        )
        
        with pytest.raises(ValueError, match="missing dense_vector"):
            upserter.upsert([record])
    
    def test_upsert_empty_list(self):
        """Test upserting empty list."""
        mock_store = Mock()
        upserter = VectorUpserter(Mock(), vector_store=mock_store)
        
        result = upserter.upsert([])
        
        assert result["upserted_count"] == 0
        assert not mock_store.upsert.called
    
    def test_upsert_failure_raises(self):
        """Test that upsert failure raises RuntimeError."""
        mock_store = Mock()
        mock_store.upsert.side_effect = Exception("Store error")
        upserter = VectorUpserter(Mock(), vector_store=mock_store)
        
        record = ChunkRecord(
            id="c1",
            text="hello",
            dense_vector=[0.1],
            metadata={}
        )
        
        with pytest.raises(RuntimeError, match="Store error"):
            upserter.upsert([record])


class TestVectorUpsertBatch:
    """Test batch upsert operations."""
    
    def test_batch_upsert(self):
        """Test upserting in batches."""
        mock_store = Mock()
        upserter = VectorUpserter(Mock(), vector_store=mock_store)
        
        records = [
            ChunkRecord(
                id="",
                text=f"chunk {i}",
                dense_vector=[0.1, 0.2],
                metadata={"source_path": "doc.pdf", "chunk_index": i}
            )
            for i in range(10)
        ]
        
        result = upserter.upsert_batch(records, batch_size=3)
        
        assert result["upserted_count"] == 10
        # Should be called 4 times: 3 + 3 + 3 + 1
        assert mock_store.upsert.call_count == 4
    
    def test_batch_upsert_empty(self):
        """Test batch upsert with empty list."""
        mock_store = Mock()
        upserter = VectorUpserter(Mock(), vector_store=mock_store)
        
        result = upserter.upsert_batch([], batch_size=5)
        
        assert result["upserted_count"] == 0
        assert not mock_store.upsert.called


class TestIdempotency:
    """Test idempotent behavior."""
    
    def test_same_content_same_id(self):
        """Test that same content produces same ID across upserts."""
        mock_store = Mock()
        upserter = VectorUpserter(Mock(), vector_store=mock_store)
        
        record1 = ChunkRecord(
            id="",
            text="hello world",
            dense_vector=[0.1, 0.2],
            metadata={"source_path": "doc.pdf", "chunk_index": 0}
        )
        record2 = ChunkRecord(
            id="",
            text="hello world",
            dense_vector=[0.1, 0.2],
            metadata={"source_path": "doc.pdf", "chunk_index": 0}
        )
        
        upserter.upsert([record1])
        upserter.upsert([record2])
        
        assert record1.id == record2.id
    
    def test_different_content_different_id(self):
        """Test that different content produces different IDs."""
        mock_store = Mock()
        upserter = VectorUpserter(Mock(), vector_store=mock_store)
        
        record1 = ChunkRecord(
            id="",
            text="hello world",
            dense_vector=[0.1, 0.2],
            metadata={"source_path": "doc.pdf", "chunk_index": 0}
        )
        record2 = ChunkRecord(
            id="",
            text="different content",
            dense_vector=[0.3, 0.4],
            metadata={"source_path": "doc.pdf", "chunk_index": 1}
        )
        
        upserter.upsert([record1, record2])
        
        assert record1.id != record2.id


class TestCollection:
    """Test collection handling."""
    
    def test_upsert_with_collection(self):
        """Test upserting to a specific collection."""
        mock_store = Mock()
        upserter = VectorUpserter(Mock(), vector_store=mock_store)
        
        record = ChunkRecord(
            id="c1",
            text="hello",
            dense_vector=[0.1],
            metadata={}
        )
        
        result = upserter.upsert([record], collection="my_collection")
        
        assert result["collection"] == "my_collection"
        mock_store.upsert.assert_called_once()
        call_kwargs = mock_store.upsert.call_args[1]
        assert call_kwargs.get("collection") == "my_collection"
