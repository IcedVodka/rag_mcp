#!/usr/bin/env python3
"""
Core Types Unit Tests

Tests for Document, Chunk, ChunkRecord, DocumentMetadata, and ImageInfo data types.
"""

import sys
from pathlib import Path
from datetime import datetime

import pytest

# Add src to path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.core.types import (
    Document,
    DocumentMetadata,
    Chunk,
    ChunkRecord,
    ImageInfo,
)


class TestImageInfo:
    """Test ImageInfo data class."""

    def test_image_info_creation(self):
        """Test ImageInfo creation with all fields."""
        image = ImageInfo(
            id="doc123_1_0",
            path="data/images/collection1/doc123_1_0.png",
            page=1,
            text_offset=100,
            text_length=20,
            position={"x": 0, "y": 0, "width": 100, "height": 100}
        )
        assert image.id == "doc123_1_0"
        assert image.path == "data/images/collection1/doc123_1_0.png"
        assert image.page == 1
        assert image.text_offset == 100
        assert image.text_length == 20
        assert image.position == {"x": 0, "y": 0, "width": 100, "height": 100}

    def test_image_info_defaults(self):
        """Test ImageInfo with default values."""
        image = ImageInfo(id="img1", path="data/images/img1.png")
        assert image.page is None
        assert image.text_offset == 0
        assert image.text_length == 0
        assert image.position == {}

    def test_image_info_serialization(self):
        """Test ImageInfo to_dict and from_dict."""
        original = ImageInfo(
            id="doc123_1_0",
            path="data/images/collection1/doc123_1_0.png",
            page=1,
            text_offset=100,
            text_length=20,
        )
        data = original.to_dict()
        restored = ImageInfo.from_dict(data)
        
        assert restored.id == original.id
        assert restored.path == original.path
        assert restored.page == original.page
        assert restored.text_offset == original.text_offset
        assert restored.text_length == original.text_length


class TestDocumentMetadata:
    """Test DocumentMetadata data class."""

    def test_metadata_creation(self):
        """Test DocumentMetadata creation."""
        metadata = DocumentMetadata(
            source_path="/path/to/document.pdf",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            updated_at=datetime(2024, 1, 2, 12, 0, 0),
        )
        assert metadata.source_path == "/path/to/document.pdf"
        assert metadata.created_at == datetime(2024, 1, 1, 12, 0, 0)
        assert metadata.updated_at == datetime(2024, 1, 2, 12, 0, 0)
        assert metadata.images == []
        assert metadata.extra == {}

    def test_metadata_with_images(self):
        """Test DocumentMetadata with images."""
        image = ImageInfo(id="img1", path="data/images/img1.png", text_offset=10, text_length=15)
        metadata = DocumentMetadata(
            source_path="/path/to/doc.pdf",
            images=[image]
        )
        assert len(metadata.images) == 1
        assert metadata.images[0].id == "img1"

    def test_metadata_serialization(self):
        """Test DocumentMetadata to_dict and from_dict."""
        image = ImageInfo(id="img1", path="data/images/img1.png")
        original = DocumentMetadata(
            source_path="/path/to/doc.pdf",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            images=[image],
            extra={"author": "Test Author", "title": "Test Title"}
        )
        
        data = original.to_dict()
        
        assert data["source_path"] == "/path/to/doc.pdf"
        assert data["created_at"] == "2024-01-01T12:00:00"
        assert len(data["images"]) == 1
        assert data["author"] == "Test Author"
        assert data["title"] == "Test Title"
        
        restored = DocumentMetadata.from_dict(data.copy())
        assert restored.source_path == original.source_path
        assert restored.created_at == original.created_at
        assert len(restored.images) == 1
        assert restored.extra["author"] == "Test Author"

    def test_metadata_source_path_required(self):
        """Test that source_path is a required field."""
        metadata = DocumentMetadata(source_path="")
        assert metadata.source_path == ""


class TestDocument:
    """Test Document data class."""

    def test_document_creation(self):
        """Test Document creation."""
        metadata = DocumentMetadata(source_path="/path/to/doc.pdf")
        doc = Document(
            id="doc123",
            text="This is the document content.",
            metadata=metadata
        )
        assert doc.id == "doc123"
        assert doc.text == "This is the document content."
        assert doc.metadata.source_path == "/path/to/doc.pdf"

    def test_document_serialization(self):
        """Test Document to_dict and from_dict."""
        metadata = DocumentMetadata(
            source_path="/path/to/doc.pdf",
            extra={"title": "Test Doc"}
        )
        original = Document(
            id="doc123",
            text="Document content",
            metadata=metadata
        )
        
        data = original.to_dict()
        
        assert data["id"] == "doc123"
        assert data["text"] == "Document content"
        assert data["metadata"]["source_path"] == "/path/to/doc.pdf"
        assert data["metadata"]["title"] == "Test Doc"
        
        restored = Document.from_dict(data)
        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.metadata.source_path == original.metadata.source_path

    def test_document_with_image_placeholder(self):
        """Test Document with image placeholder in text."""
        image_id = "doc123_1_0"
        metadata = DocumentMetadata(
            source_path="/path/to/doc.pdf",
            images=[ImageInfo(id=image_id, path=f"data/images/{image_id}.png", text_offset=20, text_length=18)]
        )
        doc = Document(
            id="doc123",
            text=f"This is text with [IMAGE: {image_id}] in the middle.",
            metadata=metadata
        )
        
        # Test placeholder extraction
        placeholders = doc.get_image_placeholders()
        assert len(placeholders) == 1
        assert placeholders[0] == image_id
        
        # Test validation
        assert doc.validate_image_placeholders() is True

    def test_document_multiple_image_placeholders(self):
        """Test Document with multiple image placeholders."""
        images = [
            ImageInfo(id="img1", path="data/images/img1.png", text_offset=10, text_length=13),
            ImageInfo(id="img2", path="data/images/img2.png", text_offset=35, text_length=13),
        ]
        metadata = DocumentMetadata(
            source_path="/path/to/doc.pdf",
            images=images
        )
        doc = Document(
            id="doc123",
            text="Start [IMAGE: img1] middle [IMAGE: img2] end",
            metadata=metadata
        )
        
        placeholders = doc.get_image_placeholders()
        assert len(placeholders) == 2
        assert "img1" in placeholders
        assert "img2" in placeholders
        assert doc.validate_image_placeholders() is True

    def test_document_invalid_placeholder(self):
        """Test Document with placeholder missing metadata."""
        metadata = DocumentMetadata(
            source_path="/path/to/doc.pdf",
            images=[]  # No images defined
        )
        doc = Document(
            id="doc123",
            text="This has [IMAGE: missing_img] placeholder",
            metadata=metadata
        )
        
        placeholders = doc.get_image_placeholders()
        assert len(placeholders) == 1
        assert placeholders[0] == "missing_img"
        assert doc.validate_image_placeholders() is False

    def test_document_placeholder_format_variations(self):
        """Test different valid placeholder formats."""
        metadata = DocumentMetadata(
            source_path="/path/to/doc.pdf",
            images=[
                ImageInfo(id="img1", path="data/images/img1.png"),
                ImageInfo(id="img-with-dashes", path="data/images/img-with-dashes.png"),
                ImageInfo(id="img_underscore_123", path="data/images/img_underscore_123.png"),
            ]
        )
        doc = Document(
            id="doc123",
            text="[IMAGE: img1] and [IMAGE: img-with-dashes] and [IMAGE: img_underscore_123]",
            metadata=metadata
        )
        
        placeholders = doc.get_image_placeholders()
        assert len(placeholders) == 3
        assert "img1" in placeholders
        assert "img-with-dashes" in placeholders
        assert "img_underscore_123" in placeholders

    def test_metadata_must_contain_source_path(self):
        """Test that metadata contains required source_path field."""
        # This test verifies the metadata convention
        metadata = DocumentMetadata(source_path="/path/to/doc.pdf")
        doc = Document(id="doc1", text="content", metadata=metadata)
        
        # source_path should be accessible
        assert doc.metadata.source_path == "/path/to/doc.pdf"
        
        # Should be present in serialization
        data = doc.to_dict()
        assert "source_path" in data["metadata"]
        assert data["metadata"]["source_path"] == "/path/to/doc.pdf"


class TestChunk:
    """Test Chunk data class."""

    def test_chunk_creation(self):
        """Test Chunk creation."""
        chunk = Chunk(
            id="chunk123",
            text="This is chunk content.",
            metadata={"doc_id": "doc1"},
            start_offset=0,
            end_offset=22,
            source_ref="doc1"
        )
        assert chunk.id == "chunk123"
        assert chunk.text == "This is chunk content."
        assert chunk.metadata == {"doc_id": "doc1"}
        assert chunk.start_offset == 0
        assert chunk.end_offset == 22
        assert chunk.source_ref == "doc1"

    def test_chunk_defaults(self):
        """Test Chunk with default values."""
        chunk = Chunk(id="chunk1", text="content")
        assert chunk.metadata == {}
        assert chunk.start_offset == 0
        assert chunk.end_offset == 0
        assert chunk.source_ref is None

    def test_chunk_serialization(self):
        """Test Chunk to_dict and from_dict."""
        original = Chunk(
            id="chunk123",
            text="Chunk content",
            metadata={"doc_id": "doc1", "index": 0},
            start_offset=10,
            end_offset=30,
            source_ref="doc1"
        )
        
        data = original.to_dict()
        
        assert data["id"] == "chunk123"
        assert data["text"] == "Chunk content"
        assert data["metadata"] == {"doc_id": "doc1", "index": 0}
        assert data["start_offset"] == 10
        assert data["end_offset"] == 30
        assert data["source_ref"] == "doc1"
        
        restored = Chunk.from_dict(data)
        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.metadata == original.metadata
        assert restored.start_offset == original.start_offset
        assert restored.end_offset == original.end_offset
        assert restored.source_ref == original.source_ref

    def test_chunk_char_count(self):
        """Test Chunk character count."""
        chunk = Chunk(id="c1", text="Hello World")
        assert chunk.get_char_count() == 11

    def test_chunk_overlap_same_source(self):
        """Test Chunk overlap detection with same source."""
        chunk1 = Chunk(id="c1", text="abc", start_offset=0, end_offset=5, source_ref="doc1")
        chunk2 = Chunk(id="c2", text="bcd", start_offset=3, end_offset=8, source_ref="doc1")
        chunk3 = Chunk(id="c3", text="efg", start_offset=10, end_offset=15, source_ref="doc1")
        
        assert chunk1.overlaps_with(chunk2) is True
        assert chunk2.overlaps_with(chunk1) is True
        assert chunk1.overlaps_with(chunk3) is False
        assert chunk3.overlaps_with(chunk1) is False

    def test_chunk_overlap_different_source(self):
        """Test Chunk overlap detection with different sources."""
        chunk1 = Chunk(id="c1", text="abc", start_offset=0, end_offset=5, source_ref="doc1")
        chunk2 = Chunk(id="c2", text="abc", start_offset=0, end_offset=5, source_ref="doc2")
        
        assert chunk1.overlaps_with(chunk2) is False


class TestChunkRecord:
    """Test ChunkRecord data class."""

    def test_chunk_record_creation(self):
        """Test ChunkRecord creation."""
        record = ChunkRecord(
            id="record123",
            text="Chunk text",
            metadata={"doc_id": "doc1", "chunk_id": "chunk1"},
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={"term1": 1.0, "term2": 0.5}
        )
        assert record.id == "record123"
        assert record.text == "Chunk text"
        assert record.metadata == {"doc_id": "doc1", "chunk_id": "chunk1"}
        assert record.dense_vector == [0.1, 0.2, 0.3]
        assert record.sparse_vector == {"term1": 1.0, "term2": 0.5}

    def test_chunk_record_defaults(self):
        """Test ChunkRecord with default values."""
        record = ChunkRecord(id="r1", text="content")
        assert record.metadata == {}
        assert record.dense_vector is None
        assert record.sparse_vector is None

    def test_chunk_record_serialization(self):
        """Test ChunkRecord to_dict and from_dict."""
        original = ChunkRecord(
            id="record123",
            text="Chunk text",
            metadata={"doc_id": "doc1"},
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={"term1": 1.0}
        )
        
        data = original.to_dict()
        
        assert data["id"] == "record123"
        assert data["text"] == "Chunk text"
        assert data["metadata"] == {"doc_id": "doc1"}
        assert data["dense_vector"] == [0.1, 0.2, 0.3]
        assert data["sparse_vector"] == {"term1": 1.0}
        
        restored = ChunkRecord.from_dict(data)
        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.metadata == original.metadata
        assert restored.dense_vector == original.dense_vector
        assert restored.sparse_vector == original.sparse_vector

    def test_chunk_record_serialization_no_vectors(self):
        """Test ChunkRecord serialization without vectors."""
        original = ChunkRecord(id="r1", text="content", metadata={"key": "value"})
        
        data = original.to_dict()
        
        assert "dense_vector" not in data
        assert "sparse_vector" not in data
        
        restored = ChunkRecord.from_dict(data)
        assert restored.dense_vector is None
        assert restored.sparse_vector is None

    def test_has_vectors(self):
        """Test has_vectors method."""
        record_no_vectors = ChunkRecord(id="r1", text="content")
        record_dense = ChunkRecord(id="r2", text="content", dense_vector=[0.1, 0.2])
        record_sparse = ChunkRecord(id="r3", text="content", sparse_vector={"t1": 1.0})
        record_both = ChunkRecord(
            id="r4", text="content",
            dense_vector=[0.1, 0.2],
            sparse_vector={"t1": 1.0}
        )
        
        assert record_no_vectors.has_vectors() is False
        assert record_dense.has_vectors() is True
        assert record_sparse.has_vectors() is True
        assert record_both.has_vectors() is True

    def test_has_dense_vector(self):
        """Test has_dense_vector method."""
        record = ChunkRecord(id="r1", text="content")
        assert record.has_dense_vector() is False
        
        record.dense_vector = [0.1, 0.2]
        assert record.has_dense_vector() is True

    def test_has_sparse_vector(self):
        """Test has_sparse_vector method."""
        record = ChunkRecord(id="r1", text="content")
        assert record.has_sparse_vector() is False
        
        record.sparse_vector = {"t1": 1.0}
        assert record.has_sparse_vector() is True

    def test_get_vector_dimensions(self):
        """Test get_vector_dimensions method."""
        record_no_vectors = ChunkRecord(id="r1", text="content")
        record_dense = ChunkRecord(id="r2", text="content", dense_vector=[0.1, 0.2, 0.3])
        record_sparse = ChunkRecord(id="r3", text="content", sparse_vector={"t1": 1.0, "t2": 0.5})
        record_both = ChunkRecord(
            id="r4", text="content",
            dense_vector=[0.1] * 128,
            sparse_vector={f"t{i}": 1.0 for i in range(50)}
        )
        
        assert record_no_vectors.get_vector_dimensions() == {}
        assert record_dense.get_vector_dimensions() == {"dense": 3}
        assert record_sparse.get_vector_dimensions() == {"sparse": 2}
        assert record_both.get_vector_dimensions() == {"dense": 128, "sparse": 50}


class TestTypeStability:
    """Test type stability and JSON serialization compatibility."""

    def test_document_fields_stable(self):
        """Test Document fields are stable and complete."""
        doc = Document(
            id="test",
            text="text",
            metadata=DocumentMetadata(source_path="/path")
        )
        data = doc.to_dict()
        
        # Required fields must be present
        assert "id" in data
        assert "text" in data
        assert "metadata" in data
        
        # Values must match
        assert data["id"] == "test"
        assert data["text"] == "text"
        assert data["metadata"]["source_path"] == "/path"

    def test_chunk_fields_stable(self):
        """Test Chunk fields are stable and complete."""
        chunk = Chunk(
            id="c1",
            text="text",
            metadata={"key": "value"},
            start_offset=10,
            end_offset=20,
            source_ref="doc1"
        )
        data = chunk.to_dict()
        
        # Required fields must be present
        assert "id" in data
        assert "text" in data
        assert "metadata" in data
        assert "start_offset" in data
        assert "end_offset" in data
        assert "source_ref" in data

    def test_chunk_record_fields_stable(self):
        """Test ChunkRecord fields are stable and complete."""
        record = ChunkRecord(
            id="r1",
            text="text",
            metadata={"key": "value"},
            dense_vector=[0.1, 0.2],
            sparse_vector={"t1": 1.0}
        )
        data = record.to_dict()
        
        # Required fields must be present
        assert "id" in data
        assert "text" in data
        assert "metadata" in data
        assert "dense_vector" in data
        assert "sparse_vector" in data

    def test_metadata_extensibility(self):
        """Test metadata allows incremental extension without breaking compatibility."""
        # Add custom fields to extra
        metadata = DocumentMetadata(
            source_path="/path/to/doc.pdf",
            extra={
                "custom_field1": "value1",
                "custom_field2": 123,
                "nested": {"key": "value"}
            }
        )
        
        data = metadata.to_dict()
        
        # Custom fields should be in output
        assert data["custom_field1"] == "value1"
        assert data["custom_field2"] == 123
        assert data["nested"] == {"key": "value"}
        
        # Should be recoverable
        restored = DocumentMetadata.from_dict(data.copy())
        assert restored.extra["custom_field1"] == "value1"
        assert restored.extra["custom_field2"] == 123


class TestImagePlaceholderFormat:
    """Test image placeholder format compliance."""

    def test_standard_placeholder_format(self):
        """Test standard [IMAGE: {image_id}] format."""
        image_id = "abc123_1_0"
        text = f"Text with [IMAGE: {image_id}] placeholder"
        
        doc = Document(
            id="doc1",
            text=text,
            metadata=DocumentMetadata(
                source_path="/path",
                images=[ImageInfo(id=image_id, path=f"data/images/{image_id}.png")]
            )
        )
        
        placeholders = doc.get_image_placeholders()
        assert len(placeholders) == 1
        assert placeholders[0] == image_id

    def test_placeholder_with_spaces(self):
        """Test placeholder with extra spaces."""
        text = "Text with [IMAGE:  img123  ] placeholder"
        
        doc = Document(
            id="doc1",
            text=text,
            metadata=DocumentMetadata(source_path="/path")
        )
        
        placeholders = doc.get_image_placeholders()
        assert len(placeholders) == 1
        assert placeholders[0] == "img123"

    def test_multiple_placeholders_same_line(self):
        """Test multiple placeholders on same line."""
        text = "[IMAGE: img1] text [IMAGE: img2] more text [IMAGE: img3]"
        
        doc = Document(
            id="doc1",
            text=text,
            metadata=DocumentMetadata(source_path="/path")
        )
        
        placeholders = doc.get_image_placeholders()
        assert len(placeholders) == 3
        assert placeholders == ["img1", "img2", "img3"]

    def test_no_false_positives(self):
        """Test that similar patterns are not matched."""
        text = "Text with [FIGURE: img1] and [IMAGE img2] and (IMAGE: img3)"
        
        doc = Document(
            id="doc1",
            text=text,
            metadata=DocumentMetadata(source_path="/path")
        )
        
        placeholders = doc.get_image_placeholders()
        assert len(placeholders) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
