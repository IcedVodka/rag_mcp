#!/usr/bin/env python3
"""
Document Chunker Unit Tests

Tests for DocumentChunker class covering:
- Chunk ID generation (uniqueness, determinism)
- Metadata inheritance
- Image reference distribution
- Source reference linking
- Configuration-driven behavior
"""

import sys
from pathlib import Path
from unittest.mock import Mock

# Add src to path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pytest

from core.settings import Settings, SplitterSettings
from core.types import Document, Chunk, DocumentMetadata, ImageInfo
from ingestion.chunking import DocumentChunker
from libs.splitter.base_splitter import BaseSplitter


class FakeSplitter(BaseSplitter):
    """
    Fake splitter for isolated unit testing.
    
    Splits text by character count with overlap.
    """
    
    def split_text(self, text, trace=None):
        """Split text into chunks of fixed size."""
        if not text:
            return []
        
        chunks = []
        start = 0
        size = self.chunk_size
        overlap = self.chunk_overlap
        
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - overlap
        
        return chunks


class TestDocumentChunker:
    """Test DocumentChunker functionality."""
    
    @pytest.fixture
    def mock_settings_small(self):
        """Settings with small chunk size for testing."""
        settings = Mock(spec=Settings)
        settings.splitter = Mock(spec=SplitterSettings)
        settings.splitter.strategy = "recursive"
        settings.splitter.recursive = {"chunk_size": 50, "chunk_overlap": 10}
        return settings
    
    @pytest.fixture
    def mock_settings_large(self):
        """Settings with large chunk size for testing."""
        settings = Mock(spec=Settings)
        settings.splitter = Mock(spec=SplitterSettings)
        settings.splitter.strategy = "recursive"
        settings.splitter.recursive = {"chunk_size": 1000, "chunk_overlap": 200}
        return settings
    
    @pytest.fixture
    def sample_document(self):
        """Create a sample document for testing."""
        return Document(
            id="doc_test_001",
            text="This is the first paragraph. It contains some text for testing. "
                 "Here is the second paragraph with more content to split.",
            metadata=DocumentMetadata(
                source_path="/path/to/test.txt",
                extra={"doc_type": "text", "title": "Test Document"}
            )
        )
    
    @pytest.fixture
    def document_with_images(self):
        """Create a document with image placeholders."""
        return Document(
            id="doc_images_001",
            text="First section with [IMAGE: img_001] placeholder. "
                 "Some more text here. "
                 "Second section with [IMAGE: img_002] another placeholder.",
            metadata=DocumentMetadata(
                source_path="/path/to/doc.pdf",
                extra={"doc_type": "pdf"},
                images=[
                    ImageInfo(id="img_001", path="data/images/img_001.png", page=1),
                    ImageInfo(id="img_002", path="data/images/img_002.png", page=1),
                    ImageInfo(id="img_003", path="data/images/img_003.png", page=2),
                ]
            )
        )
    
    def test_split_document_returns_chunks(self, mock_settings_small, sample_document):
        """Test that split_document returns a list of Chunk objects."""
        chunker = DocumentChunker(mock_settings_small)
        chunker._splitter = FakeSplitter(mock_settings_small.splitter.recursive)
        
        chunks = chunker.split_document(sample_document)
        
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)
        assert len(chunks) > 0
    
    def test_split_empty_document_returns_empty_list(self, mock_settings_small):
        """Test that empty document returns empty list."""
        empty_doc = Document(
            id="doc_empty",
            text="",
            metadata=DocumentMetadata(source_path="/empty.txt")
        )
        chunker = DocumentChunker(mock_settings_small)
        chunker._splitter = FakeSplitter(mock_settings_small.splitter.recursive)
        
        chunks = chunker.split_document(empty_doc)
        
        assert chunks == []
    
    def test_chunk_id_format(self, mock_settings_small, sample_document):
        """Test that chunk IDs follow the correct format."""
        chunker = DocumentChunker(mock_settings_small)
        chunker._splitter = FakeSplitter(mock_settings_small.splitter.recursive)
        
        chunks = chunker.split_document(sample_document)
        
        for i, chunk in enumerate(chunks):
            expected_prefix = f"{sample_document.id}_{i:04d}_"
            assert chunk.id.startswith(expected_prefix)
            # Check hash part (8 hex characters)
            hash_part = chunk.id.split("_")[-1]
            assert len(hash_part) == 8
            assert all(c in "0123456789abcdef" for c in hash_part)
    
    def test_chunk_id_uniqueness(self, mock_settings_small, sample_document):
        """Test that all chunk IDs are unique within a document."""
        chunker = DocumentChunker(mock_settings_small)
        chunker._splitter = FakeSplitter(mock_settings_small.splitter.recursive)
        
        chunks = chunker.split_document(sample_document)
        ids = [c.id for c in chunks]
        
        assert len(ids) == len(set(ids))
    
    def test_chunk_id_determinism(self, mock_settings_small, sample_document):
        """Test that same document produces same chunk IDs on re-split."""
        chunker = DocumentChunker(mock_settings_small)
        chunker._splitter = FakeSplitter(mock_settings_small.splitter.recursive)
        
        chunks1 = chunker.split_document(sample_document)
        chunks2 = chunker.split_document(sample_document)
        
        ids1 = [c.id for c in chunks1]
        ids2 = [c.id for c in chunks2]
        
        assert ids1 == ids2
    
    def test_chunk_id_changes_with_content(self, mock_settings_small):
        """Test that chunk ID changes when content changes."""
        chunker = DocumentChunker(mock_settings_small)
        chunker._splitter = FakeSplitter(mock_settings_small.splitter.recursive)
        
        # Use simple text where we can control the first chunk content
        original_doc = Document(
            id="doc_id_test",
            text="First chunk content here. Second part of the document.",
            metadata=DocumentMetadata(source_path="/test.txt")
        )
        
        chunks1 = chunker.split_document(original_doc)
        
        # Modify document text at the beginning to ensure first chunk changes
        modified_doc = Document(
            id=original_doc.id,
            text="MODIFIED first chunk content here. Second part of the document.",
            metadata=original_doc.metadata
        )
        
        chunks2 = chunker.split_document(modified_doc)
        
        # First chunk ID should be different due to content change
        assert chunks1[0].id != chunks2[0].id
    
    def test_metadata_inheritance(self, mock_settings_small, sample_document):
        """Test that chunk metadata inherits from document."""
        chunker = DocumentChunker(mock_settings_small)
        chunker._splitter = FakeSplitter(mock_settings_small.splitter.recursive)
        
        chunks = chunker.split_document(sample_document)
        
        for chunk in chunks:
            assert chunk.metadata["source_path"] == sample_document.metadata.source_path
            assert chunk.metadata["doc_type"] == "text"
            assert chunk.metadata["title"] == "Test Document"
    
    def test_chunk_index_added(self, mock_settings_small, sample_document):
        """Test that chunk_index is added to metadata."""
        chunker = DocumentChunker(mock_settings_small)
        chunker._splitter = FakeSplitter(mock_settings_small.splitter.recursive)
        
        chunks = chunker.split_document(sample_document)
        
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
    
    def test_source_ref_points_to_document(self, mock_settings_small, sample_document):
        """Test that source_ref correctly points to parent document."""
        chunker = DocumentChunker(mock_settings_small)
        chunker._splitter = FakeSplitter(mock_settings_small.splitter.recursive)
        
        chunks = chunker.split_document(sample_document)
        
        for chunk in chunks:
            assert chunk.source_ref == sample_document.id
    
    def test_chunk_offsets_are_valid(self, mock_settings_small, sample_document):
        """Test that chunk offsets are valid and ordered."""
        chunker = DocumentChunker(mock_settings_small)
        chunker._splitter = FakeSplitter(mock_settings_small.splitter.recursive)
        
        chunks = chunker.split_document(sample_document)
        
        for chunk in chunks:
            assert chunk.start_offset >= 0
            assert chunk.end_offset > chunk.start_offset
            assert chunk.end_offset <= len(sample_document.text)
    
    def test_image_reference_extraction(self, mock_settings_small):
        """Test extraction of image references from chunk text."""
        chunker = DocumentChunker(mock_settings_small)
        chunker._splitter = FakeSplitter(mock_settings_small.splitter.recursive)
        
        text = "Text with [IMAGE: img_001] and [IMAGE: img_002] placeholders."
        refs = chunker._extract_image_references(text)
        
        assert refs == ["img_001", "img_002"]
    
    def test_image_reference_no_whitespace_variations(self, mock_settings_small):
        """Test image reference extraction handles whitespace variations."""
        chunker = DocumentChunker(mock_settings_small)
        chunker._splitter = FakeSplitter(mock_settings_small.splitter.recursive)
        
        # Various spacing patterns
        text1 = "[IMAGE:img_001] no space"
        text2 = "[IMAGE: img_002] single space"
        text3 = "[IMAGE:  img_003] double space"
        
        assert chunker._extract_image_references(text1) == ["img_001"]
        assert chunker._extract_image_references(text2) == ["img_002"]
        assert chunker._extract_image_references(text3) == ["img_003"]
    
    def test_chunk_without_images_has_no_images_field(self, mock_settings_small):
        """Test that chunks without image placeholders don't have images field."""
        doc = Document(
            id="doc_no_images",
            text="Just plain text without any images.",
            metadata=DocumentMetadata(
                source_path="/test.txt",
                images=[ImageInfo(id="img_001", path="/img.png")]
            )
        )
        chunker = DocumentChunker(mock_settings_small)
        chunker._splitter = FakeSplitter(mock_settings_small.splitter.recursive)
        
        chunks = chunker.split_document(doc)
        
        for chunk in chunks:
            assert "images" not in chunk.metadata
            assert "image_refs" not in chunk.metadata
    
    def test_chunk_with_images_has_images_subset(self, mock_settings_small, document_with_images):
        """Test that chunks only get images they reference."""
        chunker = DocumentChunker(mock_settings_small)
        # Use small chunk size to force splitting between images
        chunker._splitter = FakeSplitter({"chunk_size": 60, "chunk_overlap": 5})
        
        chunks = chunker.split_document(document_with_images)
        
        # Find chunk with first image
        chunk_with_img1 = None
        chunk_with_img2 = None
        
        for chunk in chunks:
            if "img_001" in chunk.text:
                chunk_with_img1 = chunk
            if "img_002" in chunk.text:
                chunk_with_img2 = chunk
        
        # Verify first image chunk
        assert chunk_with_img1 is not None
        assert "images" in chunk_with_img1.metadata
        assert "img_001" in chunk_with_img1.metadata["image_refs"]
        # Should only have img_001, not img_002 or img_003
        img_ids = [img["id"] for img in chunk_with_img1.metadata["images"]]
        assert "img_001" in img_ids
        assert "img_002" not in img_ids
        assert "img_003" not in img_ids
        
        # Verify second image chunk
        assert chunk_with_img2 is not None
        assert "img_002" in chunk_with_img2.metadata["image_refs"]
    
    def test_image_metadata_inheritance(self, mock_settings_small, document_with_images):
        """Test that image metadata is correctly inherited into chunk."""
        chunker = DocumentChunker(mock_settings_small)
        chunker._splitter = FakeSplitter({"chunk_size": 100, "chunk_overlap": 10})
        
        chunks = chunker.split_document(document_with_images)
        
        # Find chunk with image
        for chunk in chunks:
            if "images" in chunk.metadata:
                img_data = chunk.metadata["images"][0]
                assert "id" in img_data
                assert "path" in img_data
                # Verify it's a dict (serialized ImageInfo)
                assert isinstance(img_data, dict)
                break
    
    def test_configuration_driven_chunk_size(self, mock_settings_small, mock_settings_large):
        """Test that different chunk sizes produce different number of chunks."""
        doc = Document(
            id="doc_config_test",
            text="Word " * 100,  # 500 characters
            metadata=DocumentMetadata(source_path="/test.txt")
        )
        
        # Small chunk size
        chunker_small = DocumentChunker(mock_settings_small)
        chunker_small._splitter = FakeSplitter(mock_settings_small.splitter.recursive)
        chunks_small = chunker_small.split_document(doc)
        
        # Large chunk size
        chunker_large = DocumentChunker(mock_settings_large)
        chunker_large._splitter = FakeSplitter(mock_settings_large.splitter.recursive)
        chunks_large = chunker_large.split_document(doc)
        
        # Small chunks should produce more chunks
        assert len(chunks_small) > len(chunks_large)
    
    def test_chunk_text_content_matches(self, mock_settings_small, sample_document):
        """Test that chunk text matches expected content."""
        chunker = DocumentChunker(mock_settings_small)
        chunker._splitter = FakeSplitter(mock_settings_small.splitter.recursive)
        
        chunks = chunker.split_document(sample_document)
        
        for chunk in chunks:
            # Each chunk's text should be a substring of the original
            assert chunk.text in sample_document.text
    
    def test_splitter_lazy_initialization(self, mock_settings_small):
        """Test that splitter is created lazily on first use."""
        chunker = DocumentChunker(mock_settings_small)
        
        # Initially splitter should be None
        assert chunker._splitter is None
        
        # Accessing splitter property should create it
        splitter = chunker.splitter
        assert splitter is not None
        assert chunker._splitter is not None
    
    def test_document_without_metadata_extra(self, mock_settings_small):
        """Test handling of document without extra metadata."""
        doc = Document(
            id="doc_minimal",
            text="Simple text content here.",
            metadata=DocumentMetadata(source_path="/simple.txt")
        )
        chunker = DocumentChunker(mock_settings_small)
        chunker._splitter = FakeSplitter(mock_settings_small.splitter.recursive)
        
        chunks = chunker.split_document(doc)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata["source_path"] == "/simple.txt"
            assert "chunk_index" in chunk.metadata
    
    def test_multiple_image_references_same_chunk(self, mock_settings_small):
        """Test chunk with multiple image references."""
        doc = Document(
            id="doc_multi_images",
            text="Text [IMAGE: img_001] middle [IMAGE: img_002] end.",
            metadata=DocumentMetadata(
                source_path="/test.txt",
                images=[
                    ImageInfo(id="img_001", path="/img1.png"),
                    ImageInfo(id="img_002", path="/img2.png"),
                ]
            )
        )
        chunker = DocumentChunker(mock_settings_small)
        chunker._splitter = FakeSplitter({"chunk_size": 200, "chunk_overlap": 20})
        
        chunks = chunker.split_document(doc)
        
        # Find the chunk with images
        for chunk in chunks:
            if "images" in chunk.metadata:
                assert len(chunk.metadata["image_refs"]) == 2
                assert "img_001" in chunk.metadata["image_refs"]
                assert "img_002" in chunk.metadata["image_refs"]
                return
        
        pytest.fail("Expected to find a chunk with images")
    
    def test_image_not_in_document_metadata(self, mock_settings_small):
        """Test handling of image reference not found in document metadata."""
        doc = Document(
            id="doc_missing_img",
            text="Text with [IMAGE: missing_img] reference.",
            metadata=DocumentMetadata(
                source_path="/test.txt",
                images=[ImageInfo(id="existing_img", path="/exists.png")]
            )
        )
        chunker = DocumentChunker(mock_settings_small)
        chunker._splitter = FakeSplitter({"chunk_size": 200, "chunk_overlap": 20})
        
        chunks = chunker.split_document(doc)
        
        # Chunk should have image_refs but empty images list
        for chunk in chunks:
            if "image_refs" in chunk.metadata:
                # The reference exists but image data doesn't
                assert chunk.metadata["image_refs"] == ["missing_img"]
                # Images list should be empty since it's not in document metadata
                assert chunk.metadata.get("images", []) == []


class TestDocumentChunkerIntegration:
    """Integration-style tests with real splitter."""
    
    def test_with_recursive_splitter(self):
        """Test integration with actual RecursiveSplitter."""
        from core.settings import SplitterSettings, Settings
        from libs.splitter.recursive_splitter import RecursiveSplitter
        
        settings = Mock(spec=Settings)
        settings.splitter = Mock(spec=SplitterSettings)
        settings.splitter.strategy = "recursive"
        settings.splitter.recursive = {"chunk_size": 100, "chunk_overlap": 20}
        
        doc = Document(
            id="doc_integration",
            text="First paragraph with some content. "
                 "Second paragraph with different content. "
                 "Third paragraph to ensure splitting.",
            metadata=DocumentMetadata(
                source_path="/test.txt",
                extra={"author": "Test Author"}
            )
        )
        
        chunker = DocumentChunker(settings)
        chunker._splitter = RecursiveSplitter(settings.splitter.recursive)
        
        chunks = chunker.split_document(doc)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.source_ref == doc.id
            assert chunk.metadata["author"] == "Test Author"
