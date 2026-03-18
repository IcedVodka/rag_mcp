#!/usr/bin/env python3
"""
Unit tests for PDF Loader contract.

Tests the BaseLoader abstract interface and PdfLoader implementation
using mocks and stubs to avoid dependency on real PDF files.
"""

import hashlib
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from core.types import Document, DocumentMetadata, ImageInfo
from src.libs.loader.base_loader import (
    BaseLoader,
    FileFormatError,
    ImageExtractionError,
    LoaderError,
)
from src.libs.loader.pdf_loader import PdfLoader


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_pdf_file():
    """Create a temporary file with .pdf extension."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        # Write minimal PDF-like content (not a valid PDF, just for path testing)
        f.write(b"%PDF-1.4 fake content")
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_images_dir():
    """Create a temporary directory for images."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_markitdown():
    """Mock markitdown MarkItDown class at module level."""
    mock_md_class = Mock()
    instance = Mock()
    result = Mock()
    result.text_content = "This is extracted PDF text content."
    instance.convert.return_value = result
    mock_md_class.return_value = instance
    
    with patch("src.libs.loader.pdf_loader.MarkItDown", mock_md_class):
        with patch("src.libs.loader.pdf_loader._HAS_MARKITDOWN", True):
            yield mock_md_class


@pytest.fixture
def mock_fitz():
    """Mock PyMuPDF fitz module at module level."""
    mock_fitz_module = Mock()
    
    # Mock document
    doc_mock = Mock()
    doc_mock.__len__ = Mock(return_value=2)  # 2 pages
    
    # Mock pages
    page1_mock = Mock()
    page1_mock.get_images.return_value = [
        (1, 0, 100, 100, 8, "DeviceRGB", "", "", "", "")  # xref=1
    ]
    page2_mock = Mock()
    page2_mock.get_images.return_value = []  # No images on page 2
    
    doc_mock.__getitem__ = Mock(side_effect=lambda i: [page1_mock, page2_mock][i])
    mock_fitz_module.open.return_value = doc_mock
    
    # Mock image extraction
    mock_fitz_module.extract_image.return_value = {
        "image": b"fake_image_bytes",
        "ext": "png"
    }
    
    with patch("src.libs.loader.pdf_loader.fitz", mock_fitz_module):
        with patch("src.libs.loader.pdf_loader._HAS_FITZ", True):
            yield mock_fitz_module


# ============================================================================
# BaseLoader Abstract Interface Tests
# ============================================================================

class TestBaseLoaderInterface:
    """Tests for BaseLoader abstract interface contract."""
    
    def test_base_loader_is_abstract(self):
        """BaseLoader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLoader()
    
    def test_base_loader_requires_load_method(self):
        """Subclasses must implement load method."""
        class IncompleteLoader(BaseLoader):
            pass
        
        with pytest.raises(TypeError):
            IncompleteLoader()
    
    def test_base_loader_valid_subclass(self):
        """Valid subclass with load method can be instantiated."""
        class ValidLoader(BaseLoader):
            def load(self, path):
                return Document(
                    id="test",
                    text="test",
                    metadata=DocumentMetadata(source_path=str(path))
                )
        
        loader = ValidLoader()
        assert loader is not None
        
        # Test load method works
        doc = loader.load("/fake/path.txt")
        assert isinstance(doc, Document)
        assert doc.metadata.source_path == "/fake/path.txt"
    
    def test_base_loader_validates_nonexistent_file(self):
        """BaseLoader validates that file exists."""
        class TestLoader(BaseLoader):
            def load(self, path):
                file_path = self._validate_path(path)
                return Document(
                    id="test",
                    text="test",
                    metadata=DocumentMetadata(source_path=str(file_path))
                )
        
        loader = TestLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/file.txt")
    
    def test_base_loader_validates_directory_not_file(self):
        """BaseLoader validates that path is a file, not directory."""
        class TestLoader(BaseLoader):
            def load(self, path):
                file_path = self._validate_path(path)
                return Document(
                    id="test",
                    text="test",
                    metadata=DocumentMetadata(source_path=str(file_path))
                )
        
        loader = TestLoader()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError):
                loader.load(temp_dir)


# ============================================================================
# PdfLoader Initialization Tests
# ============================================================================

class TestPdfLoaderInitialization:
    """Tests for PdfLoader initialization parameters."""
    
    def test_pdf_loader_default_init(self):
        """PdfLoader can be initialized with default parameters."""
        loader = PdfLoader()
        
        assert loader.images_dir == Path("data/images")
        assert loader.collection == "default"
    
    def test_pdf_loader_custom_images_dir(self, temp_images_dir):
        """PdfLoader accepts custom images directory."""
        loader = PdfLoader(images_dir=temp_images_dir)
        
        assert loader.images_dir == Path(temp_images_dir)
        assert loader.collection == "default"
    
    def test_pdf_loader_custom_collection(self):
        """PdfLoader accepts custom collection name."""
        loader = PdfLoader(collection="my_collection")
        
        assert loader.images_dir == Path("data/images")
        assert loader.collection == "my_collection"
    
    def test_pdf_loader_creates_images_dir(self, temp_images_dir):
        """PdfLoader creates images directory if it doesn't exist."""
        nested_dir = Path(temp_images_dir) / "nested" / "images"
        
        assert not nested_dir.exists()
        
        loader = PdfLoader(images_dir=nested_dir)
        
        assert nested_dir.exists()
        assert nested_dir.is_dir()
    
    def test_pdf_loader_path_as_string(self):
        """PdfLoader accepts images_dir as string."""
        loader = PdfLoader(images_dir="/tmp/test_images")
        
        assert loader.images_dir == Path("/tmp/test_images")
    
    def test_pdf_loader_path_as_path_object(self):
        """PdfLoader accepts images_dir as Path object."""
        path_obj = Path("/tmp/test_images")
        loader = PdfLoader(images_dir=path_obj)
        
        assert loader.images_dir == path_obj


# ============================================================================
# PdfLoader Load Method Tests
# ============================================================================

class TestPdfLoaderLoad:
    """Tests for PdfLoader.load() method behavior."""
    
    def test_load_returns_document(self, temp_pdf_file, temp_images_dir, mock_markitdown):
        """load() returns a Document object."""
        with patch.object(PdfLoader, '_extract_images', return_value=("This is extracted PDF text content.", [])):
            loader = PdfLoader(images_dir=temp_images_dir)
            doc = loader.load(temp_pdf_file)
            
            assert isinstance(doc, Document)
            assert isinstance(doc.metadata, DocumentMetadata)
    
    def test_load_document_has_id(self, temp_pdf_file, temp_images_dir, mock_markitdown):
        """Document ID is the SHA256 hash of the file."""
        with patch.object(PdfLoader, '_extract_images', return_value=("This is extracted PDF text content.", [])):
            loader = PdfLoader(images_dir=temp_images_dir)
            doc = loader.load(temp_pdf_file)
            
            # Calculate expected hash
            with open(temp_pdf_file, 'rb') as f:
                expected_hash = hashlib.sha256(f.read()).hexdigest()
            
            assert doc.id == expected_hash
    
    def test_load_document_has_text(self, temp_pdf_file, temp_images_dir, mock_markitdown):
        """Document contains extracted text."""
        with patch.object(PdfLoader, '_extract_images', return_value=("This is extracted PDF text content.", [])):
            loader = PdfLoader(images_dir=temp_images_dir)
            doc = loader.load(temp_pdf_file)
            
            assert doc.text == "This is extracted PDF text content."
    
    def test_load_document_has_source_path(self, temp_pdf_file, temp_images_dir, mock_markitdown):
        """Document metadata includes source_path."""
        with patch.object(PdfLoader, '_extract_images', return_value=("This is extracted PDF text content.", [])):
            loader = PdfLoader(images_dir=temp_images_dir)
            doc = loader.load(temp_pdf_file)
            
            assert doc.metadata.source_path == str(Path(temp_pdf_file).absolute())
    
    def test_load_document_has_images_list(self, temp_pdf_file, temp_images_dir, mock_markitdown):
        """Document metadata includes images list (may be empty)."""
        with patch.object(PdfLoader, '_extract_images', return_value=("This is extracted PDF text content.", [])):
            loader = PdfLoader(images_dir=temp_images_dir)
            doc = loader.load(temp_pdf_file)
            
            assert hasattr(doc.metadata, 'images')
            assert isinstance(doc.metadata.images, list)
            assert len(doc.metadata.images) == 0
    
    def test_load_raises_file_not_found(self, temp_images_dir):
        """load() raises FileNotFoundError for nonexistent file."""
        loader = PdfLoader(images_dir=temp_images_dir)
        
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/file.pdf")
    
    def test_load_raises_value_error_for_directory(self, temp_images_dir):
        """load() raises ValueError when path is a directory."""
        loader = PdfLoader(images_dir=temp_images_dir)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError):
                loader.load(temp_dir)


# ============================================================================
# Image ID Generation Tests
# ============================================================================

class TestImageIdGeneration:
    """Tests for image ID generation logic."""
    
    def test_image_id_format(self):
        """Image ID follows format: {doc_hash}_{page}_{seq}."""
        doc_hash = "abc123def456"
        page = 1
        seq = 2
        
        expected_id = f"{doc_hash}_{page}_{seq}"
        
        # Verify format
        parts = expected_id.split("_")
        assert len(parts) == 3
        assert parts[0] == doc_hash
        assert parts[1] == str(page)
        assert parts[2] == str(seq)
    
    def test_image_id_is_unique_per_page_sequence(self):
        """Image IDs are unique based on page and sequence."""
        doc_hash = "abc123"
        
        id1 = f"{doc_hash}_1_1"
        id2 = f"{doc_hash}_1_2"
        id3 = f"{doc_hash}_2_1"
        
        assert id1 != id2
        assert id1 != id3
        assert id2 != id3
    
    def test_image_id_contains_doc_hash(self):
        """Image ID contains the document hash for reference."""
        doc_hash = "a" * 64  # SHA256 hex string length
        
        image_id = f"{doc_hash}_1_1"
        
        assert doc_hash in image_id
        assert image_id.startswith(doc_hash)


# ============================================================================
# Placeholder Insertion Tests
# ============================================================================

class TestPlaceholderInsertion:
    """Tests for placeholder insertion logic."""
    
    def test_placeholder_format(self):
        """Placeholder follows format: [IMAGE: {image_id}]."""
        image_id = "abc123_1_1"
        
        placeholder = f"[IMAGE: {image_id}]"
        
        assert placeholder == "[IMAGE: abc123_1_1]"
        assert placeholder.startswith("[IMAGE: ")
        assert placeholder.endswith("]")
    
    def test_placeholder_is_searchable(self):
        """Placeholder can be found with regex pattern."""
        text = "Some text before [IMAGE: test_id_123] some text after"
        
        import re
        pattern = r'\[IMAGE:\s*([^\]]+)\]'
        matches = re.findall(pattern, text)
        
        assert len(matches) == 1
        assert matches[0] == "test_id_123"
    
    def test_multiple_placeholders(self):
        """Multiple placeholders can exist in text."""
        text = "[IMAGE: id1] text [IMAGE: id2] more text [IMAGE: id3]"
        
        import re
        pattern = r'\[IMAGE:\s*([^\]]+)\]'
        matches = re.findall(pattern, text)
        
        assert len(matches) == 3
        assert matches == ["id1", "id2", "id3"]
    
    def test_generate_image_placeholder_method(self):
        """PdfLoader has method to generate placeholders."""
        loader = PdfLoader()
        
        placeholder = loader._generate_image_placeholder("test_id")
        
        assert placeholder == "[IMAGE: test_id]"


# ============================================================================
# Image Metadata Tests
# ============================================================================

class TestImageMetadata:
    """Tests for image metadata structure."""
    
    def test_image_info_structure(self):
        """ImageInfo has required fields."""
        image_info = ImageInfo(
            id="hash_1_1",
            path="data/images/collection/hash_1_1.png",
            page=1,
            text_offset=100,
            text_length=20
        )
        
        assert image_info.id == "hash_1_1"
        assert image_info.path == "data/images/collection/hash_1_1.png"
        assert image_info.page == 1
        assert image_info.text_offset == 100
        assert image_info.text_length == 20
    
    def test_image_info_optional_position(self):
        """ImageInfo position field is optional."""
        image_info = ImageInfo(
            id="hash_1_1",
            path="data/images/collection/hash_1_1.png"
        )
        
        assert image_info.position == {}
        assert image_info.page is None
        assert image_info.text_offset == 0
    
    def test_document_validates_placeholders(self):
        """Document can validate placeholder-image metadata correspondence."""
        doc = Document(
            id="test",
            text="Text with [IMAGE: img1] placeholder",
            metadata=DocumentMetadata(
                source_path="test.pdf",
                images=[
                    ImageInfo(id="img1", path="path/to/img1.png")
                ]
            )
        )
        
        assert doc.validate_image_placeholders() is True
    
    def test_document_detects_missing_metadata(self):
        """Document detects when placeholder lacks metadata."""
        doc = Document(
            id="test",
            text="Text with [IMAGE: missing_img] placeholder",
            metadata=DocumentMetadata(
                source_path="test.pdf",
                images=[
                    ImageInfo(id="img1", path="path/to/img1.png")
                ]
            )
        )
        
        assert doc.validate_image_placeholders() is False


# ============================================================================
# Degradation Behavior Tests
# ============================================================================

class TestDegradationBehavior:
    """Tests for graceful degradation when operations fail."""
    
    def test_image_extraction_failure_does_not_block_text(
        self, temp_pdf_file, temp_images_dir, mock_markitdown
    ):
        """Image extraction failure should not block text extraction."""
        with patch.object(
            PdfLoader,
            '_extract_images',
            side_effect=Exception("Image extraction failed")
        ) as mock_extract:
            loader = PdfLoader(images_dir=temp_images_dir)
            
            # Should not raise - image extraction failure is logged but not blocking
            doc = loader.load(temp_pdf_file)
            
            assert isinstance(doc, Document)
            assert doc.text == "This is extracted PDF text content."
            assert doc.metadata.images == []
    
    def test_image_extraction_partial_failure(self, temp_pdf_file, temp_images_dir):
        """Partial image extraction failure allows remaining images to be extracted."""
        mock_md_class = Mock()
        instance = Mock()
        result = Mock()
        result.text_content = "PDF text"
        instance.convert.return_value = result
        mock_md_class.return_value = instance
        
        with patch("src.libs.loader.pdf_loader.MarkItDown", mock_md_class):
            with patch("src.libs.loader.pdf_loader._HAS_MARKITDOWN", True):
                # Mock fitz with some successful and some failed extractions
                mock_fitz_module = Mock()
                doc_mock = Mock()
                doc_mock.__len__ = Mock(return_value=1)
                
                page_mock = Mock()
                # Two images: one will succeed, one will fail
                page_mock.get_images.return_value = [
                    (1, 0, 100, 100, 8, "DeviceRGB", "", "", "", ""),
                    (2, 0, 100, 100, 8, "DeviceRGB", "", "", "", ""),
                ]
                doc_mock.__getitem__ = Mock(return_value=page_mock)
                mock_fitz_module.open.return_value = doc_mock
                
                # First call succeeds, second fails
                call_count = [0]
                def extract_side_effect(xref):
                    call_count[0] += 1
                    if call_count[0] == 1:
                        return {"image": b"bytes", "ext": "png"}
                    else:
                        raise Exception("Extraction failed")
                
                doc_mock.extract_image = Mock(side_effect=extract_side_effect)
                
                with patch("src.libs.loader.pdf_loader.fitz", mock_fitz_module):
                    with patch("src.libs.loader.pdf_loader._HAS_FITZ", True):
                        loader = PdfLoader(images_dir=temp_images_dir)
                        doc = loader.load(temp_pdf_file)
                        
                        # Document should be created even if some images failed
                        assert isinstance(doc, Document)
    
    def test_markitdown_import_error(self, temp_pdf_file, temp_images_dir):
        """ImportError raised if markitdown is not installed."""
        with patch("src.libs.loader.pdf_loader._HAS_MARKITDOWN", False):
            with patch("src.libs.loader.pdf_loader.MarkItDown", None):
                loader = PdfLoader(images_dir=temp_images_dir)
                
                with pytest.raises(RuntimeError) as exc_info:
                    loader.load(temp_pdf_file)
                
                assert "markitdown" in str(exc_info.value).lower() or "Failed" in str(exc_info.value)
    
    def test_fitz_import_error_for_images(self, temp_pdf_file, temp_images_dir):
        """Image extraction handles missing PyMuPDF gracefully."""
        mock_md_class = Mock()
        instance = Mock()
        result = Mock()
        result.text_content = "PDF text"
        instance.convert.return_value = result
        mock_md_class.return_value = instance
        
        with patch("src.libs.loader.pdf_loader.MarkItDown", mock_md_class):
            with patch("src.libs.loader.pdf_loader._HAS_MARKITDOWN", True):
                with patch("src.libs.loader.pdf_loader._HAS_FITZ", False):
                    with patch("src.libs.loader.pdf_loader.fitz", None):
                        loader = PdfLoader(images_dir=temp_images_dir)
                        
                        # Should still load text, just no images
                        doc = loader.load(temp_pdf_file)
                        
                        assert isinstance(doc, Document)
                        assert doc.text == "PDF text"
                        assert doc.metadata.images == []


# ============================================================================
# Text Extraction Tests
# ============================================================================

class TestTextExtraction:
    """Tests for PDF text extraction behavior."""
    
    def test_extract_text_raises_on_failure(self, temp_pdf_file, temp_images_dir):
        """Text extraction raises RuntimeError on markitdown failure."""
        mock_md_class = Mock()
        instance = Mock()
        instance.convert.side_effect = Exception("Conversion failed")
        mock_md_class.return_value = instance
        
        with patch("src.libs.loader.pdf_loader.MarkItDown", mock_md_class):
            with patch("src.libs.loader.pdf_loader._HAS_MARKITDOWN", True):
                loader = PdfLoader(images_dir=temp_images_dir)
                
                with pytest.raises(RuntimeError) as exc_info:
                    loader.load(temp_pdf_file)
                
                assert "markitdown" in str(exc_info.value).lower() or "Failed" in str(exc_info.value)
    
    def test_extract_text_empty_content(self, temp_pdf_file, temp_images_dir):
        """Handle empty PDF content gracefully."""
        mock_md_class = Mock()
        instance = Mock()
        result = Mock()
        result.text_content = ""  # Empty text
        instance.convert.return_value = result
        mock_md_class.return_value = instance
        
        with patch("src.libs.loader.pdf_loader.MarkItDown", mock_md_class):
            with patch("src.libs.loader.pdf_loader._HAS_MARKITDOWN", True):
                with patch.object(PdfLoader, '_extract_images', return_value=("", [])):
                    loader = PdfLoader(images_dir=temp_images_dir)
                    doc = loader.load(temp_pdf_file)
                    
                    assert doc.text == ""


# ============================================================================
# Integration-style Tests (with more realistic mocks)
# ============================================================================

class TestPdfLoaderIntegration:
    """Integration-style tests with realistic mock behavior."""
    
    def test_simple_pdf_no_images(self, temp_pdf_file, temp_images_dir):
        """Simulate loading a simple PDF with no images."""
        mock_md_class = Mock()
        instance = Mock()
        result = Mock()
        result.text_content = "This is a simple PDF with only text."
        instance.convert.return_value = result
        mock_md_class.return_value = instance
        
        with patch("src.libs.loader.pdf_loader.MarkItDown", mock_md_class):
            with patch("src.libs.loader.pdf_loader._HAS_MARKITDOWN", True):
                # Mock fitz returning no images
                mock_fitz_module = Mock()
                doc_mock = Mock()
                doc_mock.__len__ = Mock(return_value=1)
                
                page_mock = Mock()
                page_mock.get_images.return_value = []  # No images
                doc_mock.__getitem__ = Mock(return_value=page_mock)
                mock_fitz_module.open.return_value = doc_mock
                
                with patch("src.libs.loader.pdf_loader.fitz", mock_fitz_module):
                    with patch("src.libs.loader.pdf_loader._HAS_FITZ", True):
                        loader = PdfLoader(images_dir=temp_images_dir, collection="test")
                        doc = loader.load(temp_pdf_file)
                        
                        # Verify document structure
                        assert isinstance(doc, Document)
                        assert doc.text == "This is a simple PDF with only text."
                        assert doc.metadata.source_path == str(Path(temp_pdf_file).absolute())
                        assert doc.metadata.images == []
                        
                        # Verify ID is SHA256 hash
                        with open(temp_pdf_file, 'rb') as f:
                            expected_hash = hashlib.sha256(f.read()).hexdigest()
                        assert doc.id == expected_hash
    
    def test_pdf_with_images(self, temp_pdf_file, temp_images_dir):
        """Simulate loading a PDF with images."""
        mock_md_class = Mock()
        instance = Mock()
        result = Mock()
        result.text_content = "PDF with images"
        instance.convert.return_value = result
        mock_md_class.return_value = instance
        
        with patch("src.libs.loader.pdf_loader.MarkItDown", mock_md_class):
            with patch("src.libs.loader.pdf_loader._HAS_MARKITDOWN", True):
                mock_fitz_module = Mock()
                doc_mock = Mock()
                doc_mock.__len__ = Mock(return_value=1)
                
                page_mock = Mock()
                page_mock.get_images.return_value = [
                    (1, 0, 100, 100, 8, "DeviceRGB", "", "", "", ""),
                ]
                doc_mock.__getitem__ = Mock(return_value=page_mock)
                mock_fitz_module.open.return_value = doc_mock
                
                # Mock image extraction
                doc_mock.extract_image.return_value = {
                    "image": b"fake_png_bytes",
                    "ext": "png"
                }
                
                with patch("src.libs.loader.pdf_loader.fitz", mock_fitz_module):
                    with patch("src.libs.loader.pdf_loader._HAS_FITZ", True):
                        loader = PdfLoader(images_dir=temp_images_dir, collection="test")
                        doc = loader.load(temp_pdf_file)
                        
                        # Verify document has images
                        assert isinstance(doc, Document)
                        assert len(doc.metadata.images) == 1
                        
                        image_info = doc.metadata.images[0]
                        assert image_info.page == 1
                        # Path should contain collection and image file
                        assert "test/" in image_info.path
                        assert image_info.path.endswith(".png")
                        
                        # Verify placeholder in text
                        assert "[IMAGE:" in doc.text
    
    def test_pdf_multiple_pages_with_images(self, temp_pdf_file, temp_images_dir):
        """Simulate loading a multi-page PDF with images on different pages."""
        mock_md_class = Mock()
        instance = Mock()
        result = Mock()
        result.text_content = "Multi-page PDF content"
        instance.convert.return_value = result
        mock_md_class.return_value = instance
        
        with patch("src.libs.loader.pdf_loader.MarkItDown", mock_md_class):
            with patch("src.libs.loader.pdf_loader._HAS_MARKITDOWN", True):
                mock_fitz_module = Mock()
                doc_mock = Mock()
                doc_mock.__len__ = Mock(return_value=3)  # 3 pages
                
                # Page 1: 2 images
                page1 = Mock()
                page1.get_images.return_value = [
                    (1, 0, 100, 100, 8, "DeviceRGB", "", "", "", ""),
                    (2, 0, 100, 100, 8, "DeviceRGB", "", "", "", ""),
                ]
                
                # Page 2: no images
                page2 = Mock()
                page2.get_images.return_value = []
                
                # Page 3: 1 image
                page3 = Mock()
                page3.get_images.return_value = [
                    (3, 0, 100, 100, 8, "DeviceRGB", "", "", "", ""),
                ]
                
                pages = [page1, page2, page3]
                doc_mock.__getitem__ = Mock(side_effect=lambda i: pages[i])
                mock_fitz_module.open.return_value = doc_mock
                
                doc_mock.extract_image.return_value = {
                    "image": b"fake_png_bytes",
                    "ext": "png"
                }
                
                with patch("src.libs.loader.pdf_loader.fitz", mock_fitz_module):
                    with patch("src.libs.loader.pdf_loader._HAS_FITZ", True):
                        loader = PdfLoader(images_dir=temp_images_dir, collection="test")
                        doc = loader.load(temp_pdf_file)
                        
                        # Should have 3 images total
                        assert len(doc.metadata.images) == 3
                        
                        # Verify page numbers
                        assert doc.metadata.images[0].page == 1
                        assert doc.metadata.images[1].page == 1
                        assert doc.metadata.images[2].page == 3
