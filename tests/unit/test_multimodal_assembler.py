#!/usr/bin/env python3
"""
Unit tests for MultimodalAssembler.

Tests the multimodal response assembly functionality including:
- Image content generation from base64
- Text + Image content array assembly
- Graceful degradation when images are missing
- MIME type detection
"""

import base64
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from core.response.multimodal_assembler import MultimodalAssembler, assemble_multimodal_response
from core.response.citation_generator import Citation
from core.types import RetrievalResult


class TestMultimodalAssembler:
    """Test cases for MultimodalAssembler."""
    
    def test_init_with_default_storage(self):
        """Test that assembler initializes with default ImageStorage."""
        assembler = MultimodalAssembler()
        assert assembler.image_storage is not None
    
    def test_init_with_custom_storage(self):
        """Test that assembler initializes with custom ImageStorage."""
        mock_storage = Mock()
        assembler = MultimodalAssembler(image_storage=mock_storage)
        assert assembler.image_storage == mock_storage
    
    def test_extract_image_refs_empty_results(self):
        """Test extracting image refs from empty results."""
        assembler = MultimodalAssembler()
        image_ids = assembler._extract_image_refs([])
        assert image_ids == []
    
    def test_extract_image_refs_no_refs(self):
        """Test extracting image refs when no results have image_refs."""
        assembler = MultimodalAssembler()
        results = [
            RetrievalResult(
                chunk_id="chunk_001",
                score=0.9,
                text="No images here",
                metadata={"source_path": "doc.pdf"}
            )
        ]
        image_ids = assembler._extract_image_refs(results)
        assert image_ids == []
    
    def test_extract_image_refs_with_refs(self):
        """Test extracting image refs from results with image_refs."""
        assembler = MultimodalAssembler()
        results = [
            RetrievalResult(
                chunk_id="chunk_001",
                score=0.9,
                text="Has images",
                metadata={"image_refs": ["img_001", "img_002"]}
            )
        ]
        image_ids = assembler._extract_image_refs(results)
        assert len(image_ids) == 2
        assert "img_001" in image_ids
        assert "img_002" in image_ids
    
    def test_extract_image_refs_deduplicates(self):
        """Test that duplicate image refs are deduplicated."""
        assembler = MultimodalAssembler()
        results = [
            RetrievalResult(
                chunk_id="chunk_001",
                score=0.9,
                text="First chunk",
                metadata={"image_refs": ["img_001"]}
            ),
            RetrievalResult(
                chunk_id="chunk_002",
                score=0.8,
                text="Second chunk with same image",
                metadata={"image_refs": ["img_001"]}
            )
        ]
        image_ids = assembler._extract_image_refs(results)
        assert len(image_ids) == 1
        assert image_ids[0] == "img_001"
    
    def test_extract_image_refs_string_ref(self):
        """Test extracting single string image_ref (edge case)."""
        assembler = MultimodalAssembler()
        results = [
            RetrievalResult(
                chunk_id="chunk_001",
                score=0.9,
                text="Has image",
                metadata={"image_refs": "img_001"}  # String instead of list
            )
        ]
        image_ids = assembler._extract_image_refs(results)
        assert len(image_ids) == 1
        assert image_ids[0] == "img_001"
    
    def test_extract_image_refs_ignores_empty(self):
        """Test that empty/whitespace image refs are ignored."""
        assembler = MultimodalAssembler()
        results = [
            RetrievalResult(
                chunk_id="chunk_001",
                score=0.9,
                text="Has empty refs",
                metadata={"image_refs": ["", "  ", "img_001"]}
            )
        ]
        image_ids = assembler._extract_image_refs(results)
        assert len(image_ids) == 1
        assert image_ids[0] == "img_001"
    
    def test_guess_mime_type_png(self):
        """Test MIME type guessing for PNG."""
        assembler = MultimodalAssembler()
        assert assembler._guess_mime_type("image.png") == "image/png"
        assert assembler._guess_mime_type("image.PNG") == "image/png"
    
    def test_guess_mime_type_jpeg(self):
        """Test MIME type guessing for JPEG."""
        assembler = MultimodalAssembler()
        assert assembler._guess_mime_type("image.jpg") == "image/jpeg"
        assert assembler._guess_mime_type("image.jpeg") == "image/jpeg"
        assert assembler._guess_mime_type("image.JPG") == "image/jpeg"
    
    def test_guess_mime_type_other_formats(self):
        """Test MIME type guessing for other formats."""
        assembler = MultimodalAssembler()
        assert assembler._guess_mime_type("image.gif") == "image/gif"
        assert assembler._guess_mime_type("image.webp") == "image/webp"
        assert assembler._guess_mime_type("image.bmp") == "image/bmp"
        assert assembler._guess_mime_type("image.svg") == "image/svg+xml"
    
    def test_guess_mime_type_default(self):
        """Test MIME type defaults to PNG for unknown extensions."""
        assembler = MultimodalAssembler()
        assert assembler._guess_mime_type("image.unknown") == "image/png"
        assert assembler._guess_mime_type("image") == "image/png"
    
    def test_load_image_as_content_missing_record(self):
        """Test loading image when record doesn't exist."""
        mock_storage = Mock()
        mock_storage.get_image_record.return_value = None
        
        assembler = MultimodalAssembler(image_storage=mock_storage)
        result = assembler._load_image_as_content("non_existent_id")
        
        assert result is None
        mock_storage.get_image_record.assert_called_once_with("non_existent_id")
    
    def test_load_image_as_content_missing_file(self):
        """Test loading image when file doesn't exist."""
        mock_storage = Mock()
        mock_record = Mock()
        mock_record.file_path = "/non/existent/path.png"
        mock_storage.get_image_record.return_value = mock_record
        
        assembler = MultimodalAssembler(image_storage=mock_storage)
        result = assembler._load_image_as_content("img_id")
        
        assert result is None
    
    def test_load_image_as_content_success(self, tmp_path):
        """Test successful image loading and encoding."""
        # Create a test file
        test_file = tmp_path / "test_image.png"
        test_data = b"fake_image_data"
        test_file.write_bytes(test_data)
        
        mock_storage = Mock()
        mock_record = Mock()
        mock_record.file_path = str(test_file)
        mock_record.mime_type = "image/png"
        mock_storage.get_image_record.return_value = mock_record
        
        assembler = MultimodalAssembler(image_storage=mock_storage)
        result = assembler._load_image_as_content("img_id")
        
        assert result is not None
        assert result["type"] == "image"
        assert result["mimeType"] == "image/png"
        assert result["data"] == base64.b64encode(test_data).decode("utf-8")
    
    def test_load_image_as_content_uses_guessed_mime_type(self, tmp_path):
        """Test that guessed MIME type is used when record has none."""
        test_file = tmp_path / "test_image.jpg"
        test_file.write_bytes(b"fake_data")
        
        mock_storage = Mock()
        mock_record = Mock()
        mock_record.file_path = str(test_file)
        mock_record.mime_type = None  # No mime type in record
        mock_storage.get_image_record.return_value = mock_record
        
        assembler = MultimodalAssembler(image_storage=mock_storage)
        result = assembler._load_image_as_content("img_id")
        
        assert result is not None
        assert result["mimeType"] == "image/jpeg"  # Guessed from extension
    
    def test_assemble_response_text_only(self):
        """Test assembling response with text only."""
        assembler = MultimodalAssembler()
        
        citations = [Citation(
            id=1, source="doc.pdf", page=1, chunk_id="chunk_001",
            text="Test", score=0.9
        )]
        results = [
            RetrievalResult(
                chunk_id="chunk_001", score=0.9, text="Test",
                metadata={}
            )
        ]
        
        content = assembler.assemble_response("Hello world", citations, results)
        
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Hello world"
    
    def test_assemble_response_with_images(self, tmp_path):
        """Test assembling response with text and images."""
        # Create test image file
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"fake_image_data")
        
        mock_storage = Mock()
        mock_record = Mock()
        mock_record.file_path = str(test_file)
        mock_record.mime_type = "image/png"
        mock_storage.get_image_record.return_value = mock_record
        
        assembler = MultimodalAssembler(image_storage=mock_storage)
        
        citations = [Citation(
            id=1, source="doc.pdf", page=1, chunk_id="chunk_001",
            text="Test", score=0.9
        )]
        results = [
            RetrievalResult(
                chunk_id="chunk_001", score=0.9, text="Test",
                metadata={"image_refs": ["img_001"]}
            )
        ]
        
        content = assembler.assemble_response("Hello world", citations, results)
        
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image"
    
    def test_assemble_response_graceful_degradation(self):
        """Test graceful degradation when image loading fails."""
        mock_storage = Mock()
        mock_storage.get_image_record.return_value = None  # Image not found
        
        assembler = MultimodalAssembler(image_storage=mock_storage)
        
        citations = [Citation(
            id=1, source="doc.pdf", page=1, chunk_id="chunk_001",
            text="Test", score=0.9
        )]
        results = [
            RetrievalResult(
                chunk_id="chunk_001", score=0.9, text="Test",
                metadata={"image_refs": ["missing_img"]}
            )
        ]
        
        # Should not raise exception
        content = assembler.assemble_response("Hello world", citations, results)
        
        # Should have text but no image (since loading failed)
        assert len(content) == 1
        assert content[0]["type"] == "text"


class TestAssembleMultimodalResponse:
    """Test cases for the assemble_multimodal_response convenience function."""
    
    def test_convenience_function(self):
        """Test the convenience function works correctly."""
        citations = [Citation(
            id=1, source="doc.pdf", page=1, chunk_id="chunk_001",
            text="Test", score=0.9
        )]
        results = [
            RetrievalResult(
                chunk_id="chunk_001", score=0.9, text="Test",
                metadata={}
            )
        ]
        
        content = assemble_multimodal_response("Hello", citations, results)
        
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Hello"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
