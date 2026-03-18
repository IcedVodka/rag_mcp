#!/usr/bin/env python3
"""
Unit Tests for ImageCaptioner

Tests caption generation, fallback behavior, configuration handling,
Vision LLM integration (mocked), and error scenarios.

Total: 15 tests
- Enabled mode: 5 tests
- Fallback behavior: 5 tests  
- Configuration: 3 tests
- Error handling: 2 tests
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Adjust path for imports
import sys
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from core.types import Chunk
from core.settings import Settings
from core.trace.trace_context import TraceContext
from ingestion.transform.image_captioner import ImageCaptioner
from libs.llm.base_vision_llm import VisionResponse


# ==================== Fixtures ====================

@pytest.fixture
def mock_settings():
    """Create mock settings with image_captioner enabled."""
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.image_captioner = {
        'enabled': True,
        'prompt_path': 'config/prompts/image_captioning.txt'
    }
    settings.vision_llm = Mock()
    settings.vision_llm.enabled = True
    settings.vision_llm.provider = "openai"
    settings.vision_llm.openai = {"api_key": "test-key", "model": "gpt-4o"}
    settings.storage = Mock()
    settings.storage.image_dir = "data/images"
    return settings


@pytest.fixture
def mock_settings_disabled():
    """Create mock settings with image_captioner disabled."""
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.image_captioner = {
        'enabled': False,
        'prompt_path': 'config/prompts/image_captioning.txt'
    }
    return settings


@pytest.fixture
def mock_settings_no_vision_llm():
    """Create mock settings with Vision LLM not configured."""
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.image_captioner = {
        'enabled': True,
        'prompt_path': 'config/prompts/image_captioning.txt'
    }
    settings.vision_llm = None
    return settings


@pytest.fixture
def mock_vision_llm():
    """Create mock Vision LLM."""
    llm = Mock()
    llm.model = "gpt-4o"
    llm.__class__.__name__ = "MockVisionLLM"
    return llm


@pytest.fixture
def sample_chunks_with_images() -> List[Chunk]:
    """Create sample chunks with image references."""
    return [
        Chunk(
            id="chunk_1",
            text="This is a document with an image [IMAGE: img_001].",
            metadata={
                "source": "test",
                "image_refs": [
                    {"id": "img_001", "path": "/fake/path/img_001.png", "page": 1}
                ]
            },
            start_offset=0,
            end_offset=100,
            source_ref="doc_1"
        ),
        Chunk(
            id="chunk_2",
            text="Another chunk with two images [IMAGE: img_002] and [IMAGE: img_003].",
            metadata={
                "source": "test",
                "image_refs": [
                    {"id": "img_002", "path": "/fake/path/img_002.png", "page": 2},
                    {"id": "img_003", "path": "/fake/path/img_003.jpg", "page": 2}
                ]
            },
            start_offset=101,
            end_offset=200,
            source_ref="doc_1"
        )
    ]


@pytest.fixture
def sample_chunks_no_images() -> List[Chunk]:
    """Create sample chunks without image references."""
    return [
        Chunk(
            id="chunk_3",
            text="This is a plain text chunk without any images.",
            metadata={"source": "test"},
            start_offset=0,
            end_offset=50,
            source_ref="doc_2"
        )
    ]


@pytest.fixture
def trace_context():
    """Create a trace context for testing."""
    return TraceContext(trace_type="ingestion")


# ==================== Enabled Mode Tests (5) ====================

class TestEnabledMode:
    """Tests for enabled captioning mode with Vision LLM."""
    
    def test_generates_captions_for_images(self, mock_settings, mock_vision_llm, sample_chunks_with_images):
        """Test that captions are generated for images when Vision LLM is available."""
        # Setup mock response
        mock_response = VisionResponse(
            content="A diagram showing system architecture with three main components.",
            model="gpt-4o"
        )
        mock_vision_llm.chat_with_image.return_value = mock_response
        
        captioner = ImageCaptioner(mock_settings, vision_llm=mock_vision_llm)
        
        with patch.object(captioner, '_resolve_image_path', return_value='/fake/path/img_001.png'):
            with patch('pathlib.Path.exists', return_value=True):
                result = captioner.transform(sample_chunks_with_images)
        
        # Verify captions were generated
        assert 'image_captions' in result[0].metadata
        assert 'img_001' in result[0].metadata['image_captions']
        assert "diagram showing system architecture" in result[0].metadata['image_captions']['img_001']
    
    def test_vision_llm_called_with_correct_params(self, mock_settings, mock_vision_llm, sample_chunks_with_images):
        """Test that Vision LLM is called with correct parameters."""
        mock_response = VisionResponse(content="Test caption", model="gpt-4o")
        mock_vision_llm.chat_with_image.return_value = mock_response
        
        captioner = ImageCaptioner(mock_settings, vision_llm=mock_vision_llm)
        
        with patch.object(captioner, '_resolve_image_path', return_value='/fake/path/img_001.png'):
            with patch('pathlib.Path.exists', return_value=True):
                captioner.transform(sample_chunks_with_images)
        
        # Verify Vision LLM was called
        assert mock_vision_llm.chat_with_image.called
        call_args = mock_vision_llm.chat_with_image.call_args
        assert call_args.kwargs['image_path'] == '/fake/path/img_001.png'
        assert call_args.kwargs['text'] is not None  # Prompt template
    
    def test_multiple_images_per_chunk(self, mock_settings, mock_vision_llm, sample_chunks_with_images):
        """Test processing multiple images in a single chunk."""
        # Setup mock to return different captions based on image_id in trace
        def mock_chat_with_image(text, image_path=None, image_base64=None, trace=None):
            # Extract image_id from the trace if available
            if trace and hasattr(trace, '_current_image_id'):
                image_id = trace._current_image_id
            else:
                # Infer from call context - use image_path
                image_id = image_path
            
            if 'img_002' in str(image_id) or 'img_002' in str(image_path):
                return VisionResponse(content="Caption for image 2", model="gpt-4o")
            elif 'img_003' in str(image_id) or 'img_003' in str(image_path):
                return VisionResponse(content="Caption for image 3", model="gpt-4o")
            else:
                return VisionResponse(content="Caption for image 1", model="gpt-4o")
        
        mock_vision_llm.chat_with_image.side_effect = mock_chat_with_image
        
        captioner = ImageCaptioner(mock_settings, vision_llm=mock_vision_llm)
        
        # Use autospec to preserve function signature
        with patch.object(captioner, '_resolve_image_path', side_effect=lambda path, img_id: f'/fake/path/{img_id}.png'):
            with patch('pathlib.Path.exists', return_value=True):
                result = captioner.transform(sample_chunks_with_images)
        
        # Second chunk has two images
        assert 'image_captions' in result[1].metadata
        assert len(result[1].metadata['image_captions']) == 2
        assert result[1].metadata['image_captions']['img_002'] == "Caption for image 2"
        assert result[1].metadata['image_captions']['img_003'] == "Caption for image 3"
    
    def test_has_unprocessed_images_false_on_success(self, mock_settings, mock_vision_llm, sample_chunks_with_images):
        """Test that has_unprocessed_images is False when all images succeed."""
        mock_vision_llm.chat_with_image.return_value = VisionResponse(
            content="Test caption", model="gpt-4o"
        )
        
        captioner = ImageCaptioner(mock_settings, vision_llm=mock_vision_llm)
        
        with patch.object(captioner, '_resolve_image_path', return_value='/fake/path/img.png'):
            with patch('pathlib.Path.exists', return_value=True):
                result = captioner.transform(sample_chunks_with_images)
        
        # All images processed successfully
        assert result[0].metadata.get('has_unprocessed_images') is False
    
    def test_chunks_without_images_unchanged(self, mock_settings, mock_vision_llm, sample_chunks_no_images):
        """Test that chunks without images are returned unchanged."""
        captioner = ImageCaptioner(mock_settings, vision_llm=mock_vision_llm)
        result = captioner.transform(sample_chunks_no_images)
        
        # No image-related metadata should be added
        assert 'image_captions' not in result[0].metadata
        assert 'has_unprocessed_images' not in result[0].metadata
        assert result[0].text == sample_chunks_no_images[0].text


# ==================== Fallback Behavior Tests (5) ====================

class TestFallbackBehavior:
    """Tests for fallback behavior when Vision LLM is unavailable."""
    
    def test_fallback_when_vision_llm_not_configured(self, mock_settings_no_vision_llm, sample_chunks_with_images):
        """Test fallback when Vision LLM is not configured."""
        with patch('libs.llm.llm_factory.LLMFactory.create_vision_llm', return_value=None):
            captioner = ImageCaptioner(mock_settings_no_vision_llm)
        
        result = captioner.transform(sample_chunks_with_images)
        
        # Should mark as unprocessed
        assert result[0].metadata.get('has_unprocessed_images') is True
        # No captions should be generated
        assert 'image_captions' not in result[0].metadata or not result[0].metadata.get('image_captions')
    
    def test_fallback_when_vision_llm_call_fails(self, mock_settings, mock_vision_llm, sample_chunks_with_images):
        """Test fallback when Vision LLM call fails."""
        mock_vision_llm.chat_with_image.side_effect = Exception("API Error")
        
        captioner = ImageCaptioner(mock_settings, vision_llm=mock_vision_llm)
        
        with patch.object(captioner, '_resolve_image_path', return_value='/fake/path/img.png'):
            with patch('pathlib.Path.exists', return_value=True):
                result = captioner.transform(sample_chunks_with_images)
        
        # Should mark as unprocessed
        assert result[0].metadata.get('has_unprocessed_images') is True
        # No captions should be generated
        assert 'image_captions' not in result[0].metadata or not result[0].metadata.get('image_captions')
    
    def test_partial_fallback_single_image_failure(self, mock_settings, mock_vision_llm, sample_chunks_with_images):
        """Test partial fallback when one image fails but others succeed."""
        # First call succeeds, second fails
        mock_vision_llm.chat_with_image.side_effect = [
            VisionResponse(content="Success caption", model="gpt-4o"),
            Exception("API Error")
        ]
        
        captioner = ImageCaptioner(mock_settings, vision_llm=mock_vision_llm)
        
        # Process chunk with two images
        with patch.object(captioner, '_resolve_image_path', return_value='/fake/path/img.png'):
            with patch('pathlib.Path.exists', return_value=True):
                result = captioner.transform([sample_chunks_with_images[1]])
        
        # Should have one caption and mark unprocessed
        assert 'image_captions' in result[0].metadata
        assert len(result[0].metadata['image_captions']) == 1
        assert result[0].metadata.get('has_unprocessed_images') is True
    
    def test_image_refs_preserved_in_fallback(self, mock_settings_no_vision_llm, sample_chunks_with_images):
        """Test that image_refs are preserved in fallback mode."""
        with patch('libs.llm.llm_factory.LLMFactory.create_vision_llm', return_value=None):
            captioner = ImageCaptioner(mock_settings_no_vision_llm)
        
        original_refs = sample_chunks_with_images[0].metadata['image_refs']
        result = captioner.transform(sample_chunks_with_images)
        
        # Original image_refs should be preserved
        assert result[0].metadata.get('image_refs') == original_refs
    
    def test_disabled_captioner_returns_unchanged(self, mock_settings_disabled, sample_chunks_with_images):
        """Test that disabled captioner returns chunks unchanged."""
        captioner = ImageCaptioner(mock_settings_disabled)
        result = captioner.transform(sample_chunks_with_images)
        
        # Should return same chunks unchanged
        assert result[0].text == sample_chunks_with_images[0].text
        assert result[0].metadata == sample_chunks_with_images[0].metadata
        # No image captioning metadata should be added
        assert 'image_captions' not in result[0].metadata
        assert 'has_unprocessed_images' not in result[0].metadata


# ==================== Configuration Tests (3) ====================

class TestConfiguration:
    """Tests for configuration handling."""
    
    def test_enabled_flag_respected(self, mock_settings):
        """Test that enabled flag is respected from config."""
        captioner = ImageCaptioner(mock_settings)
        assert captioner.enabled is True
    
    def test_disabled_flag_respected(self, mock_settings_disabled):
        """Test that disabled flag is respected from config."""
        captioner = ImageCaptioner(mock_settings_disabled)
        assert captioner.enabled is False
    
    def test_custom_prompt_path(self, mock_settings):
        """Test custom prompt path configuration."""
        captioner = ImageCaptioner(mock_settings, prompt_path="custom/prompt.txt")
        assert captioner.prompt_path == "custom/prompt.txt"


# ==================== Error Handling Tests (2) ====================

class TestErrorHandling:
    """Tests for error handling scenarios."""
    
    def test_single_chunk_error_does_not_fail_batch(self, mock_settings, mock_vision_llm, sample_chunks_with_images):
        """Test that error in one chunk doesn't fail entire batch."""
        captioner = ImageCaptioner(mock_settings, vision_llm=mock_vision_llm)
        
        # Mock _process_chunk to fail on first chunk
        original_process = captioner._process_chunk
        call_count = [0]
        
        def mock_process(chunk, trace=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Test error")
            return original_process(chunk, trace)
        
        captioner._process_chunk = mock_process
        
        result = captioner.transform(sample_chunks_with_images)
        
        # Should still return all chunks
        assert len(result) == len(sample_chunks_with_images)
        # First chunk should have fallback metadata
        assert result[0].metadata.get('has_unprocessed_images') is True
    
    def test_empty_chunks_list(self, mock_settings, mock_vision_llm):
        """Test handling of empty chunks list."""
        captioner = ImageCaptioner(mock_settings, vision_llm=mock_vision_llm)
        result = captioner.transform([])
        assert result == []


# ==================== Trace Integration Tests ====================

class TestTraceIntegration:
    """Tests for trace context integration."""
    
    def test_trace_records_stages(self, mock_settings, mock_vision_llm, sample_chunks_with_images, trace_context):
        """Test that trace stages are recorded."""
        mock_vision_llm.chat_with_image.return_value = VisionResponse(
            content="Test caption", model="gpt-4o"
        )
        
        captioner = ImageCaptioner(mock_settings, vision_llm=mock_vision_llm)
        
        with patch.object(captioner, '_resolve_image_path', return_value='/fake/path/img.png'):
            with patch('pathlib.Path.exists', return_value=True):
                captioner.transform(sample_chunks_with_images, trace=trace_context)
        
        # Verify trace stages were recorded
        stage_names = [s["name"] for s in trace_context.stages]
        assert "image_captioning" in stage_names
        assert "image_captioning_complete" in stage_names
    
    def test_trace_includes_stats(self, mock_settings, mock_vision_llm, sample_chunks_with_images, trace_context):
        """Test that trace includes processing stats."""
        mock_vision_llm.chat_with_image.return_value = VisionResponse(
            content="Test caption", model="gpt-4o"
        )
        
        captioner = ImageCaptioner(mock_settings, vision_llm=mock_vision_llm)
        
        with patch.object(captioner, '_resolve_image_path', return_value='/fake/path/img.png'):
            with patch('pathlib.Path.exists', return_value=True):
                captioner.transform(sample_chunks_with_images, trace=trace_context)
        
        # Find complete stage
        complete_stage = next(
            s for s in trace_context.stages if s["name"] == "image_captioning_complete"
        )
        assert complete_stage["details"]["input_count"] == len(sample_chunks_with_images)
        assert complete_stage["details"]["total_images"] == 3  # 1 + 2 images
        assert complete_stage["details"]["processed_images"] == 3


# ==================== Vision LLM Factory Integration Tests ====================

class TestVisionLLMFactoryIntegration:
    """Tests for Vision LLM factory integration."""
    
    def test_factory_creates_vision_llm_when_needed(self, mock_settings):
        """Test that LLM factory is called when Vision LLM is needed."""
        with patch('ingestion.transform.image_captioner.LLMFactory') as mock_factory:
            mock_vision_llm = Mock()
            mock_factory.create_vision_llm.return_value = mock_vision_llm
            
            captioner = ImageCaptioner(mock_settings)
            
            mock_factory.create_vision_llm.assert_called_once_with(mock_settings)
            assert captioner._vision_llm is mock_vision_llm
    
    def test_factory_failure_graceful(self, mock_settings):
        """Test graceful handling when Vision LLM factory fails."""
        with patch('ingestion.transform.image_captioner.LLMFactory') as mock_factory:
            mock_factory.create_vision_llm.side_effect = Exception("Factory error")
            
            captioner = ImageCaptioner(mock_settings)
            
            # Should continue without Vision LLM
            assert captioner.enabled is True  # Config says enabled
            assert captioner._vision_llm is None  # But no LLM available


# ==================== Image Path Resolution Tests ====================

class TestImagePathResolution:
    """Tests for image path resolution."""
    
    def test_resolve_image_path_with_provided_path(self, mock_settings):
        """Test path resolution when path is provided."""
        captioner = ImageCaptioner(mock_settings)
        
        with patch('pathlib.Path.exists', return_value=True):
            result = captioner._resolve_image_path('/provided/path.png', 'img_001')
        
        assert result == '/provided/path.png'
    
    def test_resolve_image_path_convention_fallback(self, mock_settings):
        """Test path resolution using convention when path not provided."""
        captioner = ImageCaptioner(mock_settings)
        
        # Mock Path.exists to return True only for the convention path
        def mock_exists(self):
            return 'data/images/img_001.png' in str(self)
        
        with patch.object(Path, 'exists', mock_exists):
            result = captioner._resolve_image_path(None, 'img_001')
        
        assert result is not None
        assert 'img_001' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
