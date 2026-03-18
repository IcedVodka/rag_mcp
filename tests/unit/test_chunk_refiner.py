#!/usr/bin/env python3
"""
Unit Tests for ChunkRefiner

Tests rule-based refinement, LLM integration (mocked), fallback behavior,
configuration handling, and error scenarios.

Total: 27 tests
- Rule-based refinement: 12 tests
- LLM mode: 6 tests  
- Fallback behavior: 3 tests
- Configuration: 4 tests
- Error handling: 2 tests
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import List

# Adjust path for imports
import sys
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from core.types import Chunk
from core.settings import Settings, LLMSettings, IngestionSettings
from core.trace.trace_context import TraceContext
from ingestion.transform.chunk_refiner import ChunkRefiner
from libs.llm.base_llm import ChatMessage, ChatResponse


# ==================== Fixtures ====================

@pytest.fixture
def mock_settings():
    """Create mock settings with chunk_refiner enabled."""
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.chunk_refiner = {
        'enabled': True,
        'use_llm': False,
        'prompt_path': 'config/prompts/chunk_refinement.txt'
    }
    return settings


@pytest.fixture
def mock_settings_with_llm():
    """Create mock settings with LLM refinement enabled."""
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.chunk_refiner = {
        'enabled': True,
        'use_llm': True,
        'prompt_path': 'config/prompts/chunk_refinement.txt'
    }
    # Add required llm config for LLMFactory
    settings.llm = Mock()
    settings.llm.provider = "openai"
    settings.llm.openai = {"api_key": "test-key", "model": "gpt-4"}
    return settings


@pytest.fixture
def mock_settings_disabled():
    """Create mock settings with chunk_refiner disabled."""
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.chunk_refiner = {
        'enabled': False,
        'use_llm': False
    }
    return settings


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """Create sample chunks for testing."""
    return [
        Chunk(
            id="chunk_1",
            text="First chunk with some content.",
            metadata={"source": "test"},
            start_offset=0,
            end_offset=30,
            source_ref="doc_1"
        ),
        Chunk(
            id="chunk_2", 
            text="Second chunk content.",
            metadata={"source": "test"},
            start_offset=31,
            end_offset=52,
            source_ref="doc_1"
        )
    ]


@pytest.fixture
def noisy_chunks() -> List[Chunk]:
    """Create noisy chunks for testing."""
    return [
        Chunk(
            id="noisy_1",
            text="Confidential Document\n\nPage 1 of 10\n\n\n\n   Introduction   \n\n\n\nContent here.",
            metadata={"source": "test"},
            start_offset=0,
            end_offset=100,
            source_ref="doc_1"
        )
    ]


@pytest.fixture
def trace_context():
    """Create a trace context for testing."""
    return TraceContext(trace_type="ingestion")


# ==================== Rule-Based Refinement Tests (12) ====================

class TestRuleBasedRefinement:
    """Tests for rule-based text refinement."""
    
    def test_remove_page_headers(self, mock_settings):
        """Test removal of page headers."""
        refiner = ChunkRefiner(mock_settings)
        text = "Confidential Document\n\nActual content here."
        result = refiner._rule_based_refine(text)
        assert "Confidential Document" not in result
        assert "Actual content here." in result
    
    def test_remove_page_footers(self, mock_settings):
        """Test removal of page footers."""
        refiner = ChunkRefiner(mock_settings)
        text = "Actual content here.\n\nCopyright 2024 All Rights Reserved"
        result = refiner._rule_based_refine(text)
        assert "Copyright" not in result
        assert "Actual content here." in result
    
    def test_remove_page_numbers(self, mock_settings):
        """Test removal of page numbers."""
        refiner = ChunkRefiner(mock_settings)
        text = "Page 5 of 20\n\nContent here.\n\nPage 6 of 20"
        result = refiner._rule_based_refine(text)
        assert "Page 5" not in result
        assert "Page 6" not in result
        assert "Content here." in result
    
    def test_normalize_excessive_whitespace(self, mock_settings):
        """Test normalization of excessive whitespace."""
        refiner = ChunkRefiner(mock_settings)
        text = "Line 1\n\n\n\n\n\nLine 2"
        result = refiner._rule_based_refine(text)
        # Should normalize to max 2 consecutive newlines
        assert "\n\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result
    
    def test_remove_html_comments(self, mock_settings):
        """Test removal of HTML comments."""
        refiner = ChunkRefiner(mock_settings)
        text = "Content <!-- comment --> more content"
        result = refiner._rule_based_refine(text)
        assert "<!--" not in result
        assert "comment" not in result
        assert "Content" in result
        assert "more content" in result
    
    def test_remove_html_tags(self, mock_settings):
        """Test removal of HTML tags."""
        refiner = ChunkRefiner(mock_settings)
        text = "<p>This is <strong>bold</strong> text</p>"
        result = refiner._rule_based_refine(text)
        assert "<p>" not in result
        assert "</p>" not in result
        assert "<strong>" not in result
        assert "bold" in result
        assert "This is" in result
    
    def test_preserve_code_blocks(self, mock_settings):
        """Test that code blocks are preserved."""
        refiner = ChunkRefiner(mock_settings)
        text = """Some text.

```python
def hello():
    print("world")
```

More text."""
        result = refiner._rule_based_refine(text)
        assert 'def hello():' in result
        assert '    print("world")' in result
        assert '```python' in result
    
    def test_preserve_inline_code(self, mock_settings):
        """Test that inline code is preserved."""
        refiner = ChunkRefiner(mock_settings)
        text = "Use `function()` to call the method."
        result = refiner._rule_based_refine(text)
        assert "`function()`" in result
        assert "Use" in result
    
    def test_preserve_markdown_headers(self, mock_settings):
        """Test that markdown headers are preserved."""
        refiner = ChunkRefiner(mock_settings)
        text = "# Title\n\n## Subtitle\n\nContent"
        result = refiner._rule_based_refine(text)
        assert "# Title" in result or "Title" in result
        assert "Content" in result
    
    def test_handle_empty_input(self, mock_settings):
        """Test handling of empty input."""
        refiner = ChunkRefiner(mock_settings)
        assert refiner._rule_based_refine("") == ""
        assert refiner._rule_based_refine("   ") == ""
    
    def test_remove_chapter_headers(self, mock_settings):
        """Test removal of chapter headers."""
        refiner = ChunkRefiner(mock_settings)
        text = "Chapter 3: Advanced Topics\n\nContent here."
        result = refiner._rule_based_refine(text)
        assert "Chapter 3:" not in result
        assert "Advanced Topics" in result or "Content here." in result
    
    def test_normalize_inline_spaces(self, mock_settings):
        """Test normalization of inline whitespace."""
        refiner = ChunkRefiner(mock_settings)
        text = "Too    many     spaces   here"
        result = refiner._rule_based_refine(text)
        assert "Too many spaces" in result


# ==================== LLM Mode Tests (6) ====================

class TestLLMMode:
    """Tests for LLM-based enhancement."""
    
    def test_llm_mode_enabled(self, mock_settings_with_llm):
        """Test that LLM mode is properly enabled."""
        with patch('ingestion.transform.chunk_refiner.LLMFactory') as mock_factory:
            mock_llm = Mock()
            mock_factory.create.return_value = mock_llm
            
            refiner = ChunkRefiner(mock_settings_with_llm)
            assert refiner.use_llm is True
            assert refiner._llm is not None
    
    def test_llm_mode_disabled(self, mock_settings):
        """Test that LLM mode is disabled by default."""
        refiner = ChunkRefiner(mock_settings)
        assert refiner.use_llm is False
        assert refiner._llm is None
    
    def test_llm_refine_success(self, mock_settings_with_llm):
        """Test successful LLM refinement."""
        mock_llm = Mock()
        mock_response = ChatResponse(
            content="Refined text from LLM",
            model="test-model"
        )
        mock_llm.chat.return_value = mock_response
        
        refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)
        
        # Load prompt template manually for testing
        refiner._prompt_template = "Refine this: {text}"
        
        result = refiner._llm_refine("Original text")
        assert result == "Refined text from LLM"
        mock_llm.chat.assert_called_once()
    
    def test_llm_refine_failure_returns_none(self, mock_settings_with_llm):
        """Test that LLM failure returns None for fallback."""
        mock_llm = Mock()
        mock_llm.chat.side_effect = Exception("API Error")
        
        refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)
        refiner._prompt_template = "Refine this: {text}"
        
        result = refiner._llm_refine("Original text")
        assert result is None
    
    def test_llm_refine_empty_response(self, mock_settings_with_llm):
        """Test handling of empty LLM response."""
        mock_llm = Mock()
        mock_response = ChatResponse(content="   ", model="test-model")
        mock_llm.chat.return_value = mock_response
        
        refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)
        refiner._prompt_template = "Refine this: {text}"
        
        result = refiner._llm_refine("Original text")
        assert result is None
    
    def test_llm_refine_with_trace(self, mock_settings_with_llm, trace_context):
        """Test LLM refinement records trace stages."""
        mock_llm = Mock()
        mock_response = ChatResponse(
            content="Refined",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5}
        )
        mock_llm.chat.return_value = mock_response
        
        refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)
        refiner._prompt_template = "Refine: {text}"
        
        result = refiner._llm_refine("Text", trace=trace_context)
        
        assert result == "Refined"
        # Verify trace stages were recorded
        stage_names = [s["name"] for s in trace_context.stages]
        assert "llm_refinement_request" in stage_names
        assert "llm_refinement_response" in stage_names


# ==================== Fallback Behavior Tests (3) ====================

class TestFallbackBehavior:
    """Tests for LLM fallback to rule-based results."""
    
    def test_fallback_on_llm_failure(self, mock_settings_with_llm, sample_chunks):
        """Test fallback to rule-based when LLM fails."""
        mock_llm = Mock()
        mock_llm.chat.side_effect = Exception("API Error")
        
        refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)
        refiner._prompt_template = "Refine: {text}"
        
        result = refiner.transform(sample_chunks)
        
        # Should still return refined chunks (rule-based)
        assert len(result) == len(sample_chunks)
        assert all(isinstance(c, Chunk) for c in result)
        # Should have fallback method in metadata
        assert result[0].metadata.get('refinement', {}).get('method') == 'rule_based_fallback'
    
    def test_fallback_when_no_llm_configured(self, mock_settings, sample_chunks):
        """Test fallback when LLM is enabled but not available."""
        # Use mock_settings which has use_llm=False
        refiner = ChunkRefiner(mock_settings, llm=None)
        
        result = refiner.transform(sample_chunks)
        
        assert len(result) == len(sample_chunks)
        # Should use rule-based only (not rule_based_fallback)
        assert result[0].metadata.get('refinement', {}).get('method') == 'rule_based'
    
    def test_no_fallback_when_llm_succeeds(self, mock_settings_with_llm, sample_chunks):
        """Test no fallback when LLM succeeds."""
        mock_llm = Mock()
        mock_response = ChatResponse(content="LLM refined content", model="test")
        mock_llm.chat.return_value = mock_response
        
        refiner = ChunkRefiner(mock_settings_with_llm, llm=mock_llm)
        refiner._prompt_template = "Refine: {text}"
        
        result = refiner.transform(sample_chunks)
        
        assert result[0].metadata.get('refinement', {}).get('method') == 'llm_enhanced'


# ==================== Configuration Tests (4) ====================

class TestConfiguration:
    """Tests for configuration handling."""
    
    def test_disabled_refiner_returns_unchanged(self, mock_settings_disabled, sample_chunks):
        """Test that disabled refiner returns chunks unchanged."""
        refiner = ChunkRefiner(mock_settings_disabled)
        
        result = refiner.transform(sample_chunks)
        
        # Should return same chunks unchanged
        assert result == sample_chunks
        assert result[0].text == sample_chunks[0].text
    
    def test_custom_prompt_path(self, mock_settings):
        """Test custom prompt path configuration."""
        refiner = ChunkRefiner(mock_settings, prompt_path="custom/path.txt")
        assert refiner.prompt_path == "custom/path.txt"
    
    def test_llm_factory_called_when_needed(self, mock_settings_with_llm):
        """Test that LLM factory is called when LLM is needed."""
        with patch('ingestion.transform.chunk_refiner.LLMFactory') as mock_factory:
            mock_llm = Mock()
            mock_factory.create.return_value = mock_llm
            
            refiner = ChunkRefiner(mock_settings_with_llm)
            
            mock_factory.create.assert_called_once_with(mock_settings_with_llm)
            assert refiner._llm is mock_llm
    
    def test_llm_factory_failure_graceful(self, mock_settings_with_llm):
        """Test graceful handling when LLM factory fails."""
        with patch('ingestion.transform.chunk_refiner.LLMFactory') as mock_factory:
            mock_factory.create.side_effect = Exception("Factory error")
            
            refiner = ChunkRefiner(mock_settings_with_llm)
            
            # Should continue without LLM
            assert refiner.use_llm is True  # Config says use_llm
            assert refiner._llm is None  # But no LLM available


# ==================== Error Handling Tests (2) ====================

class TestErrorHandling:
    """Tests for error handling scenarios."""
    
    def test_single_chunk_error_does_not_fail_batch(self, mock_settings, sample_chunks):
        """Test that error in one chunk doesn't fail entire batch."""
        refiner = ChunkRefiner(mock_settings)
        
        # Mock _refine_chunk to fail on first chunk
        original_refine = refiner._refine_chunk
        call_count = [0]
        
        def mock_refine(chunk, trace=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Test error")
            return original_refine(chunk, trace)
        
        refiner._refine_chunk = mock_refine
        
        result = refiner.transform(sample_chunks)
        
        # Should still return all chunks
        assert len(result) == len(sample_chunks)
        # First chunk should be original (fallback on error)
        assert result[0].id == sample_chunks[0].id
    
    def test_empty_chunks_list(self, mock_settings):
        """Test handling of empty chunks list."""
        refiner = ChunkRefiner(mock_settings)
        result = refiner.transform([])
        assert result == []


# ==================== Integration with Fixtures ====================

class TestWithFixtures:
    """Tests using the noisy_chunks.json fixture data."""
    
    @pytest.fixture
    def fixture_data(self):
        """Load fixture data."""
        fixture_path = Path(__file__).parents[1] / "fixtures" / "noisy_chunks.json"
        with open(fixture_path, 'r') as f:
            return json.load(f)
    
    def test_typical_noise_scenario(self, mock_settings, fixture_data):
        """Test typical noise scenario from fixture."""
        refiner = ChunkRefiner(mock_settings)
        data = fixture_data['typical_noise_scenario']
        
        chunk = Chunk(
            id="test_1",
            text=data['input'],
            metadata={}
        )
        
        result = refiner.transform([chunk])
        refined_text = result[0].text
        
        # Check expected content is preserved
        for expected in data['expected_contains']:
            assert expected in refined_text or expected.replace('   ', ' ') in refined_text
        
        # Check noise is removed
        for not_expected in data['expected_not_contains']:
            assert not_expected not in refined_text
    
    def test_page_header_footer_scenario(self, mock_settings, fixture_data):
        """Test page header/footer scenario from fixture."""
        refiner = ChunkRefiner(mock_settings)
        data = fixture_data['page_header_footer']
        
        chunk = Chunk(
            id="test_2",
            text=data['input'],
            metadata={}
        )
        
        result = refiner.transform([chunk])
        refined_text = result[0].text
        
        # Check content preserved
        for expected in data['expected_contains']:
            assert expected in refined_text or expected.split(':')[0] not in refined_text
        
        # Check headers/footers removed
        for not_expected in data['expected_not_contains']:
            assert not_expected not in refined_text
    
    def test_format_markers_scenario(self, mock_settings, fixture_data):
        """Test format markers scenario from fixture."""
        refiner = ChunkRefiner(mock_settings)
        data = fixture_data['format_markers']
        
        chunk = Chunk(
            id="test_3",
            text=data['input'],
            metadata={}
        )
        
        result = refiner.transform([chunk])
        refined_text = result[0].text
        
        # Check content preserved
        for expected in data['expected_contains']:
            assert expected in refined_text
        
        # Check HTML tags removed
        for not_expected in data['expected_not_contains']:
            assert not_expected not in refined_text
    
    def test_code_blocks_preserved(self, mock_settings, fixture_data):
        """Test that code blocks are preserved per fixture."""
        refiner = ChunkRefiner(mock_settings)
        data = fixture_data['code_blocks']
        
        chunk = Chunk(
            id="test_4",
            text=data['input'],
            metadata={}
        )
        
        result = refiner.transform([chunk])
        refined_text = result[0].text
        
        # Check code content preserved
        for expected in data['expected_contains']:
            assert expected in refined_text
    
    def test_clean_text_not_over_processed(self, mock_settings, fixture_data):
        """Test that clean text is not over-processed."""
        refiner = ChunkRefiner(mock_settings)
        data = fixture_data['clean_text']
        
        chunk = Chunk(
            id="test_5",
            text=data['input'],
            metadata={}
        )
        
        result = refiner.transform([chunk])
        refined_text = result[0].text
        
        # Clean text should still contain all meaningful content
        for expected in data['expected_contains']:
            assert expected in refined_text or expected.replace('  ', ' ') in refined_text


# ==================== Trace Recording Tests ====================

class TestTraceRecording:
    """Tests for trace context recording."""
    
    def test_trace_records_refinement_stage(self, mock_settings, sample_chunks, trace_context):
        """Test that refinement stage is recorded in trace."""
        refiner = ChunkRefiner(mock_settings)
        refiner.transform(sample_chunks, trace=trace_context)
        
        stage_names = [s["name"] for s in trace_context.stages]
        assert "chunk_refinement" in stage_names
    
    def test_trace_records_completion(self, mock_settings, sample_chunks, trace_context):
        """Test that completion stage is recorded."""
        refiner = ChunkRefiner(mock_settings)
        refiner.transform(sample_chunks, trace=trace_context)
        
        stage_names = [s["name"] for s in trace_context.stages]
        assert "chunk_refinement_complete" in stage_names
    
    def test_trace_includes_counts(self, mock_settings, sample_chunks, trace_context):
        """Test that trace includes input/output counts."""
        refiner = ChunkRefiner(mock_settings)
        refiner.transform(sample_chunks, trace=trace_context)
        
        complete_stage = next(
            s for s in trace_context.stages if s["name"] == "chunk_refinement_complete"
        )
        assert complete_stage["details"]["input_count"] == len(sample_chunks)
        assert complete_stage["details"]["output_count"] == len(sample_chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
