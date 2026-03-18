#!/usr/bin/env python3
"""
Unit Tests for MetadataEnricher

Tests rule-based enrichment, LLM integration (mocked), fallback behavior,
configuration handling, output format compliance, and error scenarios.

Total: 29 tests
- Rule-based enrichment: 11 tests
- LLM mode: 8 tests
- Fallback behavior: 3 tests
- Configuration: 4 tests
- Output format: 3 tests
- Error handling: 2 tests
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from typing import List
from datetime import datetime

# Adjust path for imports
import sys
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from core.types import Chunk
from core.settings import Settings
from core.trace.trace_context import TraceContext
from ingestion.transform.metadata_enricher import MetadataEnricher
from libs.llm.base_llm import ChatMessage, ChatResponse


# ==================== Fixtures ====================

@pytest.fixture
def mock_settings():
    """Create mock settings with metadata_enricher enabled (rule-based only)."""
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.metadata_enricher = {
        'enabled': True,
        'use_llm': False,
    }
    return settings


@pytest.fixture
def mock_settings_with_llm():
    """Create mock settings with LLM enrichment enabled."""
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.metadata_enricher = {
        'enabled': True,
        'use_llm': True,
    }
    # Add required llm config for LLMFactory
    settings.llm = Mock()
    settings.llm.provider = "openai"
    settings.llm.openai = {"api_key": "test-key", "model": "gpt-4"}
    return settings


@pytest.fixture
def mock_settings_disabled():
    """Create mock settings with metadata_enricher disabled."""
    settings = Mock(spec=Settings)
    settings.ingestion = Mock()
    settings.ingestion.metadata_enricher = {
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
            text="Introduction to Machine Learning\n\nMachine learning is a subset of "
                 "artificial intelligence that enables systems to learn from data.",
            metadata={"source": "test"},
            start_offset=0,
            end_offset=100,
            source_ref="doc_1"
        ),
        Chunk(
            id="chunk_2",
            text="Python Programming Basics\n\nPython is a versatile programming language "
                 "widely used for data science, web development, and automation.",
            metadata={"source": "test"},
            start_offset=101,
            end_offset=200,
            source_ref="doc_1"
        )
    ]


@pytest.fixture
def technical_chunk() -> Chunk:
    """Create a technical chunk for tag extraction testing."""
    return Chunk(
        id="tech_chunk",
        text="Implementing REST API with FastAPI and Python\n\n"
             "FastAPI is a modern, fast web framework for building APIs with Python 3.7+. "
             "It is based on standard Python type hints and uses Starlette for the web parts "
             "and Pydantic for the data parts. The framework provides automatic API documentation "
             "using OpenAPI and JSON Schema. Version 0.100.0 includes new features for "
             "dependency injection and async database support with SQLAlchemy.",
        metadata={"source": "test"},
        start_offset=0,
        end_offset=400,
        source_ref="doc_1"
    )


@pytest.fixture
def trace_context():
    """Create a trace context for testing."""
    return TraceContext(trace_type="ingestion")


# ==================== Rule-Based Enrichment Tests (11) ====================

class TestRuleBasedEnrichment:
    """Tests for rule-based metadata extraction."""
    
    def test_extract_title_from_first_line(self, mock_settings):
        """Test title extraction from first line."""
        enricher = MetadataEnricher(mock_settings)
        chunk = Chunk(
            id="test_1",
            text="This is the Title\n\nThis is the content.",
            metadata={}
        )
        
        result = enricher._rule_based_enrich(chunk)
        assert result['title'] == "This is the Title"
    
    def test_extract_title_truncate_long_first_line(self, mock_settings):
        """Test title truncation when first line is too long."""
        enricher = MetadataEnricher(mock_settings)
        long_title = "A" * 100
        chunk = Chunk(
            id="test_1",
            text=f"{long_title}\n\nContent here.",
            metadata={}
        )
        
        result = enricher._rule_based_enrich(chunk)
        assert len(result['title']) <= 60  # 50 + "..."
        assert result['title'].endswith("...")
    
    def test_extract_title_remove_markdown_header(self, mock_settings):
        """Test removal of markdown header markers from title."""
        enricher = MetadataEnricher(mock_settings)
        chunk = Chunk(
            id="test_1",
            text="## Section Title\n\nContent here.",
            metadata={}
        )
        
        result = enricher._rule_based_enrich(chunk)
        assert "##" not in result['title']
        assert result['title'] == "Section Title"
    
    def test_extract_summary_truncate(self, mock_settings):
        """Test summary extraction with truncation."""
        enricher = MetadataEnricher(mock_settings)
        chunk = Chunk(
            id="test_1",
            text="First sentence here. Second sentence here. " * 20,
            metadata={}
        )
        
        result = enricher._rule_based_enrich(chunk)
        assert len(result['summary']) <= 110  # Around 100 + "..."
        assert result['summary'].endswith("...")
    
    def test_extract_summary_short_text(self, mock_settings):
        """Test summary extraction with short text (no truncation)."""
        enricher = MetadataEnricher(mock_settings)
        short_text = "Short text content."
        chunk = Chunk(
            id="test_1",
            text=short_text,
            metadata={}
        )
        
        result = enricher._rule_based_enrich(chunk)
        assert result['summary'] == short_text
    
    def test_extract_tags_technical_terms(self, mock_settings, technical_chunk):
        """Test extraction of technical terms as tags."""
        enricher = MetadataEnricher(mock_settings)
        result = enricher._rule_based_enrich(technical_chunk)
        
        # Should extract technical terms
        tags_lower = [t.lower() for t in result['tags']]
        assert any('fastapi' in t.lower() for t in result['tags'])
        assert any('python' in t.lower() for t in result['tags'])
    
    def test_extract_tags_max_limit(self, mock_settings, technical_chunk):
        """Test that tag extraction respects max limit."""
        enricher = MetadataEnricher(mock_settings)
        result = enricher._rule_based_enrich(technical_chunk)
        
        assert len(result['tags']) <= 5  # Default max_tags
    
    def test_extract_tags_excludes_stop_words(self, mock_settings):
        """Test that stop words are excluded from tags."""
        enricher = MetadataEnricher(mock_settings)
        chunk = Chunk(
            id="test_1",
            text="The and or but are common words that should be excluded.",
            metadata={}
        )
        
        result = enricher._rule_based_enrich(chunk)
        
        # Should not include common stop words
        stop_words = {'the', 'and', 'or', 'but', 'are', 'that', 'be'}
        tags_lower = set(t.lower() for t in result['tags'])
        assert not stop_words.intersection(tags_lower)
    
    def test_enrich_empty_chunk(self, mock_settings):
        """Test enrichment of empty chunk."""
        enricher = MetadataEnricher(mock_settings)
        chunk = Chunk(
            id="test_1",
            text="",
            metadata={}
        )
        
        result = enricher._rule_based_enrich(chunk)
        assert result['title'] == ""
        assert result['summary'] == ""
        assert result['tags'] == []
    
    def test_enriched_by_is_rule(self, mock_settings, sample_chunks):
        """Test that rule-based enrichment sets enriched_by to 'rule'."""
        enricher = MetadataEnricher(mock_settings)
        result = enricher._rule_based_enrich(sample_chunks[0])
        
        assert result['enriched_by'] == 'rule'
        assert 'enriched_at' in result
    
    def test_enriched_at_timestamp_format(self, mock_settings, sample_chunks):
        """Test that enriched_at is valid ISO timestamp."""
        enricher = MetadataEnricher(mock_settings)
        result = enricher._rule_based_enrich(sample_chunks[0])
        
        # Should be parseable as ISO timestamp
        try:
            datetime.fromisoformat(result['enriched_at'])
            assert True
        except (ValueError, KeyError):
            pytest.fail("enriched_at is not a valid ISO timestamp")


# ==================== LLM Mode Tests (8) ====================

class TestLLMMode:
    """Tests for LLM-based metadata enrichment."""
    
    def test_llm_mode_enabled(self, mock_settings_with_llm):
        """Test that LLM mode is properly enabled."""
        with patch('ingestion.transform.metadata_enricher.LLMFactory') as mock_factory:
            mock_llm = Mock()
            mock_factory.create.return_value = mock_llm
            
            enricher = MetadataEnricher(mock_settings_with_llm)
            assert enricher.use_llm is True
            assert enricher._llm is not None
    
    def test_llm_mode_disabled(self, mock_settings):
        """Test that LLM mode is disabled by default."""
        enricher = MetadataEnricher(mock_settings)
        assert enricher.use_llm is False
        assert enricher._llm is None
    
    def test_llm_enrich_success(self, mock_settings_with_llm):
        """Test successful LLM metadata enrichment."""
        mock_llm = Mock()
        mock_response = ChatResponse(
            content=json.dumps({
                "title": "LLM Generated Title",
                "summary": "LLM generated summary of the content.",
                "tags": ["ai", "machine-learning", "python"]
            }),
            model="test-model"
        )
        mock_llm.chat.return_value = mock_response
        
        enricher = MetadataEnricher(mock_settings_with_llm, llm=mock_llm)
        
        chunk = Chunk(
            id="test_1",
            text="Some test content about AI and machine learning.",
            metadata={}
        )
        
        result = enricher._llm_enrich(chunk)
        
        assert result is not None
        assert result['title'] == "LLM Generated Title"
        assert result['summary'] == "LLM generated summary of the content."
        assert result['tags'] == ["ai", "machine-learning", "python"]
        assert result['enriched_by'] == 'llm'
    
    def test_llm_enrich_text_format_response(self, mock_settings_with_llm):
        """Test LLM enrichment with text format response (not JSON)."""
        mock_llm = Mock()
        mock_response = ChatResponse(
            content="Title: Neural Networks\nSummary: Introduction to neural networks\nTags: ai, deep-learning, python",
            model="test-model"
        )
        mock_llm.chat.return_value = mock_response
        
        enricher = MetadataEnricher(mock_settings_with_llm, llm=mock_llm)
        
        chunk = Chunk(
            id="test_1",
            text="Content about neural networks.",
            metadata={}
        )
        
        result = enricher._llm_enrich(chunk)
        
        assert result is not None
        assert "Neural Networks" in result['title']
        assert "neural networks" in result['summary'].lower()
    
    def test_llm_enrich_failure_returns_none(self, mock_settings_with_llm):
        """Test that LLM failure returns None for fallback."""
        mock_llm = Mock()
        mock_llm.chat.side_effect = Exception("API Error")
        
        enricher = MetadataEnricher(mock_settings_with_llm, llm=mock_llm)
        
        chunk = Chunk(
            id="test_1",
            text="Some content.",
            metadata={}
        )
        
        result = enricher._llm_enrich(chunk)
        assert result is None
    
    def test_llm_enrich_empty_response(self, mock_settings_with_llm):
        """Test handling of empty LLM response."""
        mock_llm = Mock()
        mock_response = ChatResponse(content="   ", model="test-model")
        mock_llm.chat.return_value = mock_response
        
        enricher = MetadataEnricher(mock_settings_with_llm, llm=mock_llm)
        
        chunk = Chunk(
            id="test_1",
            text="Some content.",
            metadata={}
        )
        
        result = enricher._llm_enrich(chunk)
        assert result is None
    
    def test_llm_enrich_invalid_json_response(self, mock_settings_with_llm):
        """Test handling of invalid JSON in LLM response."""
        mock_llm = Mock()
        mock_response = ChatResponse(content="Not valid JSON", model="test-model")
        mock_llm.chat.return_value = mock_response
        
        enricher = MetadataEnricher(mock_settings_with_llm, llm=mock_llm)
        
        chunk = Chunk(
            id="test_1",
            text="Some content.",
            metadata={}
        )
        
        result = enricher._llm_enrich(chunk)
        assert result is None
    
    def test_llm_enrich_with_trace(self, mock_settings_with_llm, trace_context):
        """Test LLM enrichment records trace stages."""
        mock_llm = Mock()
        mock_response = ChatResponse(
            content=json.dumps({
                "title": "Test Title",
                "summary": "Test summary.",
                "tags": ["test"]
            }),
            model="test-model",
            usage={"prompt_tokens": 50, "completion_tokens": 20}
        )
        mock_llm.chat.return_value = mock_response
        
        enricher = MetadataEnricher(mock_settings_with_llm, llm=mock_llm)
        
        chunk = Chunk(
            id="test_1",
            text="Some content.",
            metadata={}
        )
        
        result = enricher._llm_enrich(chunk, trace=trace_context)
        
        assert result is not None
        # Verify trace stages were recorded
        stage_names = [s["name"] for s in trace_context.stages]
        assert "llm_enrichment_request" in stage_names
        assert "llm_enrichment_response" in stage_names


# ==================== Fallback Behavior Tests (3) ====================

class TestFallbackBehavior:
    """Tests for LLM fallback to rule-based results."""
    
    def test_fallback_on_llm_failure(self, mock_settings_with_llm, sample_chunks):
        """Test fallback to rule-based when LLM fails."""
        mock_llm = Mock()
        mock_llm.chat.side_effect = Exception("API Error")
        
        enricher = MetadataEnricher(mock_settings_with_llm, llm=mock_llm)
        
        result = enricher.transform(sample_chunks)
        
        # Should still return enriched chunks (rule-based)
        assert len(result) == len(sample_chunks)
        assert all(isinstance(c, Chunk) for c in result)
        # Should have fallback method in metadata
        assert result[0].metadata.get('enrichment', {}).get('method') == 'rule_based_fallback'
        # Should have fallback_reason in the metadata
        enrichment_metadata = result[0].metadata.get('enrichment', {}).get('metadata', {})
        assert enrichment_metadata.get('fallback_reason') == 'llm_failure'
    
    def test_fallback_when_no_llm_configured(self, mock_settings, sample_chunks):
        """Test behavior when LLM is not enabled."""
        enricher = MetadataEnricher(mock_settings, llm=None)
        
        result = enricher.transform(sample_chunks)
        
        assert len(result) == len(sample_chunks)
        # Should use rule-based only (not rule_based_fallback)
        assert result[0].metadata.get('enrichment', {}).get('method') == 'rule_based'
    
    def test_no_fallback_when_llm_succeeds(self, mock_settings_with_llm, sample_chunks):
        """Test no fallback when LLM succeeds."""
        mock_llm = Mock()
        mock_response = ChatResponse(
            content=json.dumps({
                "title": "LLM Title",
                "summary": "LLM summary.",
                "tags": ["llm-tag"]
            }),
            model="test-model"
        )
        mock_llm.chat.return_value = mock_response
        
        enricher = MetadataEnricher(mock_settings_with_llm, llm=mock_llm)
        
        result = enricher.transform(sample_chunks)
        
        assert result[0].metadata.get('enrichment', {}).get('method') == 'llm_enhanced'


# ==================== Configuration Tests (4) ====================

class TestConfiguration:
    """Tests for configuration handling."""
    
    def test_disabled_enricher_returns_unchanged(self, mock_settings_disabled, sample_chunks):
        """Test that disabled enricher returns chunks unchanged."""
        enricher = MetadataEnricher(mock_settings_disabled)
        
        result = enricher.transform(sample_chunks)
        
        # Should return same chunks unchanged
        assert result == sample_chunks
        assert result[0].text == sample_chunks[0].text
        # No enrichment metadata should be added
        assert 'enrichment' not in result[0].metadata
    
    def test_custom_prompt_path(self, mock_settings):
        """Test custom prompt path configuration."""
        enricher = MetadataEnricher(mock_settings, prompt_path="custom/path.txt")
        assert enricher.prompt_path == "custom/path.txt"
    
    def test_llm_factory_called_when_needed(self, mock_settings_with_llm):
        """Test that LLM factory is called when LLM is needed."""
        with patch('ingestion.transform.metadata_enricher.LLMFactory') as mock_factory:
            mock_llm = Mock()
            mock_factory.create.return_value = mock_llm
            
            enricher = MetadataEnricher(mock_settings_with_llm)
            
            mock_factory.create.assert_called_once_with(mock_settings_with_llm)
            assert enricher._llm is mock_llm
    
    def test_llm_factory_failure_graceful(self, mock_settings_with_llm):
        """Test graceful handling when LLM factory fails."""
        with patch('ingestion.transform.metadata_enricher.LLMFactory') as mock_factory:
            mock_factory.create.side_effect = Exception("Factory error")
            
            enricher = MetadataEnricher(mock_settings_with_llm)
            
            # Should continue without LLM
            assert enricher.use_llm is True  # Config says use_llm
            assert enricher._llm is None  # But no LLM available


# ==================== Output Format Tests (3) ====================

class TestOutputFormat:
    """Tests for output metadata format compliance."""
    
    def test_output_has_required_fields(self, mock_settings, sample_chunks):
        """Test that output metadata has all required fields."""
        enricher = MetadataEnricher(mock_settings)
        result = enricher.transform(sample_chunks)
        
        for chunk in result:
            enrichment = chunk.metadata.get('enrichment', {})
            metadata = enrichment.get('metadata', {})
            
            # Required fields
            assert 'title' in metadata
            assert 'summary' in metadata
            assert 'tags' in metadata
            assert 'enriched_by' in metadata
            assert 'enriched_at' in metadata
    
    def test_tags_is_list(self, mock_settings, sample_chunks):
        """Test that tags field is always a list."""
        enricher = MetadataEnricher(mock_settings)
        result = enricher.transform(sample_chunks)
        
        for chunk in result:
            metadata = chunk.metadata.get('enrichment', {}).get('metadata', {})
            assert isinstance(metadata['tags'], list)
    
    def test_enrichment_method_in_metadata(self, mock_settings, sample_chunks):
        """Test that enrichment method is recorded in metadata."""
        enricher = MetadataEnricher(mock_settings)
        result = enricher.transform(sample_chunks)
        
        for chunk in result:
            enrichment = chunk.metadata.get('enrichment', {})
            assert 'method' in enrichment
            assert enrichment['method'] in ['rule_based', 'llm_enhanced', 'rule_based_fallback']


# ==================== Error Handling Tests (2) ====================

class TestErrorHandling:
    """Tests for error handling scenarios."""
    
    def test_single_chunk_error_does_not_fail_batch(self, mock_settings, sample_chunks):
        """Test that error in one chunk doesn't fail entire batch."""
        enricher = MetadataEnricher(mock_settings)
        
        # Mock _enrich_chunk to fail on first chunk
        original_enrich = enricher._enrich_chunk
        call_count = [0]
        
        def mock_enrich(chunk, trace=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Test error")
            return original_enrich(chunk, trace)
        
        enricher._enrich_chunk = mock_enrich
        
        result = enricher.transform(sample_chunks)
        
        # Should still return all chunks
        assert len(result) == len(sample_chunks)
        # First chunk should have error fallback metadata
        assert result[0].id == sample_chunks[0].id
    
    def test_empty_chunks_list(self, mock_settings):
        """Test handling of empty chunks list."""
        enricher = MetadataEnricher(mock_settings)
        result = enricher.transform([])
        assert result == []


# ==================== Transform Integration Tests ====================

class TestTransformIntegration:
    """Integration tests for the transform method."""
    
    def test_transform_preserves_chunk_structure(self, mock_settings, sample_chunks):
        """Test that transform preserves chunk structure."""
        enricher = MetadataEnricher(mock_settings)
        result = enricher.transform(sample_chunks)
        
        for i, chunk in enumerate(result):
            assert chunk.id == sample_chunks[i].id
            assert chunk.text == sample_chunks[i].text
            assert chunk.start_offset == sample_chunks[i].start_offset
            assert chunk.end_offset == sample_chunks[i].end_offset
            assert chunk.source_ref == sample_chunks[i].source_ref
    
    def test_transform_preserves_existing_metadata(self, mock_settings, sample_chunks):
        """Test that transform preserves existing metadata."""
        enricher = MetadataEnricher(mock_settings)
        result = enricher.transform(sample_chunks)
        
        for i, chunk in enumerate(result):
            # Original metadata should be preserved
            for key, value in sample_chunks[i].metadata.items():
                assert chunk.metadata.get(key) == value
            # Plus enrichment metadata should be added
            assert 'enrichment' in chunk.metadata
    
    def test_transform_with_trace(self, mock_settings, sample_chunks, trace_context):
        """Test transform records trace stages."""
        enricher = MetadataEnricher(mock_settings)
        enricher.transform(sample_chunks, trace=trace_context)
        
        stage_names = [s["name"] for s in trace_context.stages]
        assert "metadata_enrichment" in stage_names
        assert "metadata_enrichment_complete" in stage_names
    
    def test_trace_includes_counts(self, mock_settings, sample_chunks, trace_context):
        """Test that trace includes input/output counts."""
        enricher = MetadataEnricher(mock_settings)
        enricher.transform(sample_chunks, trace=trace_context)
        
        complete_stage = next(
            s for s in trace_context.stages if s["name"] == "metadata_enrichment_complete"
        )
        assert complete_stage["details"]["input_count"] == len(sample_chunks)
        assert complete_stage["details"]["output_count"] == len(sample_chunks)


# ==================== LLM with Rule Fallback Data Tests ====================

class TestLLMWithRuleFallbackData:
    """Tests for LLM enhancement with rule-based fallback preserved."""
    
    def test_llm_result_includes_rule_fallback(self, mock_settings_with_llm):
        """Test that LLM result includes rule-based data as fallback."""
        mock_llm = Mock()
        mock_response = ChatResponse(
            content=json.dumps({
                "title": "LLM Title",
                "summary": "LLM summary.",
                "tags": ["llm-tag"]
            }),
            model="test-model"
        )
        mock_llm.chat.return_value = mock_response
        
        enricher = MetadataEnricher(mock_settings_with_llm, llm=mock_llm)
        
        chunk = Chunk(
            id="test_1",
            text="Python Programming Guide\n\nThis is a comprehensive guide to Python programming.",
            metadata={}
        )
        
        result = enricher._enrich_chunk(chunk)
        
        metadata = result.metadata.get('enrichment', {}).get('metadata', {})
        
        # Should have LLM metadata
        assert metadata['enriched_by'] == 'llm'
        assert metadata['title'] == 'LLM Title'
        
        # Should also have rule-based fallback
        assert 'rule_based_fallback' in metadata
        assert 'title' in metadata['rule_based_fallback']
        assert 'summary' in metadata['rule_based_fallback']
        assert 'tags' in metadata['rule_based_fallback']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
