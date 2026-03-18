#!/usr/bin/env python3
"""
Integration Tests for ChunkRefiner with Real LLM

Tests the ChunkRefiner with actual LLM calls to verify:
- Real LLM calls succeed
- Output quality meets expectations
- Fallback mechanism works with real failures

Requirements:
- Valid LLM configuration in config/settings.yaml
- Corresponding API key environment variable set

Set environment variable to skip:
- SKIP_LLM_TESTS=1 to skip these tests
"""

import os
import sys
import pytest
from pathlib import Path

# Adjust path for imports
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from core.types import Chunk
from core.settings import load_settings
from core.trace.trace_context import TraceContext
from ingestion.transform.chunk_refiner import ChunkRefiner
from libs.llm.llm_factory import LLMFactory
from libs.llm.base_llm import ChatMessage


# Skip all tests in this module if SKIP_LLM_TESTS is set
pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_LLM_TESTS") == "1",
    reason="LLM tests disabled via SKIP_LLM_TESTS environment variable"
)


@pytest.fixture(scope="module")
def settings():
    """Load real application settings."""
    config_path = Path(__file__).parents[2] / "config" / "settings.yaml"
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")
    try:
        return load_settings(config_path)
    except Exception as e:
        pytest.skip(f"Failed to load settings: {e}")


@pytest.fixture(scope="module")
def real_llm(settings):
    """Create real LLM instance."""
    try:
        llm = LLMFactory.create(settings)
        # Test the connection
        test_response = llm.chat([ChatMessage(role="user", content="Hi")])
        if not test_response.content:
            pytest.skip("LLM returned empty response - check API key")
        return llm
    except Exception as e:
        pytest.skip(f"Failed to create LLM: {e}")


@pytest.fixture
def trace_context():
    """Create a trace context for testing."""
    return TraceContext(trace_type="ingestion")


class TestRealLLMRefinement:
    """Integration tests with real LLM calls."""
    
    def test_real_llm_basic_refinement(self, settings, real_llm, trace_context):
        """Test basic refinement with real LLM."""
        # Create settings with LLM enabled
        settings.ingestion.chunk_refiner = {
            'enabled': True,
            'use_llm': True,
            'prompt_path': 'config/prompts/chunk_refinement.txt'
        }
        
        refiner = ChunkRefiner(settings, llm=real_llm)
        
        # Load prompt template
        assert refiner._load_prompt() is not None, "Prompt template should load"
        
        chunks = [
            Chunk(
                id="test_1",
                text="Page 1 of 10\n\nIntroduction to AI.\n\nCopyright 2024",
                metadata={}
            )
        ]
        
        result = refiner.transform(chunks, trace=trace_context)
        
        # Verify transformation occurred
        assert len(result) == 1
        assert result[0].metadata.get('refinement', {}).get('method') == 'llm_enhanced'
        
        # Verify trace was recorded
        stage_names = [s["name"] for s in trace_context.stages]
        assert "llm_refinement_request" in stage_names
        assert "llm_refinement_response" in stage_names
        
        print(f"\nRefined text: {result[0].text}")
    
    def test_real_llm_ocr_correction(self, settings, real_llm, trace_context):
        """Test LLM can correct OCR errors."""
        settings.ingestion.chunk_refiner = {
            'enabled': True,
            'use_llm': True,
            'prompt_path': 'config/prompts/chunk_refinement.txt'
        }
        
        refiner = ChunkRefiner(settings, llm=real_llm)
        
        chunks = [
            Chunk(
                id="test_ocr",
                text="The 0CR techn0l0gy has impr0ved. Art1f1cial 1ntell1gence is gr0wing.",
                metadata={}
            )
        ]
        
        result = refiner.transform(chunks, trace=trace_context)
        
        # LLM should have attempted to fix obvious OCR errors
        refined_text = result[0].text
        print(f"\nOCR refined text: {refined_text}")
        
        # Verify the text was processed
        assert result[0].metadata.get('refinement', {}).get('method') == 'llm_enhanced'
        assert len(refined_text) > 0
    
    def test_real_llm_preserves_meaning(self, settings, real_llm):
        """Test that LLM refinement preserves original meaning."""
        settings.ingestion.chunk_refiner = {
            'enabled': True,
            'use_llm': True,
            'prompt_path': 'config/prompts/chunk_refinement.txt'
        }
        
        refiner = ChunkRefiner(settings, llm=real_llm)
        
        original_text = (
            "Machine learning is a method of data analysis that automates "
            "analytical model building. It is a branch of artificial intelligence "
            "based on the idea that systems can learn from data."
        )
        
        chunks = [
            Chunk(
                id="test_meaning",
                text=original_text,
                metadata={}
            )
        ]
        
        result = refiner.transform(chunks)
        refined_text = result[0].text
        
        print(f"\nOriginal: {original_text[:100]}...")
        print(f"Refined: {refined_text[:100]}...")
        
        # Key concepts should be preserved
        assert "machine learning" in refined_text.lower() or "Machine Learning" in refined_text
        assert "artificial intelligence" in refined_text.lower() or "AI" in refined_text
        assert len(refined_text) > len(original_text) * 0.5  # Not too short
        assert len(refined_text) < len(original_text) * 2  # Not too long
    
    def test_real_llm_with_code_preservation(self, settings, real_llm):
        """Test that LLM handles code blocks correctly."""
        settings.ingestion.chunk_refiner = {
            'enabled': True,
            'use_llm': True,
            'prompt_path': 'config/prompts/chunk_refinement.txt'
        }
        
        refiner = ChunkRefiner(settings, llm=real_llm)
        
        chunks = [
            Chunk(
                id="test_code",
                text="""Here's a Python example:

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
```

Use it wisely.""",
                metadata={}
            )
        ]
        
        result = refiner.transform(chunks)
        refined_text = result[0].text
        
        print(f"\nCode refined text:\n{refined_text}")
        
        # Code should be preserved (either in block or clearly recognizable)
        assert "factorial" in refined_text
        assert "def" in refined_text or "python" in refined_text.lower()
    
    def test_fallback_on_invalid_prompt(self, settings, real_llm, trace_context):
        """Test fallback when prompt template is invalid."""
        settings.ingestion.chunk_refiner = {
            'enabled': True,
            'use_llm': True,
            'prompt_path': 'nonexistent/path.txt'  # Invalid path
        }
        
        refiner = ChunkRefiner(settings, llm=real_llm)
        
        chunks = [
            Chunk(
                id="test_fallback",
                text="Some text to refine.",
                metadata={}
            )
        ]
        
        result = refiner.transform(chunks, trace=trace_context)
        
        # Should fall back to rule-based
        assert result[0].metadata.get('refinement', {}).get('method') == 'rule_based'
        # Should still be refined (rule-based)
        assert "refinement" in result[0].metadata
    
    def test_refinement_quality_metrics(self, settings, real_llm):
        """Test that refinement produces reasonable quality metrics."""
        settings.ingestion.chunk_refiner = {
            'enabled': True,
            'use_llm': True,
            'prompt_path': 'config/prompts/chunk_refinement.txt'
        }
        
        refiner = ChunkRefiner(settings, llm=real_llm)
        
        noisy_text = """
Confidential - Internal Use Only

Page 5 of 100



Machine Learning Overview



Machine    learning    is    a    subset    of    AI.



<!-- TODO: add more content -->

Copyright 2024 Corp Inc.
"""
        
        chunks = [
            Chunk(
                id="test_quality",
                text=noisy_text,
                metadata={}
            )
        ]
        
        result = refiner.transform(chunks)
        refined = result[0]
        
        refinement_info = refined.metadata.get('refinement', {})
        original_len = refinement_info.get('original_length', 0)
        refined_len = refinement_info.get('refined_length', 0)
        
        print(f"\nOriginal length: {original_len}")
        print(f"Refined length: {refined_len}")
        print(f"Reduction: {(1 - refined_len/original_len)*100:.1f}%")
        
        # Should have reduced noise
        assert refined_len < original_len * 0.8  # At least 20% reduction
        
        # Should preserve key content
        assert "Machine Learning" in refined.text or "machine learning" in refined.text.lower()
        assert "AI" in refined.text or "artificial intelligence" in refined.text.lower()
        
        # Noise should be removed
        assert "Confidential" not in refined.text
        assert "Copyright" not in refined.text
        assert "<!--" not in refined.text


class TestRealLLMFallback:
    """Tests for fallback behavior with real LLM."""
    
    def test_real_llm_timeout_simulation(self, settings, trace_context):
        """Test fallback when LLM times out."""
        # Create a mock LLM that simulates timeout
        from unittest.mock import Mock
        
        mock_llm = Mock()
        mock_llm.chat.side_effect = TimeoutError("Connection timeout")
        
        settings.ingestion.chunk_refiner = {
            'enabled': True,
            'use_llm': True,
            'prompt_path': 'config/prompts/chunk_refinement.txt'
        }
        
        refiner = ChunkRefiner(settings, llm=mock_llm)
        refiner._prompt_template = "Refine: {text}"  # Set directly to avoid file loading
        
        chunks = [
            Chunk(
                id="test_timeout",
                text="Page 1\n\nContent here.\n\nCopyright",
                metadata={}
            )
        ]
        
        result = refiner.transform(chunks, trace=trace_context)
        
        # Should fallback to rule-based
        assert result[0].metadata.get('refinement', {}).get('method') == 'rule_based_fallback'
        
        # Verify fallback was recorded in trace
        stage_names = [s["name"] for s in trace_context.stages]
        assert "llm_refinement_error" in stage_names or "chunk_refinement_fallback" in stage_names
    
    def test_switch_between_modes(self, settings, real_llm):
        """Test switching between rule-only and LLM-enhanced modes."""
        chunk = Chunk(
            id="test_switch",
            text="Confidential\n\nImportant content here.\n\nPage 5",
            metadata={}
        )
        
        # Test rule-only mode
        settings.ingestion.chunk_refiner = {
            'enabled': True,
            'use_llm': False,
            'prompt_path': 'config/prompts/chunk_refinement.txt'
        }
        
        refiner_rule = ChunkRefiner(settings, llm=None)
        result_rule = refiner_rule.transform([chunk])
        assert result_rule[0].metadata.get('refinement', {}).get('method') == 'rule_based'
        
        # Test LLM mode
        settings.ingestion.chunk_refiner = {
            'enabled': True,
            'use_llm': True,
            'prompt_path': 'config/prompts/chunk_refinement.txt'
        }
        
        refiner_llm = ChunkRefiner(settings, llm=real_llm)
        result_llm = refiner_llm.transform([chunk])
        assert result_llm[0].metadata.get('refinement', {}).get('method') == 'llm_enhanced'
        
        print(f"\nRule-based: {result_rule[0].text}")
        print(f"LLM-enhanced: {result_llm[0].text}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
