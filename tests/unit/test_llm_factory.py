#!/usr/bin/env python3
"""
LLM Factory Tests

Tests for LLM factory routing and provider creation.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add src to path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from core.settings import Settings, LLMSettings, VisionLLMSettings
from libs.llm.llm_factory import LLMFactory
from libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse


class FakeLLM(BaseLLM):
    """Fake LLM for testing."""
    
    def chat(self, messages, temperature=None, max_tokens=None, trace=None):
        return ChatResponse(content="fake response", model="fake-model")
    
    async def achat(self, messages, temperature=None, max_tokens=None, trace=None):
        return ChatResponse(content="fake async response", model="fake-model")


class TestLLMFactory:
    """Test LLM factory routing logic."""
    
    def test_factory_routes_to_openai(self):
        """Test factory routes to OpenAI provider."""
        settings = Mock(spec=Settings)
        settings.llm = Mock(spec=LLMSettings)
        settings.llm.provider = "openai"
        settings.llm.openai = {"api_key": "test", "model": "gpt-4"}
        
        # Mock the module-level import
        with patch.dict('sys.modules', {'libs.llm.openai_llm': Mock(OpenAILLM=FakeLLM)}):
            llm = LLMFactory.create(settings)
            assert isinstance(llm, FakeLLM)
    
    def test_factory_routes_to_dashscope(self):
        """Test factory routes to DashScope provider."""
        settings = Mock(spec=Settings)
        settings.llm = Mock(spec=LLMSettings)
        settings.llm.provider = "dashscope"
        settings.llm.dashscope = {"api_key": "test", "model": "qwen-max"}
        
        with patch.dict('sys.modules', {'libs.llm.dashscope_llm': Mock(DashScopeLLM=FakeLLM)}):
            llm = LLMFactory.create(settings)
            assert isinstance(llm, FakeLLM)
    
    def test_factory_routes_to_anthropic(self):
        """Test factory routes to Anthropic provider."""
        settings = Mock(spec=Settings)
        settings.llm = Mock(spec=LLMSettings)
        settings.llm.provider = "anthropic"
        settings.llm.anthropic = {"api_key": "test", "model": "claude-3"}
        
        with patch.dict('sys.modules', {'libs.llm.anthropic_llm': Mock(AnthropicLLM=FakeLLM)}):
            llm = LLMFactory.create(settings)
            assert isinstance(llm, FakeLLM)
    
    def test_factory_routes_to_litellm(self):
        """Test factory routes to LiteLLM provider."""
        settings = Mock(spec=Settings)
        settings.llm = Mock(spec=LLMSettings)
        settings.llm.provider = "litellm"
        settings.llm.litellm = {"api_key": "test", "model": "gpt-4o"}
        
        with patch.dict('sys.modules', {'libs.llm.litellm_llm': Mock(LiteLLMLLM=FakeLLM)}):
            llm = LLMFactory.create(settings)
            assert isinstance(llm, FakeLLM)
    
    def test_factory_raises_on_unknown_provider(self):
        """Test factory raises error for unknown provider."""
        settings = Mock(spec=Settings)
        settings.llm = Mock(spec=LLMSettings)
        settings.llm.provider = "unknown_provider"
        
        with pytest.raises(ValueError) as exc_info:
            LLMFactory.create(settings)
        
        assert "unknown_provider" in str(exc_info.value).lower()
    
    def test_vision_llm_factory_returns_none_when_disabled(self):
        """Test vision LLM factory returns None when not enabled."""
        settings = Mock(spec=Settings)
        settings.vision_llm = Mock(spec=VisionLLMSettings)
        settings.vision_llm.enabled = False
        
        result = LLMFactory.create_vision_llm(settings)
        assert result is None
    
    def test_vision_llm_factory_returns_none_when_not_configured(self):
        """Test vision LLM factory returns None when vision_llm is None."""
        settings = Mock(spec=Settings)
        settings.vision_llm = None
        
        result = LLMFactory.create_vision_llm(settings)
        assert result is None


class TestFakeLLMBehavior:
    """Test Fake LLM behavior for unit testing."""
    
    def test_fake_llm_returns_response(self):
        """Test fake LLM returns a valid response."""
        llm = FakeLLM({"model": "fake-model"})
        
        messages = [ChatMessage(role="user", content="Hello")]
        response = llm.chat(messages)
        
        assert isinstance(response, ChatResponse)
        assert response.content == "fake response"
        assert response.model == "fake-model"
    
    def test_fake_llm_async_returns_response(self):
        """Test fake LLM async method returns a valid response."""
        llm = FakeLLM({"model": "fake-model"})
        
        messages = [ChatMessage(role="user", content="Hello")]
        
        import asyncio
        response = asyncio.run(llm.achat(messages))
        
        assert isinstance(response, ChatResponse)
        assert response.content == "fake async response"
