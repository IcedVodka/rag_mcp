#!/usr/bin/env python3
"""
LLM Factory - Provider-Agnostic LLM Creation

Creates appropriate LLM instances based on configuration.

Provider options:
- "litellm": Recommended - Unified interface via LiteLLM (supports 100+ providers)
- "openai": Legacy - Direct OpenAI API implementation
- "dashscope": Legacy - Direct DashScope API implementation  
- "anthropic": Legacy - Direct Anthropic API implementation
"""

from typing import Optional, Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from .base_llm import BaseLLM


class LLMFactory:
    """
    Factory for creating LLM instances.
    
    Routes to appropriate provider implementation based on settings.
    
    Recommended: Use "litellm" provider for unified access to all models.
    Legacy: Use "openai", "dashscope", "anthropic" for direct API implementations.
    """
    
    @staticmethod
    def create(settings: Settings) -> BaseLLM:
        """
        Create an LLM instance based on settings.
        
        Args:
            settings: Application settings
            
        Returns:
            Configured LLM instance
            
        Raises:
            ValueError: If provider is unknown
        """
        provider = settings.llm.provider
        
        # Recommended: LiteLLM unified provider
        if provider == "litellm":
            from .litellm_llm import LiteLLMLLM
            config = settings.llm.litellm
            return LiteLLMLLM(config)
        
        # Legacy: Direct provider implementations
        elif provider == "openai":
            from .openai_llm import OpenAILLM
            config = getattr(settings.llm, "openai", {})
            return OpenAILLM(config)
        elif provider == "dashscope":
            from .dashscope_llm import DashScopeLLM
            config = getattr(settings.llm, "dashscope", {})
            return DashScopeLLM(config)
        elif provider == "anthropic":
            from .anthropic_llm import AnthropicLLM
            config = getattr(settings.llm, "anthropic", {})
            return AnthropicLLM(config)
        else:
            raise ValueError(
                f"Unknown LLM provider: {provider}. "
                f"Supported: litellm, openai, dashscope, anthropic"
            )
    
    @staticmethod
    def create_vision_llm(settings: Settings) -> Optional[Any]:
        """
        Create a Vision LLM instance if configured.
        
        Args:
            settings: Application settings
            
        Returns:
            Vision LLM instance or None if not enabled
        """
        if not settings.vision_llm or not settings.vision_llm.enabled:
            return None
        
        provider = settings.vision_llm.provider
        
        # Recommended: LiteLLM unified provider
        if provider == "litellm":
            from .litellm_llm import LiteLLMLLM
            config = settings.vision_llm.litellm
            return LiteLLMLLM(config)
        
        # Legacy: Direct provider implementations
        elif provider == "openai":
            from .openai_vision_llm import OpenAIVisionLLM
            config = getattr(settings.vision_llm, "openai", {})
            return OpenAIVisionLLM(config)
        elif provider == "dashscope":
            from .dashscope_vision_llm import DashScopeVisionLLM
            config = getattr(settings.vision_llm, "dashscope", {})
            return DashScopeVisionLLM(config)
        elif provider == "anthropic":
            from .anthropic_vision_llm import AnthropicVisionLLM
            config = getattr(settings.vision_llm, "anthropic", {})
            return AnthropicVisionLLM(config)
        else:
            raise ValueError(
                f"Unknown Vision LLM provider: {provider}. "
                f"Supported: litellm, openai, dashscope, anthropic"
            )


class LLMProvider:
    """
    Wrapper for LLM provider with common functionality.
    
    Provides a consistent interface for trace recording and error handling.
    """
    
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm
    
    def chat_with_trace(
        self,
        messages: list[Any],
        trace: Optional[TraceContext] = None,
        **kwargs
    ) -> Any:
        """
        Chat with automatic trace recording.
        
        Args:
            messages: Chat messages
            trace: Optional trace context
            **kwargs: Additional arguments for chat
            
        Returns:
            Chat response
        """
        if trace:
            trace.record_stage(
                name="llm_chat",
                provider=self.llm.__class__.__name__,
                details={"model": self.llm.model}
            )
        
        return self.llm.chat(messages, trace=trace, **kwargs)
