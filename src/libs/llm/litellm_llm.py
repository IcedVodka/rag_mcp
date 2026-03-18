#!/usr/bin/env python3
"""
LiteLLM LLM Adapter - Unified LLM Interface via LiteLLM

Uses LiteLLM to provide unified access to 100+ LLM providers
(OpenAI, Anthropic, Gemini, Azure, Groq, Ollama, etc.)
while maintaining the project's BaseLLM interface.

LiteLLM handles:
- Provider-specific API format conversions
- Error handling and retries
- Streaming support
- Function calling
"""

import litellm
from typing import Optional, Any

from core.trace.trace_context import TraceContext
from .base_llm import BaseLLM, ChatMessage, ChatResponse, LLMError, LLMTimeoutError, LLMAuthenticationError

# Configure litellm to not print logs by default
litellm.set_verbose = False


class LiteLLMError(LLMError):
    """Wrapper for LiteLLM-specific errors."""
    pass


class LiteLLMLLM(BaseLLM):
    """
    LLM implementation using LiteLLM for unified provider access.
    
    Supports all providers that LiteLLM supports:
    - OpenAI (openai/gpt-4, openai/gpt-4o-mini, etc.)
    - Anthropic (anthropic/claude-3-sonnet-20240229, etc.)
    - Google (gemini/gemini-pro, vertex_ai/gemini-pro, etc.)
    - Azure (azure/gpt-4, etc.)
    - And 100+ more...
    
    The model format is: "provider/model_name" or just "model_name" for OpenAI
    
    Example config:
        provider: litellm
        litellm:
            model: "claude-3-5-sonnet-20241022"  # or "gpt-4o", "gemini/gemini-pro"
            api_key: "your-api-key"
            base_url: null  # optional, for custom endpoints
            timeout: 60
            max_retries: 3
            temperature: 0.7
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize LiteLLM adapter.
        
        Args:
            config: Configuration dict with keys:
                - model: Model identifier (e.g., "gpt-4o", "claude-3-sonnet", "gemini/gemini-pro")
                - api_key: API key for the provider
                - base_url: Optional custom base URL
                - timeout: Request timeout in seconds
                - max_retries: Number of retries on failure
                - temperature: Default temperature (optional)
                - max_tokens: Default max tokens (optional)
        """
        super().__init__(config)
        self.model = config.get("model", "")
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url")
        self.timeout = config.get("timeout", 60)
        self.max_retries = config.get("max_retries", 3)
        self.default_temperature = config.get("temperature")
        self.default_max_tokens = config.get("max_tokens")
        
        if not self.model:
            raise LLMError("Model name is required for LiteLLM provider")
        
        # LiteLLM can infer provider from model name for common cases,
        # but we recommend explicit format: "provider/model"
        # If no provider prefix, assume openai
        if "/" not in self.model and not self.model.startswith("gpt-"):
            # For models without provider prefix, we keep as-is
            # LiteLLM will try to route them
            pass
    
    def _prepare_messages(self, messages: list[ChatMessage]) -> list[dict]:
        """Convert ChatMessage to LiteLLM format."""
        return [{"role": m.role, "content": m.content} for m in messages]
    
    def _handle_error(self, error: Exception) -> LLMError:
        """Convert LiteLLM errors to project-specific errors."""
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            return LLMTimeoutError(f"LLM request timeout: {error}")
        elif "authentication" in error_str or "api key" in error_str or "401" in error_str:
            return LLMAuthenticationError(f"LLM authentication failed: {error}")
        else:
            return LLMError(f"LLM request failed: {error}")
    
    def chat(
        self,
        messages: list[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        trace: Optional[TraceContext] = None
    ) -> ChatResponse:
        """
        Send chat completion request via LiteLLM.
        
        Args:
            messages: List of chat messages
            temperature: Sampling temperature (0-2), overrides default if provided
            max_tokens: Maximum tokens to generate, overrides default if provided
            trace: Optional trace context for observability
            
        Returns:
            ChatResponse with generated content
        """
        if trace:
            trace.record_stage(
                name="llm_chat",
                provider="litellm",
                details={
                    "model": self.model,
                    "message_count": len(messages)
                }
            )
        
        # Use provided values or fall back to defaults
        temp = temperature if temperature is not None else self.default_temperature
        tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        
        litellm_messages = self._prepare_messages(messages)
        
        try:
            # Build kwargs dynamically
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": litellm_messages,
                "timeout": self.timeout,
                "num_retries": self.max_retries,
            }
            
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            if temp is not None:
                kwargs["temperature"] = temp
            if tokens is not None:
                kwargs["max_tokens"] = tokens
            
            response = litellm.completion(**kwargs)
            
            # Extract usage info if available
            usage = None
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0)
                }
            
            return ChatResponse(
                content=response.choices[0].message.content,
                model=getattr(response, 'model', self.model),
                usage=usage,
                raw_response=response
            )
            
        except Exception as e:
            raise self._handle_error(e)
    
    async def achat(
        self,
        messages: list[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        trace: Optional[TraceContext] = None
    ) -> ChatResponse:
        """
        Async chat completion via LiteLLM.
        
        Uses LiteLLM's native async support for better performance.
        """
        if trace:
            trace.record_stage(
                name="llm_chat_async",
                provider="litellm",
                details={
                    "model": self.model,
                    "message_count": len(messages)
                }
            )
        
        # Use provided values or fall back to defaults
        temp = temperature if temperature is not None else self.default_temperature
        tokens = max_tokens if max_tokens is not None else self.default_max_tokens
        
        litellm_messages = self._prepare_messages(messages)
        
        try:
            # Build kwargs dynamically
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": litellm_messages,
                "timeout": self.timeout,
                "num_retries": self.max_retries,
            }
            
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            if temp is not None:
                kwargs["temperature"] = temp
            if tokens is not None:
                kwargs["max_tokens"] = tokens
            
            response = await litellm.acompletion(**kwargs)
            
            # Extract usage info if available
            usage = None
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                    "total_tokens": getattr(response.usage, 'total_tokens', 0)
                }
            
            return ChatResponse(
                content=response.choices[0].message.content,
                model=getattr(response, 'model', self.model),
                usage=usage,
                raw_response=response
            )
            
        except Exception as e:
            raise self._handle_error(e)
