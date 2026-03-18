#!/usr/bin/env python3
"""
Base LLM - Abstract Interface for Language Models

Defines the contract for LLM implementations across different providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any

from core.trace.trace_context import TraceContext


@dataclass
class ChatMessage:
    """A single message in a chat conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ChatResponse:
    """Response from an LLM chat completion."""
    content: str
    model: Optional[str] = None
    usage: Optional[dict[str, int]] = None  # {"prompt_tokens": n, "completion_tokens": m}
    raw_response: Optional[Any] = None  # Provider-specific raw response


class BaseLLM(ABC):
    """
    Abstract base class for LLM implementations.
    
    All LLM providers (OpenAI, DashScope, Anthropic) must implement this interface.
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the LLM with configuration.
        
        Args:
            config: Provider-specific configuration (api_key, base_url, model, etc.)
        """
        self.config = config
        self.model = config.get("model", "")
    
    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        trace: Optional[TraceContext] = None
    ) -> ChatResponse:
        """
        Send a chat completion request to the LLM.
        
        Args:
            messages: List of chat messages
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            trace: Optional trace context for observability
            
        Returns:
            ChatResponse with generated content
            
        Raises:
            LLMError: On API errors or timeouts
        """
        pass
    
    @abstractmethod
    async def achat(
        self,
        messages: list[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        trace: Optional[TraceContext] = None
    ) -> ChatResponse:
        """Async version of chat."""
        pass


class LLMError(Exception):
    """Base exception for LLM errors."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out."""
    pass


class LLMAuthenticationError(LLMError):
    """Raised when LLM authentication fails."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when LLM rate limit is hit."""
    pass
