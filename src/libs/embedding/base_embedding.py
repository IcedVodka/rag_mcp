#!/usr/bin/env python3
"""
Base Embedding - Abstract Interface for Text Embeddings

Defines the contract for embedding model implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any

from core.trace.trace_context import TraceContext


class BaseEmbedding(ABC):
    """
    Abstract base class for embedding implementations.
    
    All embedding providers (OpenAI, DashScope) must implement this interface.
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the embedding model with configuration.
        
        Args:
            config: Provider-specific configuration (api_key, base_url, model, etc.)
        """
        self.config = config
        self.model = config.get("model", "")
        self.dimensions = config.get("dimensions", 0)
        self.batch_size = config.get("batch_size", 100)
    
    @abstractmethod
    def embed(
        self,
        texts: list[str],
        trace: Optional[TraceContext] = None
    ) -> list[list[float]]:
        """
        Embed a list of texts into vectors.
        
        Args:
            texts: List of text strings to embed
            trace: Optional trace context for observability
            
        Returns:
            List of embedding vectors (list of floats)
            
        Raises:
            EmbeddingError: On API errors or invalid input
        """
        pass
    
    @abstractmethod
    async def aembed(
        self,
        texts: list[str],
        trace: Optional[TraceContext] = None
    ) -> list[list[float]]:
        """Async version of embed."""
        pass
    
    def embed_single(
        self,
        text: str,
        trace: Optional[TraceContext] = None
    ) -> list[float]:
        """
        Embed a single text.
        
        Convenience wrapper around embed().
        
        Args:
            text: Text string to embed
            trace: Optional trace context
            
        Returns:
            Embedding vector
        """
        results = self.embed([text], trace=trace)
        return results[0] if results else []


class EmbeddingError(Exception):
    """Base exception for embedding errors."""
    pass


class EmbeddingTimeoutError(EmbeddingError):
    """Raised when embedding request times out."""
    pass


class EmbeddingAuthenticationError(EmbeddingError):
    """Raised when embedding authentication fails."""
    pass


class EmbeddingRateLimitError(EmbeddingError):
    """Raised when embedding rate limit is hit."""
    pass
