#!/usr/bin/env python3
"""
Embedding Factory - Provider-Agnostic Embedding Creation

Creates appropriate embedding instances based on configuration.

Provider options:
- "litellm": Recommended - Unified interface via LiteLLM
- "openai": Legacy - Direct OpenAI API implementation
- "dashscope": Legacy - Direct DashScope API implementation
"""

from typing import Optional, Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from .base_embedding import BaseEmbedding


class EmbeddingFactory:
    """
    Factory for creating embedding instances.
    
    Routes to appropriate provider implementation based on settings.
    
    Recommended: Use "litellm" provider for unified access to all embedding models.
    Legacy: Use "openai", "dashscope" for direct API implementations.
    """
    
    @staticmethod
    def create(settings: Settings) -> BaseEmbedding:
        """
        Create an embedding instance based on settings.
        
        Args:
            settings: Application settings
            
        Returns:
            Configured embedding instance
            
        Raises:
            ValueError: If provider is unknown
        """
        provider = settings.embedding.provider
        
        # Recommended: LiteLLM unified provider
        if provider == "litellm":
            from .litellm_embedding import LiteLLMEmbedding
            config = settings.embedding.litellm
            return LiteLLMEmbedding(config)
        
        # Legacy: Direct provider implementations
        elif provider == "openai":
            from .openai_embedding import OpenAIEmbedding
            config = getattr(settings.embedding, "openai", {})
            return OpenAIEmbedding(config)
        elif provider == "dashscope":
            from .dashscope_embedding import DashScopeEmbedding
            config = getattr(settings.embedding, "dashscope", {})
            return DashScopeEmbedding(config)
        elif provider == "anthropic":
            raise ValueError(
                "Anthropic does not provide embedding services. "
                "Please use 'litellm', 'openai' or 'dashscope' as the embedding provider."
            )
        else:
            raise ValueError(
                f"Unknown embedding provider: {provider}. "
                f"Supported: litellm, openai, dashscope"
            )


class EmbeddingProvider:
    """
    Wrapper for embedding provider with common functionality.
    
    Provides batching and trace recording.
    """
    
    def __init__(self, embedding: BaseEmbedding) -> None:
        self.embedding = embedding
    
    def embed_with_trace(
        self,
        texts: list[str],
        trace: Optional[TraceContext] = None
    ) -> list[list[float]]:
        """
        Embed with automatic trace recording.
        
        Args:
            texts: Texts to embed
            trace: Optional trace context
            
        Returns:
            Embedding vectors
        """
        if trace:
            trace.record_stage(
                name="embedding",
                provider=self.embedding.__class__.__name__,
                details={
                    "model": self.embedding.model,
                    "dimensions": self.embedding.dimensions,
                    "batch_size": self.embedding.batch_size,
                    "text_count": len(texts)
                }
            )
        
        return self.embedding.embed(texts, trace=trace)
    
    def embed_batches(
        self,
        texts: list[str],
        trace: Optional[TraceContext] = None
    ) -> list[list[float]]:
        """
        Embed texts in batches based on configured batch_size.
        
        Args:
            texts: Texts to embed
            trace: Optional trace context
            
        Returns:
            Embedding vectors
        """
        if not texts:
            return []
        
        batch_size = self.embedding.batch_size
        if batch_size <= 0:
            batch_size = len(texts)
        
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.embed_with_trace(batch, trace)
            results.extend(batch_results)
        
        return results
