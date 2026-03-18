#!/usr/bin/env python3
"""
LiteLLM Embedding Adapter - Unified Embedding Interface via LiteLLM

Uses LiteLLM to provide unified access to embedding models from
multiple providers (OpenAI, Azure, Cohere, etc.)
while maintaining the project's BaseEmbedding interface.
"""

import litellm
from typing import Optional, Any

from core.trace.trace_context import TraceContext
from .base_embedding import BaseEmbedding, EmbeddingError, EmbeddingTimeoutError, EmbeddingAuthenticationError


class LiteLLMEmbedding(BaseEmbedding):
    """
    Embedding implementation using LiteLLM for unified provider access.
    
    Supports embedding models from various providers:
    - OpenAI: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
    - Azure: azure/text-embedding-ada-002, etc.
    - Cohere: cohere/embed-english-v3.0, etc.
    - And more via LiteLLM's embedding support
    
    Example config:
        provider: litellm
        litellm:
            model: "text-embedding-3-small"  # or "azure/text-embedding-ada-002"
            api_key: "your-api-key"
            base_url: null  # optional
            dimensions: 1536
            batch_size: 100
            timeout: 30
            max_retries: 3
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize LiteLLM embedding adapter.
        
        Args:
            config: Configuration dict with keys:
                - model: Embedding model identifier
                - api_key: API key for the provider
                - base_url: Optional custom base URL
                - dimensions: Embedding dimensions (for reference)
                - batch_size: Batch size for embedding requests
                - timeout: Request timeout in seconds
                - max_retries: Number of retries on failure
        """
        super().__init__(config)
        self.model = config.get("model", "")
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url")
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        
        if not self.model:
            raise EmbeddingError("Model name is required for LiteLLM embedding provider")
    
    def _handle_error(self, error: Exception) -> EmbeddingError:
        """Convert LiteLLM errors to project-specific errors."""
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            return EmbeddingTimeoutError(f"Embedding request timeout: {error}")
        elif "authentication" in error_str or "api key" in error_str or "401" in error_str:
            return EmbeddingAuthenticationError(f"Embedding authentication failed: {error}")
        else:
            return EmbeddingError(f"Embedding request failed: {error}")
    
    def embed(
        self,
        texts: list[str],
        trace: Optional[TraceContext] = None
    ) -> list[list[float]]:
        """
        Embed texts into vectors using LiteLLM.
        
        Args:
            texts: List of text strings to embed
            trace: Optional trace context for observability
            
        Returns:
            List of embedding vectors (list of floats)
        """
        if not texts:
            return []
        
        if trace:
            trace.record_stage(
                name="embedding",
                provider="litellm",
                details={
                    "model": self.model,
                    "dimensions": self.dimensions,
                    "batch_size": self.batch_size,
                    "text_count": len(texts)
                }
            )
        
        try:
            # Build kwargs dynamically
            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": texts,
                "timeout": self.timeout,
                "num_retries": self.max_retries,
            }
            
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            
            response = litellm.embedding(**kwargs)
            
            # Extract embeddings - LiteLLM returns them in the same order
            embeddings = []
            if hasattr(response, 'data') and response.data:
                # Sort by index to ensure correct order
                indexed_embeddings = []
                for item in response.data:
                    idx = getattr(item, 'index', 0)
                    embedding = getattr(item, 'embedding', [])
                    indexed_embeddings.append((idx, embedding))
                
                # Sort by index and extract embeddings
                indexed_embeddings.sort(key=lambda x: x[0])
                embeddings = [emb[1] for emb in indexed_embeddings]
            
            # Update dimensions from first result if available
            if embeddings and not self.dimensions:
                self.dimensions = len(embeddings[0])
            
            return embeddings
            
        except Exception as e:
            raise self._handle_error(e)
    
    async def aembed(
        self,
        texts: list[str],
        trace: Optional[TraceContext] = None
    ) -> list[list[float]]:
        """
        Async embed texts using LiteLLM.
        
        Uses LiteLLM's native async support for better performance.
        """
        if not texts:
            return []
        
        if trace:
            trace.record_stage(
                name="embedding_async",
                provider="litellm",
                details={
                    "model": self.model,
                    "dimensions": self.dimensions,
                    "batch_size": self.batch_size,
                    "text_count": len(texts)
                }
            )
        
        try:
            # Build kwargs dynamically
            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": texts,
                "timeout": self.timeout,
                "num_retries": self.max_retries,
            }
            
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            
            response = await litellm.aembedding(**kwargs)
            
            # Extract embeddings
            embeddings = []
            if hasattr(response, 'data') and response.data:
                indexed_embeddings = []
                for item in response.data:
                    idx = getattr(item, 'index', 0)
                    embedding = getattr(item, 'embedding', [])
                    indexed_embeddings.append((idx, embedding))
                
                indexed_embeddings.sort(key=lambda x: x[0])
                embeddings = [emb[1] for emb in indexed_embeddings]
            
            if embeddings and not self.dimensions:
                self.dimensions = len(embeddings[0])
            
            return embeddings
            
        except Exception as e:
            raise self._handle_error(e)
