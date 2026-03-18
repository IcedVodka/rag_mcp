#!/usr/bin/env python3
"""
OpenAI Embedding - OpenAI-Compatible Embedding Implementation

Supports OpenAI and other OpenAI-compatible embedding endpoints.
"""

import httpx
from typing import Optional, Any

from core.trace.trace_context import TraceContext
from .base_embedding import BaseEmbedding, EmbeddingError, EmbeddingTimeoutError, EmbeddingAuthenticationError


class OpenAIEmbedding(BaseEmbedding):
    """Embedding implementation for OpenAI-compatible APIs."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        
        if not self.api_key:
            raise EmbeddingError("OpenAI API key is required")
    
    def _make_request(self, texts: list[str]) -> dict:
        """Make HTTP request to OpenAI embedding API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": texts
        }
        
        try:
            response = httpx.post(
                f"{self.base_url}/embeddings",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as e:
            raise EmbeddingTimeoutError(f"OpenAI embedding timeout: {e}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise EmbeddingAuthenticationError(f"OpenAI API authentication failed: {e}")
            raise EmbeddingError(f"OpenAI API error: {e}")
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding request failed: {e}")
    
    def embed(
        self,
        texts: list[str],
        trace: Optional[TraceContext] = None
    ) -> list[list[float]]:
        """Embed texts into vectors."""
        if not texts:
            return []
        
        if trace:
            trace.record_stage(
                name="embedding",
                provider="openai",
                details={"model": self.model, "text_count": len(texts)}
            )
        
        response_data = self._make_request(texts)
        
        # Extract embeddings in the same order as input
        embeddings = []
        for item in response_data["data"]:
            # Sort by index to maintain order
            embeddings.append((item["index"], item["embedding"]))
        
        embeddings.sort(key=lambda x: x[0])
        return [emb[1] for emb in embeddings]
    
    async def aembed(
        self,
        texts: list[str],
        trace: Optional[TraceContext] = None
    ) -> list[list[float]]:
        """Async embed texts."""
        import asyncio
        return await asyncio.to_thread(self.embed, texts, trace)
