#!/usr/bin/env python3
"""
DashScope Embedding - Alibaba Cloud DashScope Embedding Implementation

Supports Qwen text-embedding models via DashScope API.
"""

import httpx
from typing import Optional, Any

from core.trace.trace_context import TraceContext
from .base_embedding import BaseEmbedding, EmbeddingError, EmbeddingTimeoutError, EmbeddingAuthenticationError


class DashScopeEmbedding(BaseEmbedding):
    """Embedding implementation for Alibaba Cloud DashScope."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.timeout = config.get("timeout", 60)
        self.max_retries = config.get("max_retries", 3)
        
        if not self.api_key:
            raise EmbeddingError("DashScope API key is required")
        
        # DashScope has a batch size limit of 25
        if self.batch_size > 25:
            self.batch_size = 25
    
    def _make_request(self, texts: list[str]) -> dict:
        """Make HTTP request to DashScope embedding API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": texts
        }
        
        # Add dimensions parameter if specified (for text-embedding-v4)
        if self.dimensions > 0:
            payload["dimensions"] = self.dimensions
        
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
            raise EmbeddingTimeoutError(f"DashScope embedding timeout: {e}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise EmbeddingAuthenticationError(f"DashScope API authentication failed: {e}")
            raise EmbeddingError(f"DashScope API error: {e}")
        except Exception as e:
            raise EmbeddingError(f"DashScope embedding request failed: {e}")
    
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
                provider="dashscope",
                details={"model": self.model, "text_count": len(texts)}
            )
        
        # Handle batching for DashScope's 25 limit
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response_data = self._make_request(batch)
            
            # Extract embeddings
            batch_embeddings = []
            for item in response_data["data"]:
                batch_embeddings.append((item["index"], item["embedding"]))
            
            batch_embeddings.sort(key=lambda x: x[0])
            all_embeddings.extend([emb[1] for emb in batch_embeddings])
        
        return all_embeddings
    
    async def aembed(
        self,
        texts: list[str],
        trace: Optional[TraceContext] = None
    ) -> list[list[float]]:
        """Async embed texts."""
        import asyncio
        return await asyncio.to_thread(self.embed, texts, trace)
