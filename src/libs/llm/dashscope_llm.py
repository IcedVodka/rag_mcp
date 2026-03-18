#!/usr/bin/env python3
"""
DashScope LLM - Alibaba Cloud DashScope API Implementation

Supports Qwen series models via DashScope's OpenAI-compatible API.
"""

import httpx
from typing import Optional, Any

from core.trace.trace_context import TraceContext
from .base_llm import BaseLLM, ChatMessage, ChatResponse, LLMError, LLMTimeoutError, LLMAuthenticationError


class DashScopeLLM(BaseLLM):
    """LLM implementation for Alibaba Cloud DashScope."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.timeout = config.get("timeout", 60)
        self.max_retries = config.get("max_retries", 3)
        
        if not self.api_key:
            raise LLMError("DashScope API key is required")
    
    def _prepare_messages(self, messages: list[ChatMessage]) -> list[dict]:
        """Convert ChatMessage to OpenAI format."""
        return [{"role": m.role, "content": m.content} for m in messages]
    
    def _make_request(self, messages: list[dict], temperature: Optional[float], max_tokens: Optional[int]) -> dict:
        """Make HTTP request to DashScope API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages
        }
        
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"DashScope API timeout: {e}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise LLMAuthenticationError(f"DashScope API authentication failed: {e}")
            raise LLMError(f"DashScope API error: {e}")
        except Exception as e:
            raise LLMError(f"DashScope request failed: {e}")
    
    def chat(
        self,
        messages: list[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        trace: Optional[TraceContext] = None
    ) -> ChatResponse:
        """Send chat completion request."""
        if trace:
            trace.record_stage(
                name="llm_chat",
                provider="dashscope",
                details={"model": self.model, "message_count": len(messages)}
            )
        
        dashscope_messages = self._prepare_messages(messages)
        response_data = self._make_request(dashscope_messages, temperature, max_tokens)
        
        return ChatResponse(
            content=response_data["choices"][0]["message"]["content"],
            model=response_data.get("model"),
            usage=response_data.get("usage"),
            raw_response=response_data
        )
    
    async def achat(
        self,
        messages: list[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        trace: Optional[TraceContext] = None
    ) -> ChatResponse:
        """Async chat completion."""
        import asyncio
        return await asyncio.to_thread(self.chat, messages, temperature, max_tokens, trace)
