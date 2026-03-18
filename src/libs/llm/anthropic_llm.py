#!/usr/bin/env python3
"""
Anthropic LLM - Anthropic Claude API Implementation

Supports Claude series models via Anthropic's native API.
"""

import httpx
from typing import Optional, Any

from core.trace.trace_context import TraceContext
from .base_llm import BaseLLM, ChatMessage, ChatResponse, LLMError, LLMTimeoutError, LLMAuthenticationError


class AnthropicLLM(BaseLLM):
    """LLM implementation for Anthropic Claude."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://api.anthropic.com/v1")
        self.timeout = config.get("timeout", 60)
        self.max_retries = config.get("max_retries", 3)
        
        if not self.api_key:
            raise LLMError("Anthropic API key is required")
    
    def _prepare_messages(self, messages: list[ChatMessage]) -> tuple[Optional[str], list[dict]]:
        """
        Convert ChatMessage to Anthropic format.
        
        Returns:
            Tuple of (system_prompt, messages)
        """
        system_prompt = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        return system_prompt, anthropic_messages
    
    def _make_request(
        self,
        system_prompt: Optional[str],
        messages: list[dict],
        temperature: Optional[float],
        max_tokens: Optional[int]
    ) -> dict:
        """Make HTTP request to Anthropic API."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or 4096
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        if temperature is not None:
            payload["temperature"] = temperature
        
        try:
            response = httpx.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(f"Anthropic API timeout: {e}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise LLMAuthenticationError(f"Anthropic API authentication failed: {e}")
            raise LLMError(f"Anthropic API error: {e}")
        except Exception as e:
            raise LLMError(f"Anthropic request failed: {e}")
    
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
                provider="anthropic",
                details={"model": self.model, "message_count": len(messages)}
            )
        
        system_prompt, anthropic_messages = self._prepare_messages(messages)
        response_data = self._make_request(system_prompt, anthropic_messages, temperature, max_tokens)
        
        # Extract usage info if available
        usage = None
        if "usage" in response_data:
            usage = {
                "prompt_tokens": response_data["usage"].get("input_tokens", 0),
                "completion_tokens": response_data["usage"].get("output_tokens", 0)
            }
        
        return ChatResponse(
            content=response_data["content"][0]["text"],
            model=response_data.get("model"),
            usage=usage,
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
