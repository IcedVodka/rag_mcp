#!/usr/bin/env python3
"""
Anthropic Vision LLM - Anthropic Claude Vision Implementation

Supports Claude-3 series vision capabilities.
"""

import base64
from typing import Optional, Any, Union
from pathlib import Path

from core.trace.trace_context import TraceContext
from .base_vision_llm import BaseVisionLLM, VisionResponse, VisionLLMError, VisionLLMTimeoutError


class AnthropicVisionLLM(BaseVisionLLM):
    """Vision LLM implementation for Anthropic Claude."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://api.anthropic.com/v1")
        self.timeout = config.get("timeout", 60)
        
        if not self.api_key:
            raise VisionLLMError("Anthropic API key is required")
    
    def _prepare_image_content(
        self,
        image_path: Optional[Union[str, Path]],
        image_base64: Optional[str]
    ) -> dict:
        """Prepare image content for Anthropic API."""
        if image_base64:
            image_data = image_base64
        elif image_path:
            image_bytes = self._resize_image_if_needed(image_path)
            image_data = base64.b64encode(image_bytes).decode('utf-8')
        else:
            raise VisionLLMError("Either image_path or image_base64 must be provided")
        
        # Determine media type
        media_type = "image/jpeg"
        if image_path and str(image_path).lower().endswith('.png'):
            media_type = "image/png"
        
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data
            }
        }
    
    def chat_with_image(
        self,
        text: str,
        image_path: Optional[Union[str, Path]] = None,
        image_base64: Optional[str] = None,
        trace: Optional[TraceContext] = None
    ) -> VisionResponse:
        """Send vision request to Anthropic."""
        import httpx
        
        if trace:
            trace.record_stage(
                name="vision_llm",
                provider="anthropic",
                details={"model": self.model}
            )
        
        # Prepare message content
        image_content = self._prepare_image_content(image_path, image_base64)
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": text}
                ]
            }],
            "max_tokens": 4096
        }
        
        try:
            response = httpx.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract usage
            usage = None
            if "usage" in data:
                usage = {
                    "prompt_tokens": data["usage"].get("input_tokens", 0),
                    "completion_tokens": data["usage"].get("output_tokens", 0)
                }
            
            return VisionResponse(
                content=data["content"][0]["text"],
                model=data.get("model"),
                usage=usage,
                raw_response=data
            )
        except httpx.TimeoutException as e:
            raise VisionLLMTimeoutError(f"Anthropic vision API timeout: {e}")
        except Exception as e:
            raise VisionLLMError(f"Anthropic vision request failed: {e}")
    
    async def achat_with_image(
        self,
        text: str,
        image_path: Optional[Union[str, Path]] = None,
        image_base64: Optional[str] = None,
        trace: Optional[TraceContext] = None
    ) -> VisionResponse:
        """Async vision chat."""
        import asyncio
        return await asyncio.to_thread(
            self.chat_with_image, text, image_path, image_base64, trace
        )
