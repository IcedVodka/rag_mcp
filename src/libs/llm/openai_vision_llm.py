#!/usr/bin/env python3
"""
OpenAI Vision LLM - OpenAI GPT-4 Vision Implementation

Supports GPT-4o vision capabilities.
"""

import base64
from typing import Optional, Any, Union
from pathlib import Path

from core.trace.trace_context import TraceContext
from .base_vision_llm import BaseVisionLLM, VisionResponse, VisionLLMError


class OpenAIVisionLLM(BaseVisionLLM):
    """Vision LLM implementation for OpenAI GPT-4."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.timeout = config.get("timeout", 30)
        
        if not self.api_key:
            raise VisionLLMError("OpenAI API key is required")
    
    def _prepare_image_content(
        self,
        image_path: Optional[Union[str, Path]],
        image_base64: Optional[str]
    ) -> dict:
        """Prepare image content for OpenAI API."""
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
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{image_data}"
            }
        }
    
    def chat_with_image(
        self,
        text: str,
        image_path: Optional[Union[str, Path]] = None,
        image_base64: Optional[str] = None,
        trace: Optional[TraceContext] = None
    ) -> VisionResponse:
        """Send vision request to OpenAI."""
        import httpx
        
        if trace:
            trace.record_stage(
                name="vision_llm",
                provider="openai",
                details={"model": self.model}
            )
        
        # This is a placeholder implementation
        # Full implementation would make actual API calls
        return VisionResponse(
            content="[OpenAI Vision placeholder - implement actual API call]",
            model=self.model
        )
    
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
