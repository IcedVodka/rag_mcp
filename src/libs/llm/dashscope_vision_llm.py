#!/usr/bin/env python3
"""
DashScope Vision LLM - Alibaba Cloud DashScope Vision Implementation

Supports Qwen-VL vision capabilities.
"""

from typing import Optional, Any, Union
from pathlib import Path

from core.trace.trace_context import TraceContext
from .base_vision_llm import BaseVisionLLM, VisionResponse, VisionLLMError


class DashScopeVisionLLM(BaseVisionLLM):
    """Vision LLM implementation for DashScope Qwen-VL."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.timeout = config.get("timeout", 60)
        
        if not self.api_key:
            raise VisionLLMError("DashScope API key is required")
    
    def chat_with_image(
        self,
        text: str,
        image_path: Optional[Union[str, Path]] = None,
        image_base64: Optional[str] = None,
        trace: Optional[TraceContext] = None
    ) -> VisionResponse:
        """Send vision request to DashScope."""
        if trace:
            trace.record_stage(
                name="vision_llm",
                provider="dashscope",
                details={"model": self.model}
            )
        
        # Placeholder implementation
        return VisionResponse(
            content="[DashScope Vision placeholder - implement actual API call]",
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
