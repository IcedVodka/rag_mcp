#!/usr/bin/env python3
"""
Base Vision LLM - Abstract Interface for Vision Language Models

Defines the contract for multimodal LLM implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any, Union
from pathlib import Path

from core.trace.trace_context import TraceContext


@dataclass
class VisionMessage:
    """A message that can include images."""
    role: str  # "system", "user", "assistant"
    content: str
    image_path: Optional[Union[str, Path]] = None
    image_base64: Optional[str] = None


@dataclass
class VisionResponse:
    """Response from a vision LLM."""
    content: str
    model: Optional[str] = None
    usage: Optional[dict[str, int]] = None
    raw_response: Optional[Any] = None


class BaseVisionLLM(ABC):
    """
    Abstract base class for vision LLM implementations.
    
    All vision providers (OpenAI, DashScope, Anthropic) must implement this interface.
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the vision LLM with configuration.
        
        Args:
            config: Provider-specific configuration
        """
        self.config = config
        self.model = config.get("model", "")
        self.max_image_size = config.get("max_image_size", 2048)
    
    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image_path: Optional[Union[str, Path]] = None,
        image_base64: Optional[str] = None,
        trace: Optional[TraceContext] = None
    ) -> VisionResponse:
        """
        Send a chat request with image input.
        
        Args:
            text: Text prompt
            image_path: Path to image file (optional)
            image_base64: Base64-encoded image (optional)
            trace: Optional trace context
            
        Returns:
            VisionResponse with generated content
        """
        pass
    
    @abstractmethod
    async def achat_with_image(
        self,
        text: str,
        image_path: Optional[Union[str, Path]] = None,
        image_base64: Optional[str] = None,
        trace: Optional[TraceContext] = None
    ) -> VisionResponse:
        """Async version of chat_with_image."""
        pass
    
    def _resize_image_if_needed(self, image_path: Union[str, Path]) -> bytes:
        """
        Resize image if it exceeds max dimensions.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image bytes (possibly resized)
        """
        try:
            from PIL import Image
            import io
            
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Resize if too large
                width, height = img.size
                max_size = self.max_image_size
                
                if width > max_size or height > max_size:
                    ratio = min(max_size / width, max_size / height)
                    new_size = (int(width * ratio), int(height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save to bytes
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                return buffer.getvalue()
        except ImportError:
            # PIL not available, return original
            return Path(image_path).read_bytes()
        except Exception:
            # On error, return original
            return Path(image_path).read_bytes()


class VisionLLMError(Exception):
    """Base exception for vision LLM errors."""
    pass


class VisionLLMTimeoutError(VisionLLMError):
    """Raised when vision LLM request times out."""
    pass
