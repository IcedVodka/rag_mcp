#!/usr/bin/env python3
"""
Image Captioner - Vision LLM-based Image Captioning Transform

Generates descriptive captions for images referenced in chunks using a Vision LLM.
Features graceful degradation when Vision LLM is unavailable or fails.

Caption Output Format (added to chunk.metadata):
{
    "image_captions": Dict[str, str],  # image_id -> caption mapping
    "has_unprocessed_images": bool     # True if some images failed to process
}

Configuration (via settings.ingestion.image_captioner):
- enabled: bool - Whether to enable captioning (default: False)
- prompt_path: str - Path to caption prompt template file

Vision LLM Configuration (via settings.vision_llm):
- enabled: bool - Whether Vision LLM is enabled
- provider: str - LLM provider (litellm, openai, dashscope, anthropic)
"""

import logging
from pathlib import Path
from typing import List, Optional, Any, Dict

from core.types import Chunk
from core.settings import Settings
from core.trace.trace_context import TraceContext
from libs.llm.llm_factory import LLMFactory
from libs.llm.base_vision_llm import BaseVisionLLM, VisionResponse
from .base_transform import BaseTransform

logger = logging.getLogger(__name__)


class ImageCaptioner(BaseTransform):
    """
    Image captioning transform using Vision LLM.
    
    This transform processes chunks that contain image references and generates
    descriptive captions using a Vision-capable LLM. It handles multiple images
    per chunk and gracefully degrades when:
    - Vision LLM is not configured or disabled
    - Vision LLM call fails (timeout, auth error, etc.)
    - Individual image processing fails (other images continue)
    
    The generated captions are stored in chunk.metadata["image_captions"] as a
    dictionary mapping image IDs to their captions. If any images fail to
    process, chunk.metadata["has_unprocessed_images"] is set to True.
    
    Configuration example:
        >>> settings = load_settings("config/settings.yaml")
        >>> captioner = ImageCaptioner(settings)
        >>> captioned_chunks = captioner.transform(chunks, trace)
    
    Attributes:
        enabled: Whether captioning is enabled via config
        vision_llm: Optional Vision LLM instance
        prompt_template: Template for caption generation prompts
    """
    
    def __init__(
        self,
        settings: Settings,
        vision_llm: Optional[BaseVisionLLM] = None,
        prompt_path: Optional[str] = None
    ) -> None:
        """
        Initialize the image captioner.
        
        Args:
            settings: Application settings containing image_captioner and vision_llm config
            vision_llm: Optional pre-configured Vision LLM instance. If not provided,
                       will be created from settings if enabled.
            prompt_path: Optional path to prompt template. Overrides config.
            
        Note:
            If Vision LLM creation fails, the captioner will operate in
            fallback mode (has_unprocessed_images=True, no captions).
        """
        super().__init__()
        
        self.settings = settings
        
        # Get image_captioner config from settings
        captioner_config = getattr(settings.ingestion, 'image_captioner', {})
        self.enabled = captioner_config.get('enabled', False)
        
        # Determine prompt path
        self.prompt_path = prompt_path or captioner_config.get('prompt_path')
        self._prompt_template: Optional[str] = None
        
        # Initialize Vision LLM
        self._vision_llm: Optional[BaseVisionLLM] = vision_llm
        if self.enabled and self._vision_llm is None:
            try:
                self._vision_llm = LLMFactory.create_vision_llm(settings)
                if self._vision_llm:
                    logger.info(f"Created Vision LLM for captioning: {self._vision_llm.model}")
                else:
                    logger.debug("Vision LLM not configured or disabled")
            except Exception as e:
                logger.warning(f"Failed to create Vision LLM for captioning: {e}. "
                              "Will operate in fallback mode.")
                self._vision_llm = None
        
        logger.debug(f"ImageCaptioner initialized: enabled={self.enabled}, "
                    f"vision_llm={'available' if self._vision_llm else 'unavailable'}")
    
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """
        Transform chunks by generating image captions.
        
        Processing flow:
        1. If not enabled, return chunks unchanged
        2. For each chunk with image_refs, generate captions
        3. If Vision LLM unavailable/fails, mark has_unprocessed_images=True
        4. If individual image fails, continue with others
        
        Args:
            chunks: List of chunks to process
            trace: Optional trace context for recording stages
            
        Returns:
            List of chunks with image captions added to metadata
        """
        if not self.enabled:
            logger.debug("ImageCaptioner disabled, returning chunks unchanged")
            return chunks
        
        if not chunks:
            return []
        
        if trace:
            trace.record_stage(
                name="image_captioning",
                method="vision_llm" if self._vision_llm else "fallback",
                provider=self._vision_llm.__class__.__name__ if self._vision_llm else "none"
            )
        
        captioned_chunks: List[Chunk] = []
        total_images = 0
        processed_images = 0
        
        for chunk in chunks:
            try:
                captioned_chunk, stats = self._process_chunk(chunk, trace)
                captioned_chunks.append(captioned_chunk)
                total_images += stats['total']
                processed_images += stats['processed']
            except Exception as e:
                logger.error(f"Failed to process chunk {chunk.id}: {e}")
                # Fall back to original chunk with unprocessed flag
                try:
                    fallback_chunk = self._create_fallback_chunk(chunk)
                    captioned_chunks.append(fallback_chunk)
                    # Count images for stats
                    image_refs = chunk.metadata.get('image_refs', [])
                    if isinstance(image_refs, list):
                        total_images += len(image_refs)
                except Exception:
                    # Last resort: return original
                    captioned_chunks.append(chunk)
        
        if trace:
            trace.record_stage(
                name="image_captioning_complete",
                details={
                    "input_count": len(chunks),
                    "output_count": len(captioned_chunks),
                    "total_images": total_images,
                    "processed_images": processed_images,
                    "failed_images": total_images - processed_images,
                    "vision_llm_available": self._vision_llm is not None
                }
            )
        
        return captioned_chunks
    
    def _process_chunk(
        self,
        chunk: Chunk,
        trace: Optional[TraceContext] = None
    ) -> tuple[Chunk, Dict[str, int]]:
        """
        Process a single chunk to generate image captions.
        
        Args:
            chunk: Chunk to process
            trace: Optional trace context
            
        Returns:
            Tuple of (processed chunk, stats dict with 'total' and 'processed' counts)
        """
        image_refs = chunk.metadata.get('image_refs', [])
        
        # Handle case where image_refs is not a list
        if not isinstance(image_refs, list) or not image_refs:
            # No images to process, return unchanged
            return chunk, {'total': 0, 'processed': 0}
        
        # Prepare image info for processing
        images_to_process = []
        for img_ref in image_refs:
            if isinstance(img_ref, dict):
                img_id = img_ref.get('id')
                img_path = img_ref.get('path')
                if img_id and img_path:
                    images_to_process.append({'id': img_id, 'path': img_path})
            elif isinstance(img_ref, str):
                # Simple string ID - assume path convention
                images_to_process.append({'id': img_ref, 'path': None})
        
        if not images_to_process:
            return chunk, {'total': 0, 'processed': 0}
        
        # Generate captions
        captions: Dict[str, str] = {}
        failed_images: List[str] = []
        
        for img_info in images_to_process:
            caption = self._generate_caption(
                image_path=img_info['path'],
                image_id=img_info['id'],
                trace=trace
            )
            if caption:
                captions[img_info['id']] = caption
            else:
                failed_images.append(img_info['id'])
        
        # Create enriched chunk
        enriched_metadata = chunk.metadata.copy()
        
        if captions:
            enriched_metadata['image_captions'] = captions
        
        # Mark if any images were unprocessed
        if failed_images or not self._vision_llm:
            enriched_metadata['has_unprocessed_images'] = True
        else:
            enriched_metadata['has_unprocessed_images'] = False
        
        processed_chunk = Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=enriched_metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref
        )
        
        stats = {
            'total': len(images_to_process),
            'processed': len(captions)
        }
        
        return processed_chunk, stats
    
    def _create_fallback_chunk(self, chunk: Chunk) -> Chunk:
        """
        Create a fallback chunk when processing fails.
        
        Preserves original image_refs and marks all images as unprocessed.
        
        Args:
            chunk: Original chunk
            
        Returns:
            Chunk with has_unprocessed_images=True
        """
        enriched_metadata = chunk.metadata.copy()
        enriched_metadata['has_unprocessed_images'] = True
        
        return Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=enriched_metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref
        )
    
    def _generate_caption(
        self,
        image_path: Optional[str],
        image_id: str,
        trace: Optional[TraceContext] = None
    ) -> Optional[str]:
        """
        Generate caption for a single image using Vision LLM.
        
        Args:
            image_path: Path to image file (may be None)
            image_id: Unique identifier for the image
            trace: Optional trace context
            
        Returns:
            Generated caption string or None if generation failed
        """
        # Check if Vision LLM is available
        if not self._vision_llm:
            logger.debug(f"No Vision LLM available for image {image_id}")
            return None
        
        # Resolve image path
        resolved_path = self._resolve_image_path(image_path, image_id)
        if not resolved_path or not Path(resolved_path).exists():
            logger.warning(f"Image file not found: {resolved_path or image_id}")
            return None
        
        # Load prompt template
        if self._prompt_template is None:
            self._prompt_template = self._load_prompt()
            if self._prompt_template is None:
                logger.warning("No prompt template available for captioning")
                return None
        
        try:
            if trace:
                trace.record_stage(
                    name="caption_generation_request",
                    provider=self._vision_llm.__class__.__name__,
                    details={
                        "image_id": image_id,
                        "model": self._vision_llm.model
                    }
                )
            
            # Call Vision LLM
            response: VisionResponse = self._vision_llm.chat_with_image(
                text=self._prompt_template,
                image_path=resolved_path,
                trace=trace
            )
            
            if trace:
                trace.record_stage(
                    name="caption_generation_response",
                    details={
                        "image_id": image_id,
                        "prompt_tokens": response.usage.get("prompt_tokens") if response.usage else None,
                        "completion_tokens": response.usage.get("completion_tokens") if response.usage else None
                    }
                )
            
            # Extract and clean caption
            caption = response.content.strip() if response.content else None
            
            if caption:
                logger.debug(f"Generated caption for image {image_id}: {caption[:100]}...")
                return caption
            else:
                logger.warning(f"Empty caption received for image {image_id}")
                return None
                
        except Exception as e:
            logger.warning(f"Caption generation failed for image {image_id}: {e}")
            if trace:
                trace.record_stage(
                    name="caption_generation_error",
                    details={"image_id": image_id, "error": str(e)}
                )
            return None
    
    def _resolve_image_path(self, image_path: Optional[str], image_id: str) -> Optional[str]:
        """
        Resolve image path from reference or construct from convention.
        
        Args:
            image_path: Optional path from image_ref
            image_id: Image identifier
            
        Returns:
            Resolved path or None if cannot be determined
        """
        if image_path and Path(image_path).exists():
            return image_path
        
        # Try to construct from convention: data/images/{collection}/{image_id}.png
        storage = getattr(self.settings, 'storage', None)
        if storage:
            image_dir = getattr(storage, 'image_dir', 'data/images')
        else:
            image_dir = 'data/images'
        
        # Try common image formats
        base_paths = [
            Path(image_dir) / f"{image_id}.png",
            Path(image_dir) / f"{image_id}.jpg",
            Path(image_dir) / f"{image_id}.jpeg",
            Path(image_dir) / 'default' / f"{image_id}.png",
            Path(image_dir) / 'default' / f"{image_id}.jpg",
        ]
        
        for path in base_paths:
            if path.exists():
                return str(path)
        
        # Return original path even if not found (let caller handle)
        return image_path
    
    def _load_prompt(self, prompt_path: Optional[str] = None) -> Optional[str]:
        """
        Load prompt template from file.
        
        Args:
            prompt_path: Path to prompt file. Uses instance prompt_path if not provided.
            
        Returns:
            Prompt template string or default prompt if file not found
        """
        path = prompt_path or self.prompt_path
        
        if not path:
            return self._get_default_prompt()
        
        try:
            full_path = Path(path)
            if not full_path.is_absolute():
                # Try relative to project root
                full_path = Path(__file__).parents[3] / path
            
            with open(full_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            logger.debug(f"Loaded prompt template from: {full_path}")
            return template
            
        except FileNotFoundError:
            logger.warning(f"Prompt template file not found: {path}, using default")
            return self._get_default_prompt()
        except Exception as e:
            logger.warning(f"Failed to load prompt template: {e}, using default")
            return self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """Get default prompt template for image captioning."""
        return """You are an expert at analyzing and describing technical images, 
diagrams, charts, and screenshots. Provide a detailed, comprehensive description 
of the provided image.

Describe:
1. Image type (diagram, chart, screenshot, etc.)
2. Key elements and their relationships
3. Any visible text or labels
4. Technical context if applicable

Provide a clear, structured description in 2-4 sentences."""


# Export for use in other modules
__all__ = ["ImageCaptioner"]
