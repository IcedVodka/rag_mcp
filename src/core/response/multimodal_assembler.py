#!/usr/bin/env python3
"""
Multimodal Assembler - Assemble MCP responses with text and images.

This module provides the MultimodalAssembler class that constructs MCP-compliant
responses containing both text content and base64-encoded images. When retrieval
results contain image references (image_refs), the assembler loads the images,
encodes them as base64, and includes them as ImageContent in the response.
"""

import base64
import logging
from pathlib import Path
from typing import Any, Optional

from core.types import RetrievalResult
from core.response.citation_generator import Citation
from ingestion.storage.image_storage import ImageStorage

logger = logging.getLogger(__name__)


class MultimodalAssembler:
    """
    Assembler for multimodal MCP responses (Text + Image).
    
    This class constructs responses that include:
    - Human-readable Markdown text with inline citations
    - Base64-encoded images referenced by retrieval results
    - Structured citation data for client-side processing
    
    Attributes:
        image_storage: ImageStorage instance for retrieving image files
    """
    
    def __init__(self, image_storage: Optional[ImageStorage] = None):
        """
        Initialize MultimodalAssembler.
        
        Args:
            image_storage: ImageStorage instance for image retrieval.
                          If not provided, creates a default one.
        """
        self.image_storage = image_storage or ImageStorage()
    
    def assemble_response(
        self,
        text_content: str,
        citations: list[Citation],
        retrieval_results: list[RetrievalResult]
    ) -> list[dict[str, Any]]:
        """
        Assemble MCP content array with text and images.
        
        This method:
        1. Creates a TextContent entry with the markdown text
        2. Extracts image_refs from retrieval results
        3. Loads each referenced image and encodes as base64
        4. Creates ImageContent entries for each image
        5. Returns combined content array
        
        If image loading fails, gracefully degrades to text-only response.
        
        Args:
            text_content: Markdown text content to include
            citations: List of citations for the retrieval results
            retrieval_results: List of retrieval results (may contain image_refs)
            
        Returns:
            MCP content array with TextContent and optional ImageContent entries
        """
        # Start with text content
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": text_content
            }
        ]
        
        # Extract unique image references from retrieval results
        image_ids = self._extract_image_refs(retrieval_results)
        
        if not image_ids:
            return content
        
        # Load and encode each image
        for image_id in image_ids:
            image_content = self._load_image_as_content(image_id)
            if image_content:
                content.append(image_content)
        
        return content
    
    def _extract_image_refs(
        self,
        retrieval_results: list[RetrievalResult]
    ) -> list[str]:
        """
        Extract unique image references from retrieval results.
        
        Args:
            retrieval_results: List of retrieval results
            
        Returns:
            List of unique image IDs referenced by the results
        """
        image_ids: set[str] = set()
        
        for result in retrieval_results:
            metadata = result.metadata or {}
            image_refs = metadata.get("image_refs")
            
            if isinstance(image_refs, list):
                for ref in image_refs:
                    if isinstance(ref, str) and ref.strip():
                        image_ids.add(ref.strip())
            elif isinstance(image_refs, str) and image_refs.strip():
                # Handle case where image_refs might be a single string
                image_ids.add(image_refs.strip())
        
        return list(image_ids)
    
    def _load_image_as_content(self, image_id: str) -> Optional[dict[str, Any]]:
        """
        Load an image and create ImageContent dict.
        
        Args:
            image_id: Unique image identifier
            
        Returns:
            ImageContent dict with base64 data, or None if loading fails
        """
        try:
            # Get image record for mime type
            record = self.image_storage.get_image_record(image_id)
            if not record:
                logger.warning(f"Image not found in index: {image_id}")
                return None
            
            # Read image file
            file_path = record.file_path
            if not file_path or not Path(file_path).exists():
                logger.warning(f"Image file not found: {file_path}")
                return None
            
            # Read and encode image
            with open(file_path, "rb") as f:
                image_data = f.read()
            
            base64_data = base64.b64encode(image_data).decode("utf-8")
            
            # Determine mime type
            mime_type = record.mime_type or self._guess_mime_type(file_path)
            
            return {
                "type": "image",
                "data": base64_data,
                "mimeType": mime_type
            }
            
        except Exception as e:
            logger.error(f"Failed to load image {image_id}: {e}")
            return None
    
    def _guess_mime_type(self, file_path: str) -> str:
        """
        Guess MIME type from file extension.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            MIME type string (defaults to image/png)
        """
        ext = Path(file_path).suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".svg": "image/svg+xml",
        }
        return mime_types.get(ext, "image/png")


def assemble_multimodal_response(
    text_content: str,
    citations: list[Citation],
    retrieval_results: list[RetrievalResult],
    image_storage: Optional[ImageStorage] = None
) -> list[dict[str, Any]]:
    """
    Convenience function to assemble multimodal response content.
    
    Args:
        text_content: Markdown text content
        citations: List of citations
        retrieval_results: List of retrieval results
        image_storage: Optional ImageStorage instance
        
    Returns:
        MCP content array with text and images
    """
    assembler = MultimodalAssembler(image_storage)
    return assembler.assemble_response(text_content, citations, retrieval_results)
