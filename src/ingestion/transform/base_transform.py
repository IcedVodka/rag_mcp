#!/usr/bin/env python3
"""
Base Transform - Abstract Interface for Document Transformations

Defines the contract for transformation steps in the ingestion pipeline.
Transformations may include text refinement, OCR correction, image captioning,
metadata enrichment, and other preprocessing operations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any

from core.types import Chunk
from core.trace.trace_context import TraceContext


class BaseTransform(ABC):
    """
    Abstract base class for document/chunk transformations.
    
    Transformations are processing steps that modify chunks during the ingestion
    pipeline. They can be used for text cleaning, format normalization,
    OCR error correction, LLM-based enhancement, and more.
    
    Each transform receives a list of chunks and an optional trace context,
    and returns a list of transformed chunks. The transform may modify chunks
    in place or create new chunk instances.
    
    Example:
        >>> class MyTransform(BaseTransform):
        ...     def transform(self, chunks, trace=None):
        ...         for chunk in chunks:
        ...             chunk.text = chunk.text.lower()
        ...         return chunks
    """
    
    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        """
        Initialize the transform with configuration.
        
        Args:
            config: Optional configuration dictionary for the transform.
                   Subclasses may define specific configuration options.
        """
        self.config = config or {}
    
    @abstractmethod
    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """
        Transform a list of chunks.
        
        This is the main entry point for transformation. Implementations
        should process the input chunks and return the transformed results.
        
        The transform should:
        - Handle empty input gracefully (return empty list)
        - Preserve chunk metadata unless explicitly modifying it
        - Record processing stages to the trace context if provided
        - Handle errors appropriately (log and continue, or raise based on config)
        
        Args:
            chunks: List of chunks to transform
            trace: Optional trace context for recording processing stages
            
        Returns:
            List of transformed chunks (may be same objects or new instances)
            
        Raises:
            TransformError: If transformation fails and configured to raise on error
        """
        pass
    
    def __call__(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """
        Make transform callable as a function.
        
        Args:
            chunks: List of chunks to transform
            trace: Optional trace context
            
        Returns:
            List of transformed chunks
        """
        return self.transform(chunks, trace)


class TransformError(Exception):
    """Base exception for transformation errors."""
    pass


class TransformConfigError(TransformError):
    """Raised when transform configuration is invalid."""
    pass


class TransformProcessingError(TransformError):
    """Raised when transformation processing fails."""
    
    def __init__(
        self,
        message: str,
        chunk_id: Optional[str] = None,
        original_error: Optional[Exception] = None
    ) -> None:
        """
        Initialize processing error with context.
        
        Args:
            message: Error message
            chunk_id: ID of the chunk that caused the error
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.chunk_id = chunk_id
        self.original_error = original_error
