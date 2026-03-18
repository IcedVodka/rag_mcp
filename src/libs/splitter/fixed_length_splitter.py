#!/usr/bin/env python3
"""
Fixed Length Splitter - Fixed-Length Text Splitter

Simple character-based splitting with fixed chunk size.
"""

from typing import Optional, Any

from core.trace.trace_context import TraceContext
from .base_splitter import BaseSplitter


class FixedLengthSplitter(BaseSplitter):
    """
    Simple fixed-length text splitter.
    
    Splits text into chunks of specified size with optional overlap.
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
    
    def split_text(
        self,
        text: str,
        trace: Optional[TraceContext] = None
    ) -> list[str]:
        """Split text into fixed-length chunks."""
        if not text:
            return []
        
        if trace:
            trace.record_stage(
                name="text_splitting",
                method="FixedLengthSplitter",
                details={
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap
                }
            )
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunks.append(text[start:end])
            
            # Move start position, considering overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop if overlap >= chunk_size
            if start >= end:
                start = end
        
        return chunks
