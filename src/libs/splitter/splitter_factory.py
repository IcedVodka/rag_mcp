#!/usr/bin/env python3
"""
Splitter Factory - Strategy-Agnostic Splitter Creation

Creates appropriate text splitting instances based on configuration.
"""

from typing import Optional, Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from .base_splitter import BaseSplitter


class SplitterFactory:
    """
    Factory for creating text splitter instances.
    
    Routes to appropriate strategy implementation based on settings.
    """
    
    @staticmethod
    def create(settings: Settings) -> BaseSplitter:
        """
        Create a splitter instance based on settings.
        
        Args:
            settings: Application settings
            
        Returns:
            Configured splitter instance
            
        Raises:
            ValueError: If strategy is unknown
        """
        strategy = settings.splitter.strategy
        config = getattr(settings.splitter, strategy, {})
        
        if strategy == "recursive":
            from .recursive_splitter import RecursiveSplitter
            return RecursiveSplitter(config)
        elif strategy == "semantic":
            from .semantic_splitter import SemanticSplitter
            return SemanticSplitter(config)
        elif strategy == "fixed":
            from .fixed_length_splitter import FixedLengthSplitter
            return FixedLengthSplitter(config)
        else:
            raise ValueError(f"Unknown splitter strategy: {strategy}")


class SplitterProvider:
    """
    Wrapper for splitter with common functionality.
    
    Provides trace recording and statistics.
    """
    
    def __init__(self, splitter: BaseSplitter) -> None:
        self.splitter = splitter
    
    def split_with_trace(
        self,
        text: str,
        trace: Optional[TraceContext] = None
    ) -> list[str]:
        """
        Split text with automatic trace recording.
        
        Args:
            text: Text to split
            trace: Optional trace context
            
        Returns:
            Text chunks
        """
        chunks = self.splitter.split_text(text, trace)
        
        if trace:
            trace.record_stage(
                name="text_splitting",
                method=self.splitter.__class__.__name__,
                details={
                    "chunk_count": len(chunks),
                    "chunk_size": self.splitter.chunk_size,
                    "chunk_overlap": self.splitter.chunk_overlap,
                    "original_length": len(text)
                }
            )
        
        return chunks
