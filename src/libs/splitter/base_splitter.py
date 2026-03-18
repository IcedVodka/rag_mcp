#!/usr/bin/env python3
"""
Base Splitter - Abstract Interface for Text Splitting

Defines the contract for text splitting strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any

from core.trace.trace_context import TraceContext


class BaseSplitter(ABC):
    """
    Abstract base class for text splitting strategies.
    
    All splitters (Recursive, Semantic, Fixed) must implement this interface.
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the splitter with configuration.
        
        Args:
            config: Strategy-specific configuration
        """
        self.config = config
        self.chunk_size = config.get("chunk_size", 1000)
        self.chunk_overlap = config.get("chunk_overlap", 200)
    
    @abstractmethod
    def split_text(
        self,
        text: str,
        trace: Optional[TraceContext] = None
    ) -> list[str]:
        """
        Split text into chunks.
        
        Args:
            text: Text to split
            trace: Optional trace context for observability
            
        Returns:
            List of text chunks
        """
        pass
    
    def split_documents(
        self,
        documents: list[str],
        trace: Optional[TraceContext] = None
    ) -> list[list[str]]:
        """
        Split multiple documents.
        
        Args:
            documents: List of documents to split
            trace: Optional trace context
            
        Returns:
            List of chunk lists (one per document)
        """
        results = []
        for doc in documents:
            chunks = self.split_text(doc, trace)
            results.append(chunks)
        return results


class SplitterError(Exception):
    """Base exception for splitter errors."""
    pass


class SplitterConfigError(SplitterError):
    """Raised when splitter configuration is invalid."""
    pass
