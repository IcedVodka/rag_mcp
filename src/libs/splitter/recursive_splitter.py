#!/usr/bin/env python3
"""
Recursive Splitter - LangChain-based Recursive Text Splitter

Uses LangChain's RecursiveCharacterTextSplitter for intelligent text splitting.
"""

from typing import Optional, Any

from core.trace.trace_context import TraceContext
from .base_splitter import BaseSplitter, SplitterError


class RecursiveSplitter(BaseSplitter):
    """
    Text splitter using LangChain's recursive character splitting.
    
    Tries to split on separators in order, keeping related text together.
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.separators = config.get("separators", ["\n\n", "\n", ". ", "? ", "! ", " ", ""])
        self.keep_separator = config.get("keep_separator", True)
        
        # Try to import LangChain's splitter
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
                keep_separator=self.keep_separator,
                strip_whitespace=True
            )
        except ImportError:
            # Fallback to simple implementation
            self._splitter = None
    
    def _simple_split(self, text: str) -> list[str]:
        """Simple fallback splitting when LangChain is not available."""
        if not text:
            return []
        
        chunks = []
        current_chunk = ""
        
        # Simple character-based splitting
        for para in text.split("\n\n"):
            if not para.strip():
                continue
                
            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If paragraph itself is too long, split it
                if len(para) > self.chunk_size:
                    # Split by sentences
                    sentences = para.replace(". ", ".\n").split("\n")
                    current_chunk = ""
                    for sent in sentences:
                        if len(current_chunk) + len(sent) > self.chunk_size:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sent
                        else:
                            current_chunk += " " + sent if current_chunk else sent
                else:
                    current_chunk = para
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def split_text(
        self,
        text: str,
        trace: Optional[TraceContext] = None
    ) -> list[str]:
        """Split text into chunks."""
        if trace:
            trace.record_stage(
                name="text_splitting",
                method="RecursiveSplitter",
                details={
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap
                }
            )
        
        if self._splitter:
            return self._splitter.split_text(text)
        else:
            return self._simple_split(text)
