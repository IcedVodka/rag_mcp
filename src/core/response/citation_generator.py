#!/usr/bin/env python3
"""
Citation Generator - Generate structured citations from retrieval results.

This module provides the CitationGenerator class that creates structured
 citation information from retrieval results, enabling transparent
 source attribution in MCP tool responses.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from core.types import RetrievalResult


@dataclass
class Citation:
    """
    Structured citation for a retrieved result.
    
    Attributes:
        id: Sequential citation number (1-indexed)
        source: Source file name or path
        page: Page number in the source document (optional)
        chunk_id: Unique chunk identifier
        text: The retrieved text content
        score: Relevance score
        metadata: Additional metadata associated with the chunk
    """
    id: int
    source: str
    page: Optional[int]
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "id": self.id,
            "source": self.source,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "text": self.text[:500] if self.text else "",  # Truncate for display
            "score": round(self.score, 4),
        }
        # Include relevant metadata fields if present
        for key in ["title", "doc_type", "collection"]:
            if key in self.metadata:
                result[key] = self.metadata[key]
        return result


class CitationGenerator:
    """
    Generator for creating structured citations from retrieval results.
    
    This class transforms RetrievalResult objects into structured Citation
    objects that can be used for source attribution in MCP responses.
    
    Attributes:
        None - stateless generator
    """
    
    def generate(self, retrieval_results: list[RetrievalResult]) -> list[Citation]:
        """
        Generate citations from retrieval results.
        
        Args:
            retrieval_results: List of retrieval results from search
            
        Returns:
            List of Citation objects with sequential IDs
        """
        citations = []
        
        for idx, result in enumerate(retrieval_results, start=1):
            citation = self._create_citation(idx, result)
            citations.append(citation)
        
        return citations
    
    def _create_citation(self, citation_id: int, result: RetrievalResult) -> Citation:
        """
        Create a single citation from a retrieval result.
        
        Args:
            citation_id: Sequential citation number
            result: Retrieval result to convert
            
        Returns:
            Citation object with extracted metadata
        """
        metadata = result.metadata or {}
        
        # Extract source information from metadata
        source = self._extract_source(metadata)
        
        # Extract page number from metadata
        page = self._extract_page(metadata)
        
        return Citation(
            id=citation_id,
            source=source,
            page=page,
            chunk_id=result.chunk_id,
            text=result.text,
            score=result.score,
            metadata=metadata.copy()
        )
    
    def _extract_source(self, metadata: dict[str, Any]) -> str:
        """
        Extract source file name from metadata.
        
        Args:
            metadata: Chunk metadata dictionary
            
        Returns:
            Source file name or path, or "unknown" if not found
        """
        # Try common source fields
        for key in ["source_path", "source", "file_path", "file", "doc_id"]:
            if key in metadata and metadata[key]:
                source = metadata[key]
                # If it's a path, return just the filename
                if isinstance(source, str) and "/" in source:
                    return source.split("/")[-1]
                return str(source)
        
        return "unknown"
    
    def _extract_page(self, metadata: dict[str, Any]) -> Optional[int]:
        """
        Extract page number from metadata.
        
        Args:
            metadata: Chunk metadata dictionary
            
        Returns:
            Page number if found, None otherwise
        """
        # Try common page fields
        for key in ["page", "page_num", "page_number", "slide", "slide_num"]:
            if key in metadata:
                value = metadata[key]
                if isinstance(value, int):
                    return value
                if isinstance(value, str) and value.isdigit():
                    return int(value)
        
        return None


def format_citation_markdown(citations: list[Citation]) -> str:
    """
    Format citations as a Markdown reference section.
    
    Args:
        citations: List of citations to format
        
    Returns:
        Markdown formatted citation list
    """
    if not citations:
        return ""
    
    lines = ["\n---", "\n**References:**\n"]
    
    for citation in citations:
        page_info = f", p.{citation.page}" if citation.page else ""
        lines.append(
            f"[{citation.id}] **{citation.source}**{page_info} "
            f"(score: {citation.score:.3f})"
        )
    
    return "\n".join(lines)


def format_inline_citation(citation_id: int) -> str:
    """
    Format an inline citation marker.
    
    Args:
        citation_id: Citation number
        
    Returns:
        Formatted citation marker like "[1]"
    """
    return f"[{citation_id}]"
