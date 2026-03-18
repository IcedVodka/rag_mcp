#!/usr/bin/env python3
"""
Response Builder - Build MCP-formatted responses with citations.

This module provides the ResponseBuilder class that constructs MCP-compliant
responses from retrieval results, including both human-readable Markdown
and structured citation data.

Supports multimodal responses (Text + Image) when retrieval results
contain image references.
"""

from typing import Any, Optional

from core.types import RetrievalResult
from core.response.citation_generator import (
    CitationGenerator, 
    Citation,
    format_citation_markdown,
    format_inline_citation
)
from core.response.multimodal_assembler import MultimodalAssembler


class ResponseBuilder:
    """
    Builder for MCP-formatted responses with citations.
    
    This class constructs responses that include:
    - Human-readable Markdown text with inline citations [1], [2], etc.
    - Base64-encoded images when retrieval results contain image_refs
    - Structured citation data for client-side processing
    
    Attributes:
        citation_generator: Generator for creating citations
        multimodal_assembler: Assembler for multimodal content (text + images)
        enable_multimodal: Whether to include images in responses
    """
    
    def __init__(
        self,
        citation_generator: Optional[CitationGenerator] = None,
        multimodal_assembler: Optional[MultimodalAssembler] = None,
        enable_multimodal: bool = True
    ):
        """
        Initialize ResponseBuilder.
        
        Args:
            citation_generator: Optional citation generator instance.
                              If not provided, creates a default one.
            multimodal_assembler: Optional multimodal assembler instance.
                                 If not provided, creates a default one.
            enable_multimodal: Whether to include images in responses.
                              Set to False for text-only responses.
        """
        self.citation_generator = citation_generator or CitationGenerator()
        self.multimodal_assembler = multimodal_assembler or MultimodalAssembler()
        self.enable_multimodal = enable_multimodal
    
    def build(
        self, 
        retrieval_results: list[RetrievalResult], 
        query: str,
        include_references: bool = True
    ) -> dict[str, Any]:
        """
        Build an MCP-formatted response from retrieval results.
        
        Args:
            retrieval_results: List of retrieval results from search
            query: Original query string (for context)
            include_references: Whether to include reference section in markdown
            
        Returns:
            MCP-formatted response dictionary with content and structuredContent
        """
        # Handle empty results
        if not retrieval_results:
            return self._build_empty_response(query)
        
        # Generate citations
        citations = self.citation_generator.generate(retrieval_results)
        
        # Build markdown content
        markdown_text = self._build_markdown(
            retrieval_results, citations, query, include_references
        )
        
        # Build structured content
        structured_content = self._build_structured_content(
            retrieval_results, citations, query
        )
        
        # Build MCP content (text + optional images)
        if self.enable_multimodal:
            content = self.multimodal_assembler.assemble_response(
                markdown_text, citations, retrieval_results
            )
        else:
            content = [{"type": "text", "text": markdown_text}]
        
        # Build MCP response
        return {
            "content": content,
            "structuredContent": structured_content,
            "isError": False
        }
    
    def _build_empty_response(self, query: str) -> dict[str, Any]:
        """
        Build a response for when no results are found.
        
        Args:
            query: The original query
            
        Returns:
            Friendly response indicating no results
        """
        message = (
            f"I couldn't find any relevant information for your query: \"{query}\"\n\n"
            "Suggestions:\n"
            "- Try using different keywords or phrasing\n"
            "- Check if the document collection contains relevant content\n"
            "- Try a more general search term"
        )
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": message
                }
            ],
            "structuredContent": {
                "query": query,
                "result_count": 0,
                "citations": []
            },
            "isError": False
        }
    
    def _build_markdown(
        self,
        results: list[RetrievalResult],
        citations: list[Citation],
        query: str,
        include_references: bool
    ) -> str:
        """
        Build human-readable Markdown content.
        
        Args:
            results: Retrieval results
            citations: Generated citations
            query: Original query
            include_references: Whether to include reference section
            
        Returns:
            Markdown formatted text
        """
        lines = []
        
        # Add summary
        lines.append(f"Found {len(results)} relevant result(s) for: \"{query}\"\n")
        
        # Add each result with citation
        for i, (result, citation) in enumerate(zip(results, citations)):
            lines.append(self._format_result_section(result, citation, i + 1))
        
        # Add reference section if requested
        if include_references:
            lines.append(format_citation_markdown(citations))
        
        return "\n".join(lines)
    
    def _format_result_section(
        self, 
        result: RetrievalResult, 
        citation: Citation,
        index: int
    ) -> str:
        """
        Format a single result as a Markdown section.
        
        Args:
            result: Retrieval result
            citation: Associated citation
            index: Result index (1-based)
            
        Returns:
            Markdown formatted section
        """
        lines = []
        
        # Result header with citation
        inline_cite = format_inline_citation(citation.id)
        
        # Source info
        page_info = f" (p.{citation.page})" if citation.page else ""
        header = f"**Result {index}**{inline_cite}: {citation.source}{page_info}"
        lines.append(header)
        
        # Content snippet (truncated if very long)
        text = result.text.strip()
        max_length = 800
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        lines.append(f"> {text}\n")
        
        return "\n".join(lines)
    
    def _build_structured_content(
        self,
        results: list[RetrievalResult],
        citations: list[Citation],
        query: str
    ) -> dict[str, Any]:
        """
        Build structured content for programmatic access.
        
        Args:
            results: Retrieval results
            citations: Generated citations
            query: Original query
            
        Returns:
            Structured content dictionary
        """
        return {
            "query": query,
            "result_count": len(results),
            "citations": [c.to_dict() for c in citations]
        }


def build_simple_response(
    text: str, 
    is_error: bool = False
) -> dict[str, Any]:
    """
    Build a simple text response without citations.
    
    Args:
        text: Response text
        is_error: Whether this is an error response
        
    Returns:
        Simple MCP response
    """
    return {
        "content": [
            {
                "type": "text",
                "text": text
            }
        ],
        "isError": is_error
    }
