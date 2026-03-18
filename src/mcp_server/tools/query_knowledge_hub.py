#!/usr/bin/env python3
"""
Query Knowledge Hub Tool - Main retrieval tool for MCP Server.

This module implements the query_knowledge_hub tool that orchestrates
the complete retrieval flow: Hybrid Search → Reranking → Response Building.
"""

import logging
from typing import Any, Optional

from core.settings import Settings, load_settings
from core.types import RetrievalResult
from core.query_engine.hybrid_search import HybridSearch
from core.query_engine.reranker import CoreReranker
from core.response.response_builder import ResponseBuilder
from core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Query Engine that orchestrates hybrid search and reranking.
    
    This class manages the lifecycle of search components and provides
    a clean interface for executing queries.
    
    Attributes:
        settings: Application settings
        hybrid_search: Hybrid search component
        reranker: Reranker component (optional)
        response_builder: Response builder
    """
    
    def __init__(
        self,
        settings: Settings,
        hybrid_search: Optional[HybridSearch] = None,
        reranker: Optional[CoreReranker] = None,
    ):
        """
        Initialize QueryEngine.
        
        Args:
            settings: Application settings
            hybrid_search: Optional hybrid search instance
            reranker: Optional reranker instance
        """
        self.settings = settings
        
        # Initialize or use provided components
        self.hybrid_search = hybrid_search or self._create_hybrid_search()
        
        # Reranker is optional based on configuration
        self.reranker: Optional[CoreReranker] = None
        if reranker is not None:
            self.reranker = reranker
        elif self._should_use_reranker():
            self.reranker = CoreReranker(settings)
        
        self.response_builder = ResponseBuilder()
        
        logger.info(
            f"QueryEngine initialized: hybrid_search={self.hybrid_search is not None}, "
            f"reranker={self.reranker is not None}"
        )
    
    def _create_hybrid_search(self) -> HybridSearch:
        """Create HybridSearch instance from settings."""
        # For now, create a basic HybridSearch without injected components
        # In production, these would be properly initialized with real retrievers
        return HybridSearch(self.settings)
    
    def _should_use_reranker(self) -> bool:
        """Check if reranker should be used based on settings."""
        backend = getattr(self.settings.reranker, 'backend', 'none')
        return backend not in ('none', '', None)
    
    async def query(
        self,
        query: str,
        top_k: int = 10,
        collection: str = "default",
        trace: Optional[TraceContext] = None
    ) -> dict[str, Any]:
        """
        Execute a query against the knowledge hub.
        
        Flow:
        1. Execute hybrid search
        2. Rerank results (if enabled)
        3. Build MCP response with citations
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            collection: Document collection to search
            trace: Optional trace context for observability
            
        Returns:
            MCP-formatted response with content and structuredContent
        """
        if trace:
            trace.record_stage(
                "query_start",
                details={"query": query, "top_k": top_k, "collection": collection}
            )
        
        try:
            # Step 1: Hybrid Search
            logger.info(f"Executing hybrid search for: {query}")
            
            # Create filters for collection if specified
            filters = {"collection": collection} if collection and collection != "default" else None
            
            # Execute search
            search_results = self.hybrid_search.search(
                query=query,
                top_k=top_k * 2,  # Retrieve more for reranking
                filters=filters,
                trace=trace
            )
            
            logger.info(f"Hybrid search returned {len(search_results)} results")
            
            # Step 2: Reranking (if enabled)
            final_results = search_results
            if self.reranker and search_results:
                logger.info("Running reranking...")
                rerank_info = self.reranker.rerank(query, search_results, trace)
                final_results = rerank_info.results
                logger.info(
                    f"Reranking complete: {len(final_results)} results, "
                    f"fallback={rerank_info.fallback}"
                )
            
            # Limit to requested top_k
            final_results = final_results[:top_k]
            
            # Step 3: Build Response
            response = self.response_builder.build(final_results, query)
            
            if trace:
                trace.record_stage(
                    "query_complete",
                    details={
                        "result_count": len(final_results),
                        "has_references": len(final_results) > 0
                    }
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}", exc_info=True)
            
            if trace:
                trace.record_stage("query_error", details={"error": str(e)})
            
            # Return error response
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Sorry, an error occurred while searching: {str(e)}"
                    }
                ],
                "structuredContent": {
                    "query": query,
                    "error": str(e),
                    "result_count": 0,
                    "citations": []
                },
                "isError": True
            }


# Global query engine instance (initialized lazily)
_query_engine: Optional[QueryEngine] = None
_settings: Optional[Settings] = None


def _get_settings() -> Settings:
    """Get or load settings."""
    global _settings
    if _settings is None:
        import os
        config_path = os.environ.get(
            "KNOWLEDGE_HUB_CONFIG", 
            "config/settings.yaml"
        )
        _settings = load_settings(config_path)
    return _settings


def _get_query_engine() -> QueryEngine:
    """Get or initialize query engine."""
    global _query_engine
    if _query_engine is None:
        settings = _get_settings()
        _query_engine = QueryEngine(settings)
    return _query_engine


async def query_knowledge_hub(
    query: str,
    top_k: int = 10,
    collection: str = "default"
) -> dict[str, Any]:
    """
    Query the knowledge hub to retrieve relevant information.
    
    This is the main MCP tool for searching the knowledge base. It performs
    hybrid search (dense + sparse) with optional reranking and returns
    results with full citation information.
    
    Args:
        query: The search query string. Be specific for best results.
        top_k: Number of top results to return (default: 10, max: 50)
        collection: Document collection to search (default: "default")
        
    Returns:
        MCP-formatted response containing:
        - content[0]: Human-readable Markdown with inline citations [1], [2], etc.
        - structuredContent.citations: List of citation objects with:
          - id: Citation number
          - source: Source file name
          - page: Page number (if available)
          - chunk_id: Unique chunk identifier
          - text: Retrieved text snippet
          - score: Relevance score
          
    Example:
        >>> result = await query_knowledge_hub(
        ...     query="What is the refund policy?",
        ...     top_k=5,
        ...     collection="docs"
        ... )
        >>> print(result["content"][0]["text"])  # Markdown output
        >>> print(result["structuredContent"]["citations"])  # Structured citations
    """
    # Validate inputs
    if not query or not isinstance(query, str):
        raise ValueError("Query must be a non-empty string")
    
    # Clamp top_k to reasonable range
    top_k = max(1, min(top_k, 50))
    
    # Get query engine and execute
    engine = _get_query_engine()
    return await engine.query(query, top_k, collection)


# Tool schema for MCP registration
TOOL_NAME = "query_knowledge_hub"
TOOL_DESCRIPTION = (
    "Search the knowledge hub for relevant information. "
    "Returns results with citations showing source document and page numbers."
)
TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query string. Be specific for best results."
        },
        "top_k": {
            "type": "integer",
            "description": "Number of top results to return (default: 10, max: 50)",
            "minimum": 1,
            "maximum": 50,
            "default": 10
        },
        "collection": {
            "type": "string",
            "description": "Document collection to search (default: 'default')",
            "default": "default"
        }
    },
    "required": ["query"]
}


def reset_engine():
    """Reset the global query engine (useful for testing)."""
    global _query_engine, _settings
    _query_engine = None
    _settings = None
