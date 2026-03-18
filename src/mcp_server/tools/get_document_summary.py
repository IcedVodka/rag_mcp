#!/usr/bin/env python3
"""
Get Document Summary Tool - Retrieve document metadata and statistics.

This module implements the get_document_summary tool that returns
document summary information (title, summary, tags) and statistics
(chunk count, total characters) for a given document ID.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

from core.settings import Settings, load_settings
from libs.vector_store.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


class DocumentSummaryProvider:
    """
    Provider for document summary information.
    
    This class coordinates data collection from ChromaDB to retrieve
    document metadata and calculate statistics.
    
    Attributes:
        settings: Application settings
        chroma_store: ChromaDB vector store instance
    """
    
    def __init__(
        self,
        settings: Settings,
        chroma_store: Optional[ChromaStore] = None,
    ):
        """
        Initialize DocumentSummaryProvider.
        
        Args:
            settings: Application settings
            chroma_store: Optional ChromaStore instance
        """
        self.settings = settings
        self.chroma_store = chroma_store or self._create_chroma_store()
        
        logger.info(
            f"DocumentSummaryProvider initialized"
        )
    
    def _create_chroma_store(self) -> ChromaStore:
        """Create ChromaStore instance from settings."""
        provider = self.settings.vector_store.provider
        if provider != "chroma":
            raise ValueError(f"get_document_summary only supports chroma provider, got: {provider}")
        
        config = getattr(self.settings.vector_store, provider, {})
        return ChromaStore(config)
    
    def _get_document_chunks(self, doc_id: str, collection: str = "default") -> list[dict[str, Any]]:
        """
        Get all chunks for a document from ChromaDB.
        
        Args:
            doc_id: Document ID (filename or path)
            collection: Collection name
            
        Returns:
            List of chunk records with id, text, and metadata
        """
        # Try different source field matching strategies
        # Strategy 1: Exact match on source
        filters = {"source": doc_id}
        chunks = self.chroma_store.get_by_metadata(filters)
        
        if not chunks:
            # Strategy 2: Try with just the filename (basename)
            doc_name = Path(doc_id).name
            filters = {"source": doc_name}
            chunks = self.chroma_store.get_by_metadata(filters)
        
        if not chunks:
            # Strategy 3: Try source contains doc_id
            # Note: ChromaDB doesn't support contains directly in where clause,
            # so we need to get all and filter (inefficient but works for small collections)
            pass
        
        return chunks
    
    def _aggregate_metadata(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Aggregate metadata from document chunks.
        
        Args:
            chunks: List of chunk records
            
        Returns:
            Aggregated metadata dictionary
        """
        if not chunks:
            return {}
        
        # Use the first chunk's metadata as base
        first_chunk = chunks[0]
        metadata = first_chunk.get("metadata", {})
        
        # Extract key fields
        title = metadata.get("title", "")
        summary = metadata.get("summary", "")
        tags = metadata.get("tags", [])
        source = metadata.get("source", "")
        
        # Collect all unique page numbers
        pages = set()
        for chunk in chunks:
            chunk_meta = chunk.get("metadata", {})
            page = chunk_meta.get("page")
            if page is not None:
                try:
                    pages.add(int(page))
                except (ValueError, TypeError):
                    pass
        
        return {
            "title": title,
            "summary": summary,
            "tags": tags if isinstance(tags, list) else [tags] if tags else [],
            "source": source,
            "pages": sorted(list(pages)) if pages else []
        }
    
    def _calculate_statistics(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Calculate document statistics from chunks.
        
        Args:
            chunks: List of chunk records
            
        Returns:
            Statistics dictionary with chunk_count and total_chars
        """
        chunk_count = len(chunks)
        total_chars = sum(len(chunk.get("text", "")) for chunk in chunks)
        
        return {
            "chunk_count": chunk_count,
            "total_chars": total_chars
        }
    
    async def get_summary(self, doc_id: str, collection: str = "default") -> dict[str, Any]:
        """
        Get summary information for a document.
        
        Args:
            doc_id: Document ID (filename or path)
            collection: Collection name
            
        Returns:
            MCP-formatted response with document summary:
            - content[0]: Human-readable Markdown with document info
            - structuredContent: Structured document metadata and statistics
        """
        try:
            # Validate doc_id
            if not doc_id or not isinstance(doc_id, str):
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": "## 错误\n\n文档ID不能为空。"
                        }
                    ],
                    "structuredContent": {
                        "error": "Document ID is required",
                        "doc_id": doc_id
                    },
                    "isError": True
                }
            
            # Get all chunks for the document
            chunks = self._get_document_chunks(doc_id, collection)
            
            if not chunks:
                # Document not found
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"## 文档未找到\n\n未找到文档: `{doc_id}`\n\n请检查文档ID是否正确，或先使用 `list_collections` 查看可用集合。"
                        }
                    ],
                    "structuredContent": {
                        "error": "Document not found",
                        "doc_id": doc_id,
                        "collection": collection
                    },
                    "isError": True
                }
            
            # Aggregate metadata and calculate statistics
            metadata = self._aggregate_metadata(chunks)
            stats = self._calculate_statistics(chunks)
            
            # Build source path
            source_path = metadata.get("source", doc_id)
            if not source_path.startswith("/") and not source_path.startswith("data/"):
                # Construct full path
                source_path = f"data/documents/{collection}/{source_path}"
            
            # Build human-readable text
            title_display = metadata.get("title") or doc_id
            lines = [f"## 文档摘要: {title_display}\n"]
            
            if metadata.get("title"):
                lines.append(f"**标题**: {metadata['title']}")
            
            if metadata.get("summary"):
                lines.append(f"**摘要**: {metadata['summary']}")
            
            tags = metadata.get("tags", [])
            if tags:
                tags_str = ", ".join(f"`{tag}`" for tag in tags)
                lines.append(f"**标签**: {tags_str}")
            
            lines.append(f"**统计**: {stats['chunk_count']}个chunks, 共{stats['total_chars']}字符")
            
            if metadata.get("pages"):
                pages_str = ", ".join(str(p) for p in metadata["pages"][:10])
                if len(metadata["pages"]) > 10:
                    pages_str += f" ... (共{len(metadata['pages'])}页)"
                lines.append(f"**页码**: {pages_str}")
            
            lines.append(f"**来源**: {source_path}")
            
            text_content = "\n".join(lines)
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": text_content
                    }
                ],
                "structuredContent": {
                    "doc_id": doc_id,
                    "title": metadata.get("title", ""),
                    "summary": metadata.get("summary", ""),
                    "tags": metadata.get("tags", []),
                    "source_path": source_path,
                    "chunk_count": stats["chunk_count"],
                    "total_chars": stats["total_chars"],
                    "pages": metadata.get("pages", [])
                }
            }
            
        except Exception as e:
            logger.error(f"Get document summary failed: {e}", exc_info=True)
            
            # Return error response
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"## 错误\n\n获取文档摘要失败: {str(e)}"
                    }
                ],
                "structuredContent": {
                    "error": str(e),
                    "doc_id": doc_id,
                    "collection": collection
                },
                "isError": True
            }


# Global provider instance (initialized lazily)
_summary_provider: Optional[DocumentSummaryProvider] = None
_settings: Optional[Settings] = None


def _get_settings() -> Settings:
    """Get or load settings."""
    global _settings
    if _settings is None:
        config_path = os.environ.get(
            "KNOWLEDGE_HUB_CONFIG", 
            "config/settings.yaml"
        )
        _settings = load_settings(config_path)
    return _settings


def _get_summary_provider() -> DocumentSummaryProvider:
    """Get or initialize summary provider."""
    global _summary_provider
    if _summary_provider is None:
        settings = _get_settings()
        _summary_provider = DocumentSummaryProvider(settings)
    return _summary_provider


async def get_document_summary(
    doc_id: str,
    collection: str = "default"
) -> dict[str, Any]:
    """
    Get summary information for a specific document.
    
    Returns document metadata (title, summary, tags) and statistics
    (chunk count, total characters) for the specified document.
    
    Args:
        doc_id: Document ID or filename (e.g., "example.pdf")
        collection: Document collection name (default: "default")
        
    Returns:
        MCP-formatted response containing:
        - content[0]: Human-readable Markdown with document info
        - structuredContent: Structured document metadata:
          - doc_id: Document identifier
          - title: Document title
          - summary: Document summary/description
          - tags: List of document tags
          - source_path: Full path to the document
          - chunk_count: Number of chunks in the vector store
          - total_chars: Total character count across all chunks
          - pages: List of page numbers (if available)
          
    Example:
        >>> result = await get_document_summary(
        ...     doc_id="example.pdf",
        ...     collection="default"
        ... )
        >>> print(result["content"][0]["text"])  # Markdown output
        >>> print(result["structuredContent"]["title"])  # Document title
    """
    provider = _get_summary_provider()
    return await provider.get_summary(doc_id, collection)


def reset_provider():
    """Reset the global summary provider (useful for testing)."""
    global _summary_provider, _settings
    _summary_provider = None
    _settings = None


# Tool schema for MCP registration
TOOL_NAME = "get_document_summary"
TOOL_DESCRIPTION = (
    "Get summary information for a specific document including title, summary, "
    "tags, chunk count, and total character count. Use this to get document metadata."
)
TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "doc_id": {
            "type": "string",
            "description": "Document ID or filename (e.g., 'example.pdf')"
        },
        "collection": {
            "type": "string",
            "description": "Document collection name (default: 'default')",
            "default": "default"
        }
    },
    "required": ["doc_id"]
}
