#!/usr/bin/env python3
"""
List Collections Tool - List available document collections with statistics.

This module implements the list_collections tool that scans the document directory
and returns collection information including document counts, chunk counts, and image counts.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

from core.settings import Settings, load_settings
from libs.vector_store.chroma_store import ChromaStore
from ingestion.storage.image_storage import ImageStorage

logger = logging.getLogger(__name__)


class CollectionsLister:
    """
    Collections lister that gathers statistics from multiple sources.
    
    This class coordinates data collection from:
    - Document directory (data/documents/)
    - ChromaDB vector store (chunk counts)
    - Image storage (image counts)
    
    Attributes:
        settings: Application settings
        chroma_store: ChromaDB vector store instance
        image_storage: Image storage instance
        document_dir: Path to document storage directory
    """
    
    def __init__(
        self,
        settings: Settings,
        chroma_store: Optional[ChromaStore] = None,
        image_storage: Optional[ImageStorage] = None,
    ):
        """
        Initialize CollectionsLister.
        
        Args:
            settings: Application settings
            chroma_store: Optional ChromaStore instance
            image_storage: Optional ImageStorage instance
        """
        self.settings = settings
        
        # Initialize or use provided components
        self.chroma_store = chroma_store or self._create_chroma_store()
        self.image_storage = image_storage or self._create_image_storage()
        
        # Get document directory from settings
        self.document_dir = Path(
            getattr(settings, 'storage', {}).get('document_dir', 'data/documents')
            if hasattr(settings, 'storage') and isinstance(settings.storage, dict)
            else 'data/documents'
        )
        
        logger.info(
            f"CollectionsLister initialized: document_dir={self.document_dir}"
        )
    
    def _create_chroma_store(self) -> ChromaStore:
        """Create ChromaStore instance from settings."""
        provider = self.settings.vector_store.provider
        if provider != "chroma":
            raise ValueError(f"list_collections only supports chroma provider, got: {provider}")
        
        config = getattr(self.settings.vector_store, provider, {})
        return ChromaStore(config)
    
    def _create_image_storage(self) -> ImageStorage:
        """Create ImageStorage instance from settings."""
        # Get image storage path from settings
        if hasattr(self.settings, 'ingestion') and hasattr(self.settings.ingestion, 'images'):
            storage_path = self.settings.ingestion.images.get('storage_path', 'data/images')
        else:
            storage_path = 'data/images'
        
        return ImageStorage(base_path=storage_path)
    
    def _get_collections_from_directory(self) -> list[str]:
        """
        Scan document directory to find collections.
        
        Returns:
            List of collection names found in data/documents/
        """
        collections = []
        
        if not self.document_dir.exists():
            logger.warning(f"Document directory does not exist: {self.document_dir}")
            return collections
        
        # Each subdirectory is a collection
        for item in self.document_dir.iterdir():
            if item.is_dir():
                collections.append(item.name)
        
        return sorted(collections)
    
    def _count_documents_in_collection(self, collection_name: str) -> int:
        """
        Count documents in a collection directory.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Number of documents (files) in the collection
        """
        collection_dir = self.document_dir / collection_name
        
        if not collection_dir.exists():
            return 0
        
        # Count files (excluding directories)
        count = 0
        for item in collection_dir.rglob("*"):
            if item.is_file():
                count += 1
        
        return count
    
    def _get_collection_chunk_count(self, collection_name: str) -> int:
        """
        Get chunk count for a collection from ChromaDB.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Number of chunks (vectors) in the collection
        """
        try:
            stats = self.chroma_store.get_collection_stats(collection_name)
            return stats.get("count", 0) if stats.get("exists", False) else 0
        except Exception as e:
            logger.warning(f"Failed to get chunk count for {collection_name}: {e}")
            return 0
    
    def _get_collection_image_count(self, collection_name: str) -> int:
        """
        Get image count for a collection from ImageStorage.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Number of images in the collection
        """
        try:
            images = self.image_storage.list_images(collection=collection_name)
            return len(images)
        except Exception as e:
            logger.warning(f"Failed to get image count for {collection_name}: {e}")
            return 0
    
    async def list_collections(self) -> dict[str, Any]:
        """
        List all collections with their statistics.
        
        Returns:
            MCP-formatted response with collection information:
            - content[0]: Human-readable Markdown with collection list
            - structuredContent.collections: List of collection objects with:
              - name: Collection name
              - documents: Number of documents
              - chunks: Number of chunks (vectors)
              - images: Number of images
        """
        try:
            # Get collections from directory
            collection_names = self._get_collections_from_directory()
            
            # Also check ChromaDB for collections that might not have documents yet
            try:
                chroma_collections = self.chroma_store.list_collections()
                for coll_name in chroma_collections:
                    if coll_name not in collection_names:
                        collection_names.append(coll_name)
                collection_names = sorted(collection_names)
            except Exception as e:
                logger.warning(f"Failed to list ChromaDB collections: {e}")
            
            # If no collections found, return friendly message
            if not collection_names:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": "## 知识库集合\n\n暂无集合。请先添加文档到知识库。\n\n提示：使用文档上传功能添加文档到 `data/documents/{collection}/` 目录。"
                        }
                    ],
                    "structuredContent": {
                        "collections": [],
                        "total_collections": 0
                    }
                }
            
            # Gather statistics for each collection
            collections = []
            for name in collection_names:
                doc_count = self._count_documents_in_collection(name)
                chunk_count = self._get_collection_chunk_count(name)
                image_count = self._get_collection_image_count(name)
                
                collections.append({
                    "name": name,
                    "documents": doc_count,
                    "chunks": chunk_count,
                    "images": image_count
                })
            
            # Build human-readable text
            lines = ["## 知识库集合\n"]
            for i, coll in enumerate(collections, 1):
                lines.append(
                    f"{i}. **{coll['name']}** - "
                    f"{coll['documents']}个文档, "
                    f"{coll['chunks']}个chunks, "
                    f"{coll['images']}张图片"
                )
            
            text_content = "\n".join(lines)
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": text_content
                    }
                ],
                "structuredContent": {
                    "collections": collections,
                    "total_collections": len(collections)
                }
            }
            
        except Exception as e:
            logger.error(f"List collections failed: {e}", exc_info=True)
            
            # Return error response
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"获取集合列表失败: {str(e)}"
                    }
                ],
                "structuredContent": {
                    "error": str(e),
                    "collections": []
                },
                "isError": True
            }


# Global lister instance (initialized lazily)
_collections_lister: Optional[CollectionsLister] = None
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


def _get_collections_lister() -> CollectionsLister:
    """Get or initialize collections lister."""
    global _collections_lister
    if _collections_lister is None:
        settings = _get_settings()
        _collections_lister = CollectionsLister(settings)
    return _collections_lister


async def list_collections() -> dict[str, Any]:
    """
    List all available document collections in the knowledge hub.
    
    Returns information about each collection including:
    - Number of documents
    - Number of chunks (vectors)
    - Number of images
    
    Returns:
        MCP-formatted response containing:
        - content[0]: Human-readable Markdown with collection list
        - structuredContent.collections: List of collection objects with statistics
        
    Example:
        >>> result = await list_collections()
        >>> print(result["content"][0]["text"])  # Markdown output
        >>> print(result["structuredContent"]["collections"])  # Structured data
    """
    lister = _get_collections_lister()
    return await lister.list_collections()


# Tool schema for MCP registration
TOOL_NAME = "list_collections"
TOOL_DESCRIPTION = (
    "List all available document collections in the knowledge hub. "
    "Returns collection names with document counts, chunk counts, and image counts."
)
TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": []
}


def reset_lister():
    """Reset the global collections lister (useful for testing)."""
    global _collections_lister, _settings
    _collections_lister = None
    _settings = None
