#!/usr/bin/env python3
"""
List Collections Tool Tests

Tests the list_collections tool for listing document collections with statistics.
"""

import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

from mcp_server.tools.list_collections import (
    list_collections,
    CollectionsLister,
    reset_lister,
    TOOL_NAME,
    TOOL_DESCRIPTION,
)
from core.settings import Settings


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test."""
    reset_lister()
    yield
    reset_lister()


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock(spec=Settings)
    settings.vector_store = MagicMock()
    settings.vector_store.provider = "chroma"
    settings.vector_store.chroma = {
        "persist_directory": "data/db/chroma",
        "collection_name": "default",
        "distance_function": "cosine"
    }
    settings.storage = {"document_dir": "data/documents"}
    settings.ingestion = MagicMock()
    settings.ingestion.images = {"storage_path": "data/images"}
    return settings


class TestCollectionsListerInit:
    """Test CollectionsLister initialization."""
    
    def test_initialization_with_mock_components(self, mock_settings):
        """Test initialization with mocked components."""
        mock_chroma = MagicMock()
        mock_image_storage = MagicMock()
        
        lister = CollectionsLister(
            settings=mock_settings,
            chroma_store=mock_chroma,
            image_storage=mock_image_storage
        )
        
        assert lister.settings == mock_settings
        assert lister.chroma_store == mock_chroma
        assert lister.image_storage == mock_image_storage


class TestGetCollectionsFromDirectory:
    """Test scanning document directory for collections."""
    
    def test_empty_directory(self, mock_settings, tmp_path):
        """Test with empty document directory."""
        mock_chroma = MagicMock()
        mock_image_storage = MagicMock()
        
        lister = CollectionsLister(
            settings=mock_settings,
            chroma_store=mock_chroma,
            image_storage=mock_image_storage
        )
        lister.document_dir = tmp_path
        
        collections = lister._get_collections_from_directory()
        assert collections == []
    
    def test_multiple_collections(self, mock_settings, tmp_path):
        """Test with multiple collection directories."""
        mock_chroma = MagicMock()
        mock_image_storage = MagicMock()
        
        # Create collection directories
        (tmp_path / "default").mkdir()
        (tmp_path / "tech_docs").mkdir()
        (tmp_path / "legal").mkdir()
        
        lister = CollectionsLister(
            settings=mock_settings,
            chroma_store=mock_chroma,
            image_storage=mock_image_storage
        )
        lister.document_dir = tmp_path
        
        collections = lister._get_collections_from_directory()
        assert collections == ["default", "legal", "tech_docs"]


class TestCountDocumentsInCollection:
    """Test counting documents in collections."""
    
    def test_count_documents(self, mock_settings, tmp_path):
        """Test counting documents in a collection."""
        mock_chroma = MagicMock()
        mock_image_storage = MagicMock()
        
        # Create collection with documents
        coll_dir = tmp_path / "default"
        coll_dir.mkdir()
        (coll_dir / "doc1.pdf").write_text("content1")
        (coll_dir / "doc2.txt").write_text("content2")
        
        lister = CollectionsLister(
            settings=mock_settings,
            chroma_store=mock_chroma,
            image_storage=mock_image_storage
        )
        lister.document_dir = tmp_path
        
        count = lister._count_documents_in_collection("default")
        assert count == 2
    
    def test_count_nested_documents(self, mock_settings, tmp_path):
        """Test counting documents in nested subdirectories."""
        mock_chroma = MagicMock()
        mock_image_storage = MagicMock()
        
        # Create nested structure
        coll_dir = tmp_path / "default"
        subdir = coll_dir / "subdir"
        subdir.mkdir(parents=True)
        (coll_dir / "doc1.pdf").write_text("content1")
        (subdir / "doc2.txt").write_text("content2")
        
        lister = CollectionsLister(
            settings=mock_settings,
            chroma_store=mock_chroma,
            image_storage=mock_image_storage
        )
        lister.document_dir = tmp_path
        
        count = lister._count_documents_in_collection("default")
        assert count == 2
    
    def test_nonexistent_collection(self, mock_settings, tmp_path):
        """Test counting documents in non-existent collection."""
        mock_chroma = MagicMock()
        mock_image_storage = MagicMock()
        
        lister = CollectionsLister(
            settings=mock_settings,
            chroma_store=mock_chroma,
            image_storage=mock_image_storage
        )
        lister.document_dir = tmp_path
        
        count = lister._count_documents_in_collection("nonexistent")
        assert count == 0


class TestGetCollectionStats:
    """Test getting collection statistics."""
    
    def test_get_chunk_count_existing(self, mock_settings):
        """Test getting chunk count for existing collection."""
        mock_chroma = MagicMock()
        mock_chroma.get_collection_stats.return_value = {"count": 128, "exists": True}
        mock_image_storage = MagicMock()
        
        lister = CollectionsLister(
            settings=mock_settings,
            chroma_store=mock_chroma,
            image_storage=mock_image_storage
        )
        
        count = lister._get_collection_chunk_count("default")
        assert count == 128
    
    def test_get_chunk_count_nonexistent(self, mock_settings):
        """Test getting chunk count for non-existent collection."""
        mock_chroma = MagicMock()
        mock_chroma.get_collection_stats.return_value = {"count": 0, "exists": False}
        mock_image_storage = MagicMock()
        
        lister = CollectionsLister(
            settings=mock_settings,
            chroma_store=mock_chroma,
            image_storage=mock_image_storage
        )
        
        count = lister._get_collection_chunk_count("nonexistent")
        assert count == 0
    
    def test_get_image_count(self, mock_settings):
        """Test getting image count for a collection."""
        mock_chroma = MagicMock()
        mock_image_storage = MagicMock()
        mock_image_storage.list_images.return_value = [
            MagicMock(), MagicMock(), MagicMock()  # 3 mock images
        ]
        
        lister = CollectionsLister(
            settings=mock_settings,
            chroma_store=mock_chroma,
            image_storage=mock_image_storage
        )
        
        count = lister._get_collection_image_count("default")
        assert count == 3


class TestListCollectionsAsync:
    """Test async list_collections method."""
    
    @pytest.mark.asyncio
    async def test_empty_collections(self, mock_settings, tmp_path):
        """Test listing when no collections exist."""
        mock_chroma = MagicMock()
        mock_chroma.list_collections.return_value = []
        mock_image_storage = MagicMock()
        
        lister = CollectionsLister(
            settings=mock_settings,
            chroma_store=mock_chroma,
            image_storage=mock_image_storage
        )
        lister.document_dir = tmp_path
        
        result = await lister.list_collections()
        
        assert "content" in result
        assert "structuredContent" in result
        assert result["structuredContent"]["total_collections"] == 0
        assert result["structuredContent"]["collections"] == []
        assert "暂无集合" in result["content"][0]["text"]
    
    @pytest.mark.asyncio
    async def test_multiple_collections_with_stats(self, mock_settings, tmp_path):
        """Test listing multiple collections with statistics."""
        mock_chroma = MagicMock()
        mock_chroma.list_collections.return_value = []
        mock_chroma.get_collection_stats.side_effect = lambda name: {
            "default": {"count": 128, "exists": True},
            "tech_docs": {"count": 64, "exists": True}
        }.get(name, {"count": 0, "exists": False})
        
        mock_image_storage = MagicMock()
        mock_image_storage.list_images.side_effect = lambda collection: {
            "default": [MagicMock()] * 10,
            "tech_docs": [MagicMock()] * 5
        }.get(collection, [])
        
        # Create collection directories with documents
        (tmp_path / "default").mkdir()
        (tmp_path / "default" / "doc1.pdf").write_text("content")
        (tmp_path / "default" / "doc2.pdf").write_text("content")
        (tmp_path / "tech_docs").mkdir()
        (tmp_path / "tech_docs" / "doc1.md").write_text("content")
        
        lister = CollectionsLister(
            settings=mock_settings,
            chroma_store=mock_chroma,
            image_storage=mock_image_storage
        )
        lister.document_dir = tmp_path
        
        result = await lister.list_collections()
        
        assert result["structuredContent"]["total_collections"] == 2
        
        collections = result["structuredContent"]["collections"]
        assert len(collections) == 2
        
        # Check default collection
        default_coll = next(c for c in collections if c["name"] == "default")
        assert default_coll["documents"] == 2
        assert default_coll["chunks"] == 128
        assert default_coll["images"] == 10
        
        # Check tech_docs collection
        tech_coll = next(c for c in collections if c["name"] == "tech_docs")
        assert tech_coll["documents"] == 1
        assert tech_coll["chunks"] == 64
        assert tech_coll["images"] == 5
        
        # Check markdown output
        text = result["content"][0]["text"]
        assert "## 知识库集合" in text
        assert "default" in text
        assert "tech_docs" in text
    
    @pytest.mark.asyncio
    async def test_collection_from_chroma_only(self, mock_settings, tmp_path):
        """Test listing collection that exists only in ChromaDB."""
        mock_chroma = MagicMock()
        # list_collections returns string names directly (simulating [c.name for c in collections])
        mock_chroma.list_collections.return_value = ["indexed_only"]
        mock_chroma.get_collection_stats.return_value = {"count": 50, "exists": True}
        mock_image_storage = MagicMock()
        mock_image_storage.list_images.return_value = []
        
        lister = CollectionsLister(
            settings=mock_settings,
            chroma_store=mock_chroma,
            image_storage=mock_image_storage
        )
        lister.document_dir = tmp_path
        
        result = await lister.list_collections()
        
        collections = result["structuredContent"]["collections"]
        assert len(collections) == 1
        assert collections[0]["name"] == "indexed_only"
        assert collections[0]["chunks"] == 50
        assert collections[0]["documents"] == 0  # No documents in directory


class TestListCollectionsTool:
    """Test the list_collections tool function."""
    
    @pytest.mark.asyncio
    async def test_tool_returns_correct_format(self):
        """Test that tool returns correctly formatted response."""
        # This test will use actual file system if data/documents exists
        # Otherwise it should handle gracefully
        result = await list_collections()
        
        assert "content" in result
        assert "structuredContent" in result
        assert isinstance(result["content"], list)
        assert result["content"][0]["type"] == "text"
        assert "text" in result["content"][0]
        assert "collections" in result["structuredContent"]


class TestToolSchema:
    """Test tool schema constants."""
    
    def test_tool_name(self):
        """Test tool name is correct."""
        assert TOOL_NAME == "list_collections"
    
    def test_tool_description(self):
        """Test tool description exists."""
        assert TOOL_DESCRIPTION is not None
        assert len(TOOL_DESCRIPTION) > 0
        assert "collection" in TOOL_DESCRIPTION.lower()
