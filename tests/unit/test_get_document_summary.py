#!/usr/bin/env python3
"""
Get Document Summary Tool Tests

Tests the get_document_summary tool for retrieving document metadata and statistics.
"""

import pytest
from unittest.mock import MagicMock, patch

from mcp_server.tools.get_document_summary import (
    get_document_summary,
    DocumentSummaryProvider,
    reset_provider,
    TOOL_NAME,
    TOOL_DESCRIPTION,
)
from core.settings import Settings


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test."""
    reset_provider()
    yield
    reset_provider()


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
    return settings


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        {
            "id": "chunk_1",
            "text": "This is the first chunk of the document.",
            "metadata": {
                "source": "example.pdf",
                "title": "Example Technical Document",
                "summary": "A comprehensive guide to system architecture.",
                "tags": ["architecture", "design", "technical"],
                "page": 1
            }
        },
        {
            "id": "chunk_2",
            "text": "This is the second chunk with more content.",
            "metadata": {
                "source": "example.pdf",
                "title": "Example Technical Document",
                "summary": "A comprehensive guide to system architecture.",
                "tags": ["architecture", "design", "technical"],
                "page": 1
            }
        },
        {
            "id": "chunk_3",
            "text": "Third chunk on a different page.",
            "metadata": {
                "source": "example.pdf",
                "title": "Example Technical Document",
                "summary": "A comprehensive guide to system architecture.",
                "tags": ["architecture", "design", "technical"],
                "page": 2
            }
        }
    ]


class TestDocumentSummaryProviderInit:
    """Test DocumentSummaryProvider initialization."""
    
    def test_initialization_with_mock_components(self, mock_settings):
        """Test initialization with mocked components."""
        mock_chroma = MagicMock()
        
        provider = DocumentSummaryProvider(
            settings=mock_settings,
            chroma_store=mock_chroma
        )
        
        assert provider.settings == mock_settings
        assert provider.chroma_store == mock_chroma
    
    def test_initialization_unsupported_provider(self, mock_settings):
        """Test initialization with unsupported provider."""
        mock_settings.vector_store.provider = "qdrant"
        
        with pytest.raises(ValueError, match="only supports chroma provider"):
            DocumentSummaryProvider(settings=mock_settings)


class TestGetDocumentChunks:
    """Test getting document chunks from ChromaDB."""
    
    def test_get_chunks_exact_match(self, mock_settings, sample_chunks):
        """Test getting chunks with exact source match."""
        mock_chroma = MagicMock()
        mock_chroma.get_by_metadata.return_value = sample_chunks
        
        provider = DocumentSummaryProvider(
            settings=mock_settings,
            chroma_store=mock_chroma
        )
        
        chunks = provider._get_document_chunks("example.pdf")
        
        assert len(chunks) == 3
        mock_chroma.get_by_metadata.assert_called_with({"source": "example.pdf"})
    
    def test_get_chunks_basename_match(self, mock_settings, sample_chunks):
        """Test getting chunks with basename match."""
        mock_chroma = MagicMock()
        # First call returns empty, second call returns chunks
        mock_chroma.get_by_metadata.side_effect = [[], sample_chunks]
        
        provider = DocumentSummaryProvider(
            settings=mock_settings,
            chroma_store=mock_chroma
        )
        
        chunks = provider._get_document_chunks("data/documents/default/example.pdf")
        
        assert len(chunks) == 3
        # Should try with basename
        assert mock_chroma.get_by_metadata.call_count == 2


class TestAggregateMetadata:
    """Test metadata aggregation from chunks."""
    
    def test_aggregate_basic_metadata(self, mock_settings, sample_chunks):
        """Test basic metadata aggregation."""
        mock_chroma = MagicMock()
        provider = DocumentSummaryProvider(
            settings=mock_settings,
            chroma_store=mock_chroma
        )
        
        metadata = provider._aggregate_metadata(sample_chunks)
        
        assert metadata["title"] == "Example Technical Document"
        assert metadata["summary"] == "A comprehensive guide to system architecture."
        assert metadata["tags"] == ["architecture", "design", "technical"]
        assert metadata["source"] == "example.pdf"
        assert sorted(metadata["pages"]) == [1, 2]
    
    def test_aggregate_empty_chunks(self, mock_settings):
        """Test aggregation with empty chunks."""
        mock_chroma = MagicMock()
        provider = DocumentSummaryProvider(
            settings=mock_settings,
            chroma_store=mock_chroma
        )
        
        metadata = provider._aggregate_metadata([])
        
        assert metadata == {}
    
    def test_aggregate_single_page(self, mock_settings):
        """Test aggregation with single page."""
        mock_chroma = MagicMock()
        provider = DocumentSummaryProvider(
            settings=mock_settings,
            chroma_store=mock_chroma
        )
        
        chunks = [
            {
                "id": "chunk_1",
                "text": "Content",
                "metadata": {"page": 5, "source": "doc.pdf"}
            }
        ]
        
        metadata = provider._aggregate_metadata(chunks)
        
        assert metadata["pages"] == [5]


class TestCalculateStatistics:
    """Test statistics calculation."""
    
    def test_calculate_stats(self, mock_settings, sample_chunks):
        """Test statistics calculation."""
        mock_chroma = MagicMock()
        provider = DocumentSummaryProvider(
            settings=mock_settings,
            chroma_store=mock_chroma
        )
        
        stats = provider._calculate_statistics(sample_chunks)
        
        assert stats["chunk_count"] == 3
        # Total chars: 41 + 43 + 32 = 116
        assert stats["total_chars"] == 115
    
    def test_calculate_empty_stats(self, mock_settings):
        """Test statistics with empty chunks."""
        mock_chroma = MagicMock()
        provider = DocumentSummaryProvider(
            settings=mock_settings,
            chroma_store=mock_chroma
        )
        
        stats = provider._calculate_statistics([])
        
        assert stats["chunk_count"] == 0
        assert stats["total_chars"] == 0


class TestGetSummaryAsync:
    """Test async get_summary method."""
    
    @pytest.mark.asyncio
    async def test_successful_summary(self, mock_settings, sample_chunks):
        """Test successful document summary retrieval."""
        mock_chroma = MagicMock()
        mock_chroma.get_by_metadata.return_value = sample_chunks
        
        provider = DocumentSummaryProvider(
            settings=mock_settings,
            chroma_store=mock_chroma
        )
        
        result = await provider.get_summary("example.pdf", "default")
        
        assert "content" in result
        assert "structuredContent" in result
        assert "isError" not in result
        
        # Check structured content
        sc = result["structuredContent"]
        assert sc["doc_id"] == "example.pdf"
        assert sc["title"] == "Example Technical Document"
        assert sc["summary"] == "A comprehensive guide to system architecture."
        assert sc["tags"] == ["architecture", "design", "technical"]
        assert sc["chunk_count"] == 3
        assert sc["total_chars"] == 115
        assert sorted(sc["pages"]) == [1, 2]
        
        # Check text content
        text = result["content"][0]["text"]
        assert "文档摘要" in text
        assert "Example Technical Document" in text
        assert "3个chunks" in text
    
    @pytest.mark.asyncio
    async def test_document_not_found(self, mock_settings):
        """Test handling of non-existent document."""
        mock_chroma = MagicMock()
        mock_chroma.get_by_metadata.return_value = []
        
        provider = DocumentSummaryProvider(
            settings=mock_settings,
            chroma_store=mock_chroma
        )
        
        result = await provider.get_summary("nonexistent.pdf", "default")
        
        assert "isError" in result
        assert result["isError"] is True
        assert "structuredContent" in result
        assert result["structuredContent"]["error"] == "Document not found"
        
        # Check error message in content
        text = result["content"][0]["text"]
        assert "文档未找到" in text
    
    @pytest.mark.asyncio
    async def test_empty_doc_id(self, mock_settings):
        """Test handling of empty doc_id."""
        mock_chroma = MagicMock()
        provider = DocumentSummaryProvider(
            settings=mock_settings,
            chroma_store=mock_chroma
        )
        
        result = await provider.get_summary("", "default")
        
        assert "isError" in result
        assert result["isError"] is True
        assert result["structuredContent"]["error"] == "Document ID is required"
    
    @pytest.mark.asyncio
    async def test_missing_metadata_fields(self, mock_settings):
        """Test handling of chunks with missing metadata fields."""
        chunks = [
            {
                "id": "chunk_1",
                "text": "Content without full metadata.",
                "metadata": {
                    "source": "minimal.pdf"
                    # No title, summary, or tags
                }
            }
        ]
        
        mock_chroma = MagicMock()
        mock_chroma.get_by_metadata.return_value = chunks
        
        provider = DocumentSummaryProvider(
            settings=mock_settings,
            chroma_store=mock_chroma
        )
        
        result = await provider.get_summary("minimal.pdf", "default")
        
        assert "isError" not in result
        sc = result["structuredContent"]
        assert sc["title"] == ""
        assert sc["summary"] == ""
        assert sc["tags"] == []
        assert sc["chunk_count"] == 1


class TestGetDocumentSummaryTool:
    """Test the get_document_summary tool function."""
    
    @pytest.mark.asyncio
    async def test_tool_with_mocked_provider(self, mock_settings, sample_chunks):
        """Test tool function with mocked provider."""
        mock_chroma = MagicMock()
        mock_chroma.get_by_metadata.return_value = sample_chunks
        
        with patch(
            "mcp_server.tools.get_document_summary._get_settings",
            return_value=mock_settings
        ):
            with patch(
                "mcp_server.tools.get_document_summary.ChromaStore",
                return_value=mock_chroma
            ):
                result = await get_document_summary("example.pdf", "default")
        
        assert "content" in result
        assert "structuredContent" in result
        assert result["structuredContent"]["doc_id"] == "example.pdf"


class TestToolSchema:
    """Test tool schema constants."""
    
    def test_tool_name(self):
        """Test tool name is correct."""
        from mcp_server.tools.get_document_summary import TOOL_NAME
        assert TOOL_NAME == "get_document_summary"
    
    def test_tool_description(self):
        """Test tool description exists."""
        from mcp_server.tools.get_document_summary import TOOL_DESCRIPTION
        assert TOOL_DESCRIPTION is not None
        assert len(TOOL_DESCRIPTION) > 0
        assert "summary" in TOOL_DESCRIPTION.lower()
    
    def test_tool_input_schema(self):
        """Test tool input schema structure."""
        from mcp_server.tools.get_document_summary import TOOL_INPUT_SCHEMA
        
        assert TOOL_INPUT_SCHEMA["type"] == "object"
        assert "doc_id" in TOOL_INPUT_SCHEMA["properties"]
        assert "collection" in TOOL_INPUT_SCHEMA["properties"]
        assert TOOL_INPUT_SCHEMA["required"] == ["doc_id"]
        
        # Check doc_id property
        doc_id_prop = TOOL_INPUT_SCHEMA["properties"]["doc_id"]
        assert doc_id_prop["type"] == "string"
        
        # Check collection property has default
        collection_prop = TOOL_INPUT_SCHEMA["properties"]["collection"]
        assert collection_prop["type"] == "string"
        assert collection_prop["default"] == "default"
