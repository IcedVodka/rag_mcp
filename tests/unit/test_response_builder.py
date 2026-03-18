#!/usr/bin/env python3
"""
Unit tests for ResponseBuilder and CitationGenerator.
"""

import pytest
from typing import Any

from core.types import RetrievalResult
from core.response.citation_generator import (
    Citation,
    CitationGenerator,
    format_citation_markdown,
    format_inline_citation,
)
from core.response.response_builder import (
    ResponseBuilder,
    build_simple_response,
)


class TestCitation:
    """Test Citation dataclass."""
    
    def test_citation_creation(self):
        """Test creating a Citation object."""
        citation = Citation(
            id=1,
            source="test.pdf",
            page=5,
            chunk_id="chunk_001",
            text="Test content",
            score=0.95,
            metadata={"title": "Test Doc"}
        )
        
        assert citation.id == 1
        assert citation.source == "test.pdf"
        assert citation.page == 5
        assert citation.chunk_id == "chunk_001"
        assert citation.score == 0.95
    
    def test_citation_to_dict(self):
        """Test Citation.to_dict() method."""
        citation = Citation(
            id=1,
            source="test.pdf",
            page=5,
            chunk_id="chunk_001",
            text="Test content that might be long" * 10,
            score=0.951234,
            metadata={"title": "Test Doc", "doc_type": "pdf", "collection": "docs"}
        )
        
        result = citation.to_dict()
        
        assert result["id"] == 1
        assert result["source"] == "test.pdf"
        assert result["page"] == 5
        assert result["chunk_id"] == "chunk_001"
        assert result["score"] == 0.9512  # Rounded to 4 decimals
        assert len(result["text"]) <= 500  # Truncated
        assert result["title"] == "Test Doc"
        assert result["doc_type"] == "pdf"
        assert result["collection"] == "docs"


class TestCitationGenerator:
    """Test CitationGenerator class."""
    
    @pytest.fixture
    def generator(self):
        return CitationGenerator()
    
    @pytest.fixture
    def sample_results(self) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id="chunk_001",
                score=0.95,
                text="First result content",
                metadata={"source_path": "/docs/test.pdf", "page": 5, "title": "Test Doc"}
            ),
            RetrievalResult(
                chunk_id="chunk_002",
                score=0.85,
                text="Second result content",
                metadata={"source_path": "/docs/other.md", "page": 10}
            ),
            RetrievalResult(
                chunk_id="chunk_003",
                score=0.75,
                text="Third result with no page",
                metadata={"source": "plain.txt"}
            ),
        ]
    
    def test_generate_citations(self, generator, sample_results):
        """Test generating citations from results."""
        citations = generator.generate(sample_results)
        
        assert len(citations) == 3
        
        # Check sequential IDs
        assert citations[0].id == 1
        assert citations[1].id == 2
        assert citations[2].id == 3
        
        # Check source extraction
        assert citations[0].source == "test.pdf"  # Extracted from path
        assert citations[1].source == "other.md"
        assert citations[2].source == "plain.txt"
        
        # Check page extraction
        assert citations[0].page == 5
        assert citations[1].page == 10
        assert citations[2].page is None
    
    def test_generate_empty_results(self, generator):
        """Test generating citations from empty results."""
        citations = generator.generate([])
        assert citations == []
    
    def test_extract_source_from_various_fields(self, generator):
        """Test source extraction handles various metadata fields."""
        test_cases = [
            ({"source_path": "/path/to/file.pdf"}, "file.pdf"),
            ({"source": "document.md"}, "document.md"),
            ({"file_path": "/a/b/c.txt"}, "c.txt"),
            ({"file": "test.doc"}, "test.doc"),
            ({"doc_id": "doc123"}, "doc123"),
            ({}, "unknown"),
        ]
        
        for metadata, expected in test_cases:
            result = RetrievalResult(
                chunk_id="test",
                score=0.5,
                text="test",
                metadata=metadata
            )
            citation = generator._create_citation(1, result)
            assert citation.source == expected
    
    def test_extract_page_from_various_fields(self, generator):
        """Test page extraction handles various metadata fields."""
        test_cases = [
            ({"page": 5}, 5),
            ({"page_num": 10}, 10),
            ({"page_number": "15"}, 15),  # String number
            ({"slide": 3}, 3),
            ({}, None),
            ({"page": None}, None),
        ]
        
        for metadata, expected in test_cases:
            result = RetrievalResult(
                chunk_id="test",
                score=0.5,
                text="test",
                metadata=metadata
            )
            citation = generator._create_citation(1, result)
            assert citation.page == expected


class TestCitationFormatting:
    """Test citation formatting functions."""
    
    def test_format_inline_citation(self):
        """Test inline citation formatting."""
        assert format_inline_citation(1) == "[1]"
        assert format_inline_citation(10) == "[10]"
        assert format_inline_citation(99) == "[99]"
    
    def test_format_citation_markdown(self):
        """Test Markdown citation list formatting."""
        citations = [
            Citation(id=1, source="doc1.pdf", page=5, chunk_id="c1", text="text1", score=0.95),
            Citation(id=2, source="doc2.md", page=None, chunk_id="c2", text="text2", score=0.85),
        ]
        
        markdown = format_citation_markdown(citations)
        
        assert "**References:**" in markdown
        assert "[1] **doc1.pdf**, p.5" in markdown
        assert "(score: 0.950)" in markdown
        assert "[2] **doc2.md**" in markdown
    
    def test_format_citation_markdown_empty(self):
        """Test Markdown formatting with empty list."""
        assert format_citation_markdown([]) == ""


class TestResponseBuilder:
    """Test ResponseBuilder class."""
    
    @pytest.fixture
    def builder(self):
        return ResponseBuilder()
    
    @pytest.fixture
    def sample_results(self) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id="chunk_001",
                score=0.95,
                text="This is the first result about Python programming.",
                metadata={"source_path": "/docs/python.pdf", "page": 10, "title": "Python Guide"}
            ),
            RetrievalResult(
                chunk_id="chunk_002",
                score=0.85,
                text="This is the second result about code examples.",
                metadata={"source_path": "/docs/code.md", "page": 5}
            ),
        ]
    
    def test_build_response_structure(self, builder, sample_results):
        """Test that build() returns correct MCP response structure."""
        response = builder.build(sample_results, "python programming")
        
        # Check top-level structure
        assert "content" in response
        assert "structuredContent" in response
        assert "isError" in response
        assert response["isError"] is False
        
        # Check content array
        assert len(response["content"]) == 1
        assert response["content"][0]["type"] == "text"
        assert "text" in response["content"][0]
    
    def test_build_response_markdown_content(self, builder, sample_results):
        """Test Markdown content generation."""
        response = builder.build(sample_results, "python programming")
        
        text = response["content"][0]["text"]
        
        # Check for expected elements
        assert "Found 2 relevant result(s)" in text
        assert 'python programming' in text
        assert "[1]" in text  # Inline citation
        assert "[2]" in text
        assert "**References:**" in text
        assert "python.pdf" in text
        assert "code.md" in text
    
    def test_build_response_structured_content(self, builder, sample_results):
        """Test structured content generation."""
        response = builder.build(sample_results, "python programming")
        
        structured = response["structuredContent"]
        
        assert structured["query"] == "python programming"
        assert structured["result_count"] == 2
        assert len(structured["citations"]) == 2
        
        # Check citation structure
        citation = structured["citations"][0]
        assert "id" in citation
        assert "source" in citation
        assert "page" in citation
        assert "chunk_id" in citation
        assert "score" in citation
    
    def test_build_empty_response(self, builder):
        """Test response when no results found."""
        response = builder.build([], "unknown query")
        
        assert response["isError"] is False
        assert "content" in response
        assert "structuredContent" in response
        
        # Check text content
        text = response["content"][0]["text"]
        assert "couldn't find any relevant information" in text
        assert "unknown query" in text
        assert "Suggestions:" in text
        
        # Check structured content
        structured = response["structuredContent"]
        assert structured["result_count"] == 0
        assert structured["citations"] == []
    
    def test_build_without_references(self, builder, sample_results):
        """Test building response without reference section."""
        response = builder.build(sample_results, "query", include_references=False)
        
        text = response["content"][0]["text"]
        
        # Should not contain reference section
        assert "**References:**" not in text


class TestBuildSimpleResponse:
    """Test build_simple_response helper."""
    
    def test_simple_response(self):
        """Test creating a simple response."""
        response = build_simple_response("Hello world")
        
        assert response["isError"] is False
        assert len(response["content"]) == 1
        assert response["content"][0]["type"] == "text"
        assert response["content"][0]["text"] == "Hello world"
    
    def test_simple_error_response(self):
        """Test creating a simple error response."""
        response = build_simple_response("Error occurred", is_error=True)
        
        assert response["isError"] is True
        assert response["content"][0]["text"] == "Error occurred"


class TestResponseBuilderEdgeCases:
    """Test edge cases for ResponseBuilder."""
    
    def test_long_text_truncation(self):
        """Test that very long text is truncated."""
        builder = ResponseBuilder()
        
        long_text = "A" * 2000
        results = [RetrievalResult(
            chunk_id="c1",
            score=0.9,
            text=long_text,
            metadata={"source": "test.pdf"}
        )]
        
        response = builder.build(results, "query")
        text = response["content"][0]["text"]
        
        # Text should be truncated with ellipsis
        assert "..." in text
    
    def test_special_characters_in_query(self):
        """Test handling of special characters in query."""
        builder = ResponseBuilder()
        
        results = [RetrievalResult(
            chunk_id="c1",
            score=0.9,
            text="Result",
            metadata={"source": "test.pdf"}
        )]
        
        # Query with special characters
        query = 'Test "quoted" <script>alert(1)</script>'
        response = builder.build(results, query)
        
        # Should not raise and should include query
        assert response["structuredContent"]["query"] == query
