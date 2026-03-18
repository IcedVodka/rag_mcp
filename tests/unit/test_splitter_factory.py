#!/usr/bin/env python3
"""
Splitter Factory Tests

Tests for splitter factory routing and strategy creation.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add src to path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from core.settings import Settings, SplitterSettings
from libs.splitter.splitter_factory import SplitterFactory, SplitterProvider
from libs.splitter.base_splitter import BaseSplitter


class FakeSplitter(BaseSplitter):
    """Fake splitter for testing."""
    
    def split_text(self, text, trace=None):
        # Simple split by character count
        size = self.chunk_size
        overlap = self.chunk_overlap
        
        if not text:
            return []
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - overlap
        
        return chunks


class TestSplitterFactory:
    """Test splitter factory routing logic."""
    
    def test_factory_routes_to_recursive(self):
        """Test factory routes to recursive strategy."""
        settings = Mock(spec=Settings)
        settings.splitter = Mock(spec=SplitterSettings)
        settings.splitter.strategy = "recursive"
        settings.splitter.recursive = {"chunk_size": 1000, "chunk_overlap": 200}
        
        with patch.dict('sys.modules', {'libs.splitter.recursive_splitter': Mock(RecursiveSplitter=FakeSplitter)}):
            splitter = SplitterFactory.create(settings)
            assert isinstance(splitter, FakeSplitter)
    
    def test_factory_routes_to_semantic(self):
        """Test factory routes to semantic strategy."""
        settings = Mock(spec=Settings)
        settings.splitter = Mock(spec=SplitterSettings)
        settings.splitter.strategy = "semantic"
        settings.splitter.semantic = {"chunk_size": 500}
        
        with patch.dict('sys.modules', {'libs.splitter.semantic_splitter': Mock(SemanticSplitter=FakeSplitter)}):
            splitter = SplitterFactory.create(settings)
            assert isinstance(splitter, FakeSplitter)
    
    def test_factory_routes_to_fixed(self):
        """Test factory routes to fixed strategy."""
        settings = Mock(spec=Settings)
        settings.splitter = Mock(spec=SplitterSettings)
        settings.splitter.strategy = "fixed"
        settings.splitter.fixed = {"chunk_size": 200}
        
        with patch.dict('sys.modules', {'libs.splitter.fixed_length_splitter': Mock(FixedLengthSplitter=FakeSplitter)}):
            splitter = SplitterFactory.create(settings)
            assert isinstance(splitter, FakeSplitter)
    
    def test_factory_raises_on_unknown_strategy(self):
        """Test factory raises error for unknown strategy."""
        settings = Mock(spec=Settings)
        settings.splitter = Mock(spec=SplitterSettings)
        settings.splitter.strategy = "unknown_strategy"
        
        with pytest.raises(ValueError) as exc_info:
            SplitterFactory.create(settings)
        
        assert "unknown_strategy" in str(exc_info.value).lower()


class TestFakeSplitter:
    """Test fake splitter behavior."""
    
    def test_split_text_basic(self):
        """Test basic text splitting."""
        splitter = FakeSplitter({"chunk_size": 10, "chunk_overlap": 2})
        
        text = "Hello World! This is a test."
        chunks = splitter.split_text(text)
        
        assert len(chunks) > 0
        assert all(len(c) <= 10 for c in chunks)
    
    def test_split_text_empty(self):
        """Test splitting empty text."""
        splitter = FakeSplitter({"chunk_size": 10, "chunk_overlap": 2})
        
        chunks = splitter.split_text("")
        
        assert chunks == []
    
    def test_split_documents(self):
        """Test splitting multiple documents."""
        splitter = FakeSplitter({"chunk_size": 10, "chunk_overlap": 2})
        
        docs = ["Hello World!", "Another document."]
        results = splitter.split_documents(docs)
        
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)


class TestSplitterProvider:
    """Test splitter provider wrapper."""
    
    def test_split_with_trace(self):
        """Test provider wrapper with trace."""
        splitter = FakeSplitter({"chunk_size": 20, "chunk_overlap": 5})
        provider = SplitterProvider(splitter)
        
        from core.trace.trace_context import TraceContext
        trace = TraceContext(trace_type="test")
        
        chunks = provider.split_with_trace("Hello World! This is a test message.", trace=trace)
        
        assert len(chunks) > 0
        assert len(trace.stages) == 1
        assert trace.stages[0]["name"] == "text_splitting"
        assert trace.stages[0]["details"]["chunk_count"] == len(chunks)
    
    def test_split_with_trace_none(self):
        """Test provider wrapper without trace."""
        splitter = FakeSplitter({"chunk_size": 20, "chunk_overlap": 5})
        provider = SplitterProvider(splitter)
        
        chunks = provider.split_with_trace("Hello World!", trace=None)
        
        assert len(chunks) > 0
