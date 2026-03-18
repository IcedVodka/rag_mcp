#!/usr/bin/env python3
"""
Vector Store Factory Tests

Tests for vector store factory routing and provider creation.
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

from core.settings import Settings, VectorStoreSettings
from libs.vector_store.vector_store_factory import VectorStoreFactory, VectorStoreProvider
from libs.vector_store.base_vector_store import BaseVectorStore, VectorRecord, QueryResult


class FakeVectorStore(BaseVectorStore):
    """Fake vector store for testing."""
    
    def __init__(self, config):
        super().__init__(config)
        self.data = {}
    
    def upsert(self, records, trace=None):
        for record in records:
            self.data[record.id] = record
    
    def query(self, vector, top_k=10, filters=None, trace=None):
        # Return fake results
        results = []
        for i, (id_, record) in enumerate(list(self.data.items())[:top_k]):
            results.append(QueryResult(
                id=id_,
                score=1.0 - (i * 0.1),
                text=record.text,
                metadata=record.metadata
            ))
        return results
    
    def delete(self, ids, trace=None):
        for id_ in ids:
            self.data.pop(id_, None)
    
    def get_by_ids(self, ids, trace=None):
        results = []
        for id_ in ids:
            if id_ in self.data:
                record = self.data[id_]
                results.append({
                    "id": record.id,
                    "text": record.text,
                    "metadata": record.metadata
                })
        return results
    
    def count(self):
        return len(self.data)


class TestVectorStoreFactory:
    """Test vector store factory routing logic."""
    
    def test_factory_routes_to_chroma(self):
        """Test factory routes to Chroma provider."""
        settings = Mock(spec=Settings)
        settings.vector_store = Mock(spec=VectorStoreSettings)
        settings.vector_store.provider = "chroma"
        settings.vector_store.chroma = {"persist_directory": "data/db/chroma"}
        
        with patch.dict('sys.modules', {'libs.vector_store.chroma_store': Mock(ChromaStore=FakeVectorStore)}):
            store = VectorStoreFactory.create(settings)
            assert isinstance(store, FakeVectorStore)
    
    def test_factory_routes_to_qdrant(self):
        """Test factory routes to Qdrant provider."""
        settings = Mock(spec=Settings)
        settings.vector_store = Mock(spec=VectorStoreSettings)
        settings.vector_store.provider = "qdrant"
        settings.vector_store.qdrant = {"url": "http://localhost:6333"}
        
        with patch.dict('sys.modules', {'libs.vector_store.qdrant_store': Mock(QdrantStore=FakeVectorStore)}):
            store = VectorStoreFactory.create(settings)
            assert isinstance(store, FakeVectorStore)
    
    def test_factory_raises_on_unknown_provider(self):
        """Test factory raises error for unknown provider."""
        settings = Mock(spec=Settings)
        settings.vector_store = Mock(spec=VectorStoreSettings)
        settings.vector_store.provider = "unknown_provider"
        
        with pytest.raises(ValueError) as exc_info:
            VectorStoreFactory.create(settings)
        
        assert "unknown_provider" in str(exc_info.value).lower()


class TestFakeVectorStore:
    """Test fake vector store behavior."""
    
    def test_upsert_and_query(self):
        """Test basic upsert and query."""
        store = FakeVectorStore({"collection_name": "test"})
        
        records = [
            VectorRecord(id="1", vector=[0.1, 0.2], text="text1", metadata={}),
            VectorRecord(id="2", vector=[0.3, 0.4], text="text2", metadata={}),
        ]
        store.upsert(records)
        
        results = store.query([0.1, 0.2], top_k=2)
        
        assert len(results) == 2
        assert results[0].id == "1"
    
    def test_delete(self):
        """Test delete operation."""
        store = FakeVectorStore({"collection_name": "test"})
        
        records = [VectorRecord(id="1", vector=[0.1], text="text1", metadata={})]
        store.upsert(records)
        assert store.count() == 1
        
        store.delete(["1"])
        assert store.count() == 0
    
    def test_get_by_ids(self):
        """Test get by IDs."""
        store = FakeVectorStore({"collection_name": "test"})
        
        records = [
            VectorRecord(id="1", vector=[0.1], text="text1", metadata={"key": "val"}),
        ]
        store.upsert(records)
        
        results = store.get_by_ids(["1", "2"])
        
        assert len(results) == 1
        assert results[0]["id"] == "1"
        assert results[0]["text"] == "text1"


class TestVectorStoreProvider:
    """Test vector store provider wrapper."""
    
    def test_upsert_with_trace(self):
        """Test upsert with trace recording."""
        store = FakeVectorStore({"collection_name": "test"})
        provider = VectorStoreProvider(store)
        
        from core.trace.trace_context import TraceContext
        trace = TraceContext(trace_type="test")
        
        records = [VectorRecord(id="1", vector=[0.1], text="text", metadata={})]
        provider.upsert_with_trace(records, trace)
        
        assert len(trace.stages) == 1
        assert trace.stages[0]["name"] == "vector_upsert"
    
    def test_query_with_trace(self):
        """Test query with trace recording."""
        store = FakeVectorStore({"collection_name": "test"})
        provider = VectorStoreProvider(store)
        
        # Add some data first
        store.upsert([VectorRecord(id="1", vector=[0.1], text="text", metadata={})])
        
        from core.trace.trace_context import TraceContext
        trace = TraceContext(trace_type="test")
        
        results = provider.query_with_trace([0.1], top_k=1, trace=trace)
        
        assert len(results) == 1
        assert len(trace.stages) == 1
        assert trace.stages[0]["name"] == "vector_query"
