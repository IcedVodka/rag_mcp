#!/usr/bin/env python3
"""
Smoke Tests - Import Verification

Basic smoke tests to verify all key packages can be imported.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


class TestPackageImports:
    """Test that all key packages can be imported."""
    
    def test_import_mcp_server(self) -> None:
        """Test mcp_server package imports."""
        import mcp_server
        assert hasattr(mcp_server, 'main')
    
    def test_import_core(self) -> None:
        """Test core package imports."""
        import core
        assert core is not None
    
    def test_import_ingestion(self) -> None:
        """Test ingestion package imports."""
        import ingestion
        assert ingestion is not None
    
    def test_import_libs(self) -> None:
        """Test libs package imports."""
        import libs
        assert libs is not None
    
    def test_import_observability(self) -> None:
        """Test observability package imports."""
        import observability
        assert observability is not None


class TestSubpackageImports:
    """Test that subpackages can be imported."""
    
    def test_import_core_subpackages(self) -> None:
        """Test core subpackages import."""
        from core import query_engine
        from core import response
        from core import trace
        assert query_engine is not None
        assert response is not None
        assert trace is not None
    
    def test_import_ingestion_subpackages(self) -> None:
        """Test ingestion subpackages import."""
        from ingestion import chunking
        from ingestion import transform
        from ingestion import embedding
        from ingestion import storage
        assert chunking is not None
        assert transform is not None
        assert embedding is not None
        assert storage is not None
    
    def test_import_libs_subpackages(self) -> None:
        """Test libs subpackages import."""
        from libs import loader
        from libs import llm
        from libs import embedding
        from libs import splitter
        from libs import vector_store
        from libs import reranker
        from libs import evaluator
        assert loader is not None
        assert llm is not None
        assert embedding is not None
        assert splitter is not None
        assert vector_store is not None
        assert reranker is not None
        assert evaluator is not None
