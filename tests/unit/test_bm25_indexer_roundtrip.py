#!/usr/bin/env python3
"""
BM25 Indexer Roundtrip Tests

Tests the BM25 indexer's ability to build, save, load, and query indices.
Verifies IDF calculation accuracy and document removal.
"""

import tempfile
import shutil
from pathlib import Path

import pytest

from ingestion.storage.bm25_indexer import BM25Indexer
from core.types import ChunkRecord


class TestBM25IndexerInit:
    """Test BM25Indexer initialization."""
    
    def test_default_initialization(self):
        """Test indexer initializes with default parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = BM25Indexer(index_path=tmpdir)
            assert indexer.N == 0
            assert indexer.avgdl == 0.0
            assert indexer.k1 == 1.5
            assert indexer.b == 0.75
    
    def test_custom_parameters(self):
        """Test indexer with custom BM25 parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = BM25Indexer(index_path=tmpdir, k1=2.0, b=0.5)
            assert indexer.k1 == 2.0
            assert indexer.b == 0.5
    
    def test_pickle_backend(self):
        """Test indexer with pickle backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = BM25Indexer(index_path=tmpdir, use_sqlite=False)
            assert not indexer.use_sqlite
    
    def test_creates_directory(self):
        """Test indexer creates directory if not exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "subdir" / "bm25"
            indexer = BM25Indexer(index_path=str(index_path))
            assert index_path.exists()


class TestBM25IdfCalculation:
    """Test IDF calculation accuracy."""
    
    def test_idf_single_document(self):
        """Test IDF with single document."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = BM25Indexer(index_path=tmpdir)
            
            # Add one document
            records = [
                ChunkRecord(
                    id="c1",
                    text="hello world",
                    sparse_vector={
                        "terms": ["hello", "world"],
                        "tf": [1, 1],
                        "doc_length": 2
                    }
                )
            ]
            indexer.add_documents(records, source="doc1.txt")
            
            # IDF for term in single doc: log((1 - 1 + 0.5) / (1 + 0.5) + 1) = log(1.33)
            assert indexer.N == 1
            assert "hello" in indexer._terms
            assert indexer._terms["hello"].idf > 0
    
    def test_idf_multiple_documents(self):
        """Test IDF decreases as document frequency increases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = BM25Indexer(index_path=tmpdir)
            
            # Add two documents
            records1 = [
                ChunkRecord(
                    id="c1",
                    text="common unique1",
                    sparse_vector={
                        "terms": ["common", "unique1"],
                        "tf": [1, 1],
                        "doc_length": 2
                    }
                )
            ]
            records2 = [
                ChunkRecord(
                    id="c2",
                    text="common unique2",
                    sparse_vector={
                        "terms": ["common", "unique2"],
                        "tf": [1, 1],
                        "doc_length": 2
                    }
                )
            ]
            indexer.add_documents(records1, source="doc1.txt")
            indexer.add_documents(records2, source="doc2.txt")
            
            # Common term should have lower IDF than unique terms
            common_idf = indexer._terms["common"].idf
            unique1_idf = indexer._terms["unique1"].idf
            unique2_idf = indexer._terms["unique2"].idf
            
            assert common_idf < unique1_idf
            assert common_idf < unique2_idf


class TestBM25AddDocuments:
    """Test adding documents to index."""
    
    def test_add_single_document(self):
        """Test adding a single document."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = BM25Indexer(index_path=tmpdir)
            
            records = [
                ChunkRecord(
                    id="c1",
                    text="hello world",
                    sparse_vector={
                        "terms": ["hello", "world"],
                        "tf": [1, 1],
                        "doc_length": 2
                    }
                )
            ]
            indexer.add_documents(records, source="doc1.txt")
            
            assert indexer.N == 1
            assert indexer._total_dl == 2
            assert "hello" in indexer._terms
            assert "world" in indexer._terms
    
    def test_add_multiple_documents(self):
        """Test adding multiple documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = BM25Indexer(index_path=tmpdir)
            
            records = [
                ChunkRecord(
                    id=f"c{i}",
                    text=f"term{i}",
                    sparse_vector={
                        "terms": [f"term{i}"],
                        "tf": [1],
                        "doc_length": 1
                    }
                )
                for i in range(5)
            ]
            indexer.add_documents(records, source="doc.txt")
            
            assert indexer.N == 5
            assert indexer._total_dl == 5
    
    def test_missing_sparse_vector_raises(self):
        """Test that missing sparse_vector raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = BM25Indexer(index_path=tmpdir)
            
            records = [
                ChunkRecord(
                    id="c1",
                    text="hello world",
                    sparse_vector=None  # type: ignore
                )
            ]
            
            with pytest.raises(ValueError, match="missing sparse_vector"):
                indexer.add_documents(records)
    
    def test_document_source_tracked(self):
        """Test that document source is tracked in postings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = BM25Indexer(index_path=tmpdir)
            
            records = [
                ChunkRecord(
                    id="c1",
                    text="hello",
                    sparse_vector={
                        "terms": ["hello"],
                        "tf": [1],
                        "doc_length": 1
                    }
                )
            ]
            indexer.add_documents(records, source="my_doc.pdf")
            
            posting = indexer._terms["hello"].postings[0]
            assert posting.source == "my_doc.pdf"


class TestBM25Query:
    """Test BM25 querying."""
    
    def test_query_single_term(self):
        """Test querying with a single term."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = BM25Indexer(index_path=tmpdir)
            
            records = [
                ChunkRecord(
                    id="c1",
                    text="hello world",
                    sparse_vector={
                        "terms": ["hello", "world"],
                        "tf": [1, 1],
                        "doc_length": 2
                    }
                )
            ]
            indexer.add_documents(records, source="doc.txt")
            
            results = indexer.query(["hello"])
            assert len(results) == 1
            assert results[0][0] == "c1"
            assert results[0][1] > 0
    
    def test_query_returns_top_k(self):
        """Test that query respects top_k parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = BM25Indexer(index_path=tmpdir)
            
            records = [
                ChunkRecord(
                    id=f"c{i}",
                    text=f"document number {i}",
                    sparse_vector={
                        "terms": ["document", "number", str(i)],
                        "tf": [1, 1, 1],
                        "doc_length": 3
                    }
                )
                for i in range(10)
            ]
            indexer.add_documents(records, source="doc.txt")
            
            results = indexer.query(["document"], top_k=5)
            assert len(results) <= 5
    
    def test_query_empty_keywords(self):
        """Test querying with empty keywords."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = BM25Indexer(index_path=tmpdir)
            results = indexer.query([])
            assert results == []
    
    def test_query_ranking(self):
        """Test that results are ranked by score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = BM25Indexer(index_path=tmpdir)
            
            # Doc with more occurrences should score higher
            records = [
                ChunkRecord(
                    id="c1",
                    text="keyword",
                    sparse_vector={
                        "terms": ["keyword"],
                        "tf": [1],
                        "doc_length": 1
                    }
                ),
                ChunkRecord(
                    id="c2",
                    text="keyword keyword keyword",
                    sparse_vector={
                        "terms": ["keyword"],
                        "tf": [3],
                        "doc_length": 3
                    }
                )
            ]
            indexer.add_documents(records, source="doc.txt")
            
            results = indexer.query(["keyword"])
            assert results[0][0] == "c2"  # Higher TF should rank first
    
    def test_query_case_insensitive(self):
        """Test case-insensitive querying."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = BM25Indexer(index_path=tmpdir)
            
            records = [
                ChunkRecord(
                    id="c1",
                    text="HELLO",
                    sparse_vector={
                        "terms": ["hello"],  # Stored as lowercase
                        "tf": [1],
                        "doc_length": 1
                    }
                )
            ]
            indexer.add_documents(records, source="doc.txt")
            
            results = indexer.query(["HELLO"])
            assert len(results) == 1


class TestBM25Persistence:
    """Test index persistence."""
    
    def test_save_and_load_sqlite(self):
        """Test saving and loading with SQLite backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate indexer
            indexer1 = BM25Indexer(index_path=tmpdir, use_sqlite=True)
            records = [
                ChunkRecord(
                    id="c1",
                    text="hello world",
                    sparse_vector={
                        "terms": ["hello", "world"],
                        "tf": [1, 1],
                        "doc_length": 2
                    }
                )
            ]
            indexer1.add_documents(records, source="doc.txt")
            
            # Load in new indexer
            indexer2 = BM25Indexer(index_path=tmpdir, use_sqlite=True)
            assert indexer2.N == 1
            assert "hello" in indexer2._terms
    
    def test_save_and_load_pickle(self):
        """Test saving and loading with pickle backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer1 = BM25Indexer(index_path=tmpdir, use_sqlite=False)
            records = [
                ChunkRecord(
                    id="c1",
                    text="hello world",
                    sparse_vector={
                        "terms": ["hello", "world"],
                        "tf": [1, 1],
                        "doc_length": 2
                    }
                )
            ]
            indexer1.add_documents(records, source="doc.txt")
            
            indexer2 = BM25Indexer(index_path=tmpdir, use_sqlite=False)
            assert indexer2.N == 1


class TestBM25RemoveDocument:
    """Test document removal."""
    
    def test_remove_document(self):
        """Test removing a document by source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = BM25Indexer(index_path=tmpdir)
            
            records1 = [
                ChunkRecord(
                    id="c1",
                    text="hello",
                    sparse_vector={
                        "terms": ["hello"],
                        "tf": [1],
                        "doc_length": 1
                    }
                )
            ]
            records2 = [
                ChunkRecord(
                    id="c2",
                    text="world",
                    sparse_vector={
                        "terms": ["world"],
                        "tf": [1],
                        "doc_length": 1
                    }
                )
            ]
            indexer.add_documents(records1, source="doc1.txt")
            indexer.add_documents(records2, source="doc2.txt")
            
            removed = indexer.remove_document("doc1.txt")
            assert removed == 1
            assert indexer.N == 1
            assert "hello" not in indexer._terms
            assert "world" in indexer._terms
    
    def test_remove_updates_idf(self):
        """Test that removal updates IDF values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = BM25Indexer(index_path=tmpdir)
            
            # Add two docs with common term
            records1 = [
                ChunkRecord(
                    id="c1",
                    text="common term",
                    sparse_vector={
                        "terms": ["common", "term"],
                        "tf": [1, 1],
                        "doc_length": 2
                    }
                )
            ]
            records2 = [
                ChunkRecord(
                    id="c2",
                    text="common other",
                    sparse_vector={
                        "terms": ["common", "other"],
                        "tf": [1, 1],
                        "doc_length": 2
                    }
                )
            ]
            indexer.add_documents(records1, source="doc1.txt")
            indexer.add_documents(records2, source="doc2.txt")
            
            old_idf = indexer._terms["common"].idf
            indexer.remove_document("doc1.txt")
            new_idf = indexer._terms["common"].idf
            
            # IDF should increase when document frequency decreases
            assert new_idf > old_idf


class TestBM25Stats:
    """Test statistics."""
    
    def test_get_stats(self):
        """Test getting index statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = BM25Indexer(index_path=tmpdir)
            
            records = [
                ChunkRecord(
                    id="c1",
                    text="hello world",
                    sparse_vector={
                        "terms": ["hello", "world"],
                        "tf": [1, 1],
                        "doc_length": 2
                    }
                )
            ]
            indexer.add_documents(records, source="doc.txt")
            
            stats = indexer.get_stats()
            assert stats["N"] == 1
            assert stats["num_terms"] == 2
            assert stats["total_dl"] == 2


class TestBM25Rebuild:
    """Test index rebuilding."""
    
    def test_rebuild_clears_index(self):
        """Test that rebuild clears all data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = BM25Indexer(index_path=tmpdir)
            
            records = [
                ChunkRecord(
                    id="c1",
                    text="hello",
                    sparse_vector={
                        "terms": ["hello"],
                        "tf": [1],
                        "doc_length": 1
                    }
                )
            ]
            indexer.add_documents(records, source="doc.txt")
            
            indexer.rebuild()
            
            assert indexer.N == 0
            assert len(indexer._terms) == 0
