#!/usr/bin/env python3
"""
Ingestion Pipeline Integration Tests

Tests the complete ingestion pipeline workflow.
Uses mocking to isolate the pipeline logic from external dependencies.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from ingestion.pipeline import IngestionPipeline, IngestionResult
from core.trace.trace_context import TraceContext
from core.types import Document, Chunk, ChunkRecord


@pytest.fixture
def mock_settings():
    """Create a properly configured mock settings object."""
    settings = Mock()
    
    # Configure all necessary attributes
    settings.ingestion = Mock()
    settings.ingestion.batch_size = 32
    settings.ingestion.encode_dense = True
    settings.ingestion.encode_sparse = True
    settings.ingestion.chunk_refiner = Mock()
    settings.ingestion.chunk_refiner.use_llm = False
    settings.ingestion.metadata_enricher = Mock()
    settings.ingestion.metadata_enricher.use_llm = False
    settings.ingestion.image_captioner = Mock()
    settings.ingestion.image_captioner.enabled = False
    
    settings.embedding = Mock()
    settings.embedding.provider = "mock"
    
    settings.llm = Mock()
    settings.llm.provider = "mock"
    
    settings.vision_llm = Mock()
    settings.vision_llm.provider = "mock"
    
    settings.bm25_index_path = "data/db/bm25"
    settings.vector_store = Mock()
    settings.vector_store.backend = "chroma"
    settings.vector_store.persist_path = "data/db/chroma"
    
    return settings


@pytest.fixture
def mock_pipeline(mock_settings):
    """Create a pipeline with all mock dependencies."""
    # Create mock components
    mock_checker = Mock()
    mock_loader = Mock()
    mock_chunker = Mock()
    mock_batch_processor = Mock()
    mock_upserter = Mock()
    mock_indexer = Mock()
    mock_storage = Mock()
    
    pipeline = IngestionPipeline(
        mock_settings,
        enable_transforms=False,  # Disable transforms for simplicity
        enable_image_captioning=False,
        integrity_checker=mock_checker,
        pdf_loader=mock_loader,
        chunker=mock_chunker,
        batch_processor=mock_batch_processor,
        vector_upserter=mock_upserter,
        bm25_indexer=mock_indexer,
        image_storage=mock_storage
    )
    
    return pipeline


class TestIngestionPipelineInit:
    """Test pipeline initialization."""
    
    def test_initialization_with_injected_components(self, mock_settings):
        """Test pipeline accepts injected components."""
        mock_checker = Mock()
        mock_loader = Mock()
        mock_chunker = Mock()
        mock_batch_processor = Mock()
        mock_upserter = Mock()
        mock_indexer = Mock()
        mock_storage = Mock()
        
        pipeline = IngestionPipeline(
            mock_settings,
            integrity_checker=mock_checker,
            pdf_loader=mock_loader,
            chunker=mock_chunker,
            batch_processor=mock_batch_processor,
            vector_upserter=mock_upserter,
            bm25_indexer=mock_indexer,
            image_storage=mock_storage
        )
        
        assert pipeline.integrity_checker is mock_checker
        assert pipeline.pdf_loader is mock_loader
        assert pipeline.chunker is mock_chunker


class TestIngestionPipelineRun:
    """Test complete pipeline execution with mocks."""
    
    def test_ingest_success(self, mock_pipeline):
        """Test successful ingestion flow."""
        # Setup mocks
        mock_pipeline.integrity_checker.should_skip.return_value = False
        mock_pipeline.integrity_checker.compute_sha256.return_value = "abc123"
        
        mock_doc = Mock(spec=Document)
        mock_doc.id = "doc1"
        mock_doc.text = "Test document content"
        mock_doc.metadata = {"images": []}
        mock_pipeline.pdf_loader.load.return_value = mock_doc
        
        mock_chunk = Mock(spec=Chunk)
        mock_chunk.id = "chunk1"
        mock_chunk.text = "Test chunk"
        mock_pipeline.chunker.split_document.return_value = [mock_chunk]
        
        mock_record = Mock(spec=ChunkRecord)
        mock_record.id = "record1"
        mock_pipeline.batch_processor.process.return_value = ([mock_record], [mock_record])
        
        # Execute
        result = mock_pipeline.run(
            source_path="/path/to/doc.pdf",
            collection="test_collection"
        )
        
        # Verify
        assert result.success
        assert result.chunks_processed == 1
        assert result.file_hash == "abc123"
        assert result.collection == "test_collection"
        
        # Verify all stages were called
        mock_pipeline.pdf_loader.load.assert_called_once()
        mock_pipeline.chunker.split_document.assert_called_once()
        mock_pipeline.batch_processor.process.assert_called_once()
        mock_pipeline.vector_upserter.upsert_batch.assert_called_once()
        mock_pipeline.bm25_indexer.add_documents.assert_called_once()
        mock_pipeline.integrity_checker.mark_success.assert_called_once()
    
    def test_ingest_skip_unchanged(self, mock_pipeline):
        """Test that unchanged files are skipped."""
        mock_pipeline.integrity_checker.should_skip.return_value = True
        mock_pipeline.integrity_checker.compute_sha256.return_value = "abc123"
        
        result = mock_pipeline.run(
            source_path="/path/to/doc.pdf",
            collection="test_collection"
        )
        
        assert result.success
        assert result.chunks_processed == 0  # Skipped
        
        # Loader should not be called
        mock_pipeline.pdf_loader.load.assert_not_called()
    
    def test_ingest_force_reprocess(self, mock_pipeline):
        """Test force flag bypasses skip check."""
        mock_pipeline.integrity_checker.should_skip.return_value = True
        mock_pipeline.integrity_checker.compute_sha256.return_value = "abc123"
        
        # Setup basic mocks for processing
        mock_doc = Mock(spec=Document)
        mock_doc.metadata = {"images": []}
        mock_pipeline.pdf_loader.load.return_value = mock_doc
        mock_pipeline.chunker.split_document.return_value = []
        mock_pipeline.batch_processor.process.return_value = ([], [])
        
        result = mock_pipeline.run(
            source_path="/path/to/doc.pdf",
            collection="test_collection",
            force=True
        )
        
        # Should process even though should_skip returns True
        mock_pipeline.pdf_loader.load.assert_called_once()
    
    def test_ingest_with_progress_callback(self, mock_pipeline):
        """Test progress callback is invoked."""
        mock_pipeline.integrity_checker.should_skip.return_value = False
        mock_pipeline.integrity_checker.compute_sha256.return_value = "abc123"
        
        mock_doc = Mock(spec=Document)
        mock_doc.metadata = {"images": []}
        mock_pipeline.pdf_loader.load.return_value = mock_doc
        mock_pipeline.chunker.split_document.return_value = []
        mock_pipeline.batch_processor.process.return_value = ([], [])
        
        progress_calls = []
        def on_progress(stage, current, total):
            progress_calls.append((stage, current, total))
        
        mock_pipeline.run(
            source_path="/path/to/doc.pdf",
            collection="test",
            on_progress=on_progress
        )
        
        assert len(progress_calls) > 0
    
    def test_ingest_with_trace(self, mock_pipeline):
        """Test trace context is populated."""
        mock_pipeline.integrity_checker.should_skip.return_value = False
        mock_pipeline.integrity_checker.compute_sha256.return_value = "abc123"
        
        mock_doc = Mock(spec=Document)
        mock_doc.metadata = {"images": []}
        mock_pipeline.pdf_loader.load.return_value = mock_doc
        mock_pipeline.chunker.split_document.return_value = []
        mock_pipeline.batch_processor.process.return_value = ([], [])
        
        trace = TraceContext(trace_type="ingestion")
        
        mock_pipeline.run(
            source_path="/path/to/doc.pdf",
            collection="test",
            trace=trace
        )
        
        assert trace.finished_at is not None
        assert len(trace.stages) > 0


class TestIngestionPipelineErrors:
    """Test pipeline error handling."""
    
    def test_file_not_found(self, mock_pipeline):
        """Test handling of non-existent file."""
        mock_pipeline.integrity_checker.compute_sha256.side_effect = FileNotFoundError("No such file")
        
        result = mock_pipeline.run(
            source_path="/nonexistent/file.pdf",
            collection="test"
        )
        
        assert not result.success
        assert "File hash failed" in result.error_message or "No such file" in result.error_message
    
    def test_processing_error(self, mock_pipeline):
        """Test handling of processing errors."""
        mock_pipeline.integrity_checker.should_skip.return_value = False
        mock_pipeline.integrity_checker.compute_sha256.return_value = "abc123"
        mock_pipeline.pdf_loader.load.side_effect = Exception("Parse error")
        
        result = mock_pipeline.run(
            source_path="/path/to/doc.pdf",
            collection="test"
        )
        
        assert not result.success
        assert "Parse error" in result.error_message
        mock_pipeline.integrity_checker.mark_failed.assert_called_once()


class TestIngestionResult:
    """Test IngestionResult data structure."""
    
    def test_success_result(self):
        """Test successful result creation."""
        result = IngestionResult(
            success=True,
            source_path="/path/to/doc.pdf",
            file_hash="abc123",
            collection="default",
            chunks_processed=10,
            dense_vectors_stored=10,
            sparse_vectors_stored=10,
            elapsed_seconds=5.5,
            trace_id="trace-123"
        )
        
        assert result.success
        assert result.chunks_processed == 10
        assert result.error_message is None
        assert result.trace_id == "trace-123"
    
    def test_failure_result(self):
        """Test failed result creation."""
        result = IngestionResult(
            success=False,
            source_path="/path/to/doc.pdf",
            file_hash="abc123",
            collection="default",
            error_message="Processing failed"
        )
        
        assert not result.success
        assert result.error_message == "Processing failed"


class TestPipelineStages:
    """Test individual pipeline stages."""
    
    def test_compute_file_hash(self, mock_pipeline):
        """Test file hash computation."""
        mock_pipeline.integrity_checker.compute_sha256.return_value = "hash123"
        
        result = mock_pipeline._compute_file_hash("/path/to/file.pdf")
        
        assert result == "hash123"
        mock_pipeline.integrity_checker.compute_sha256.assert_called_once_with("/path/to/file.pdf")
    
    def test_load_document(self, mock_pipeline):
        """Test document loading stage."""
        mock_doc = Mock(spec=Document)
        mock_doc.text = "Test content"
        mock_doc.metadata = {"images": []}
        mock_pipeline.pdf_loader.load.return_value = mock_doc
        
        trace = TraceContext()
        result = mock_pipeline._load_document("/path/to/doc.pdf", trace)
        
        assert result is mock_doc
        mock_pipeline.pdf_loader.load.assert_called_once_with("/path/to/doc.pdf")
    
    def test_split_document(self, mock_pipeline):
        """Test document splitting stage."""
        mock_doc = Mock(spec=Document)
        mock_doc.text = "Test content"
        mock_doc.id = "doc1"
        
        mock_chunk = Mock(spec=Chunk)
        mock_chunk.id = "chunk1"
        mock_chunk.text = "Test chunk content"
        mock_pipeline.chunker.split_document.return_value = [mock_chunk]
        
        trace = TraceContext()
        result = mock_pipeline._split_document(mock_doc, trace)
        
        assert len(result) == 1
        assert result[0] is mock_chunk
    
    def test_transform_chunks(self, mock_pipeline):
        """Test transform stage with no transforms."""
        mock_chunk = Mock(spec=Chunk)
        mock_chunk.text = "Test"
        
        trace = TraceContext()
        result = mock_pipeline._transform_chunks([mock_chunk], trace)
        
        # With no transforms, should return same chunks
        assert result == [mock_chunk]
    
    def test_encode_chunks(self, mock_pipeline):
        """Test encoding stage."""
        mock_chunk = Mock(spec=Chunk)
        
        mock_record = Mock(spec=ChunkRecord)
        mock_pipeline.batch_processor.process.return_value = ([mock_record], [mock_record])
        
        trace = TraceContext()
        dense, sparse = mock_pipeline._encode_chunks([mock_chunk], trace)
        
        assert len(dense) == 1
        assert len(sparse) == 1
    
    def test_store_vectors(self, mock_pipeline):
        """Test storage stage."""
        mock_record = Mock(spec=ChunkRecord)
        
        trace = TraceContext()
        mock_pipeline._store_vectors(
            [mock_record], [mock_record],
            "/path/to/doc.pdf",
            "test_collection",
            trace
        )
        
        mock_pipeline.vector_upserter.upsert_batch.assert_called_once()
        mock_pipeline.bm25_indexer.add_documents.assert_called_once()
