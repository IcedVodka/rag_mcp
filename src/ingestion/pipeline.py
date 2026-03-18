#!/usr/bin/env python3
"""
Ingestion Pipeline - Complete document processing orchestration.

Orchestrates the full ingestion workflow from file loading through vector storage:
integrity check → load → split → transform → encode → store

Supports progress callbacks for real-time monitoring and comprehensive error handling
with per-stage exception management.
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord, Document

# Import all pipeline stages
from ingestion.chunking.document_chunker import DocumentChunker
from ingestion.embedding.batch_processor import BatchProcessor
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from ingestion.storage.vector_upserter import VectorUpserter
from ingestion.transform.base_transform import BaseTransform
from ingestion.transform.chunk_refiner import ChunkRefiner
from ingestion.transform.image_captioner import ImageCaptioner
from ingestion.transform.metadata_enricher import MetadataEnricher

from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.loader.pdf_loader import PdfLoader


logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of a document ingestion operation."""
    success: bool
    source_path: str
    file_hash: str
    collection: str
    chunks_processed: int = 0
    images_extracted: int = 0
    dense_vectors_stored: int = 0
    sparse_vectors_stored: int = 0
    elapsed_seconds: float = 0.0
    error_message: Optional[str] = None
    trace_id: Optional[str] = None


@dataclass
class PipelineStage:
    """Pipeline stage configuration."""
    name: str
    enabled: bool = True
    description: str = ""


ProgressCallback = Callable[[str, int, int], None]


class IngestionPipeline:
    """
    Complete document ingestion pipeline orchestrator.
    
    Coordinates all ingestion stages from file integrity checking through
    final vector storage. Supports incremental updates, progress tracking,
    and comprehensive error handling.
    
    Pipeline stages:
    1. Integrity Check - Skip unchanged files using SHA256
    2. Load - Parse PDF into Document (text + images)
    3. Split - Divide into Chunks
    4. Transform - Refine, enrich metadata, caption images
    5. Encode - Generate dense and sparse vectors
    6. Store - Persist to ChromaDB and BM25 index
    
    Example:
        >>> settings = load_settings("config/settings.yaml")
        >>> pipeline = IngestionPipeline(settings)
        >>> result = pipeline.run("doc.pdf", collection="default")
        >>> print(f"Processed {result.chunks_processed} chunks")
    
    Attributes:
        settings: Application configuration
        integrity_checker: SHA256-based file deduplication
        pdf_loader: PDF parsing with image extraction
        chunker: Document to chunks conversion
        transforms: List of transformation stages
        batch_processor: Dense/sparse encoding coordinator
        vector_upserter: ChromaDB storage manager
        bm25_indexer: BM25 sparse index manager
        image_storage: Extracted image file manager
    """
    
    def __init__(
        self,
        settings: Settings,
        enable_transforms: bool = True,
        enable_image_captioning: bool = True,
        integrity_checker: Optional[SQLiteIntegrityChecker] = None,
        pdf_loader: Optional[PdfLoader] = None,
        chunker: Optional[DocumentChunker] = None,
        batch_processor: Optional[BatchProcessor] = None,
        vector_upserter: Optional[VectorUpserter] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        image_storage: Optional[ImageStorage] = None
    ) -> None:
        """
        Initialize the ingestion pipeline.
        
        Args:
            settings: Application settings
            enable_transforms: Whether to enable transform stages (refiner, enricher)
            enable_image_captioning: Whether to enable image captioning
            integrity_checker: Optional pre-configured integrity checker
            pdf_loader: Optional pre-configured PDF loader
            chunker: Optional pre-configured document chunker
            batch_processor: Optional pre-configured batch processor
            vector_upserter: Optional pre-configured vector upserter
            bm25_indexer: Optional pre-configured BM25 indexer
            image_storage: Optional pre-configured image storage
        """
        self.settings = settings
        
        # Core components (can be injected for testing)
        self.integrity_checker = integrity_checker or SQLiteIntegrityChecker()
        self.pdf_loader = pdf_loader or PdfLoader()
        self.chunker = chunker or DocumentChunker(settings)
        
        # Transform pipeline
        self.transforms: List[BaseTransform] = []
        if enable_transforms:
            self.transforms.append(ChunkRefiner(settings))
            self.transforms.append(MetadataEnricher(settings))
            if enable_image_captioning:
                self.transforms.append(ImageCaptioner(settings))
        
        # Encoding and storage (can be injected for testing)
        self.batch_processor = batch_processor or BatchProcessor(settings)
        self.vector_upserter = vector_upserter or VectorUpserter(settings)
        self.bm25_indexer = bm25_indexer or BM25Indexer(
            index_path=getattr(settings, 'bm25_index_path', 'data/db/bm25'),
            use_sqlite=True
        )
        self.image_storage = image_storage or ImageStorage()
        
        logger.info("IngestionPipeline initialized with %d transforms", 
                   len(self.transforms))
    
    def run(
        self,
        source_path: str,
        collection: str = "default",
        force: bool = False,
        on_progress: Optional[ProgressCallback] = None,
        trace: Optional[TraceContext] = None
    ) -> IngestionResult:
        """
        Execute the complete ingestion pipeline on a document.
        
        Args:
            source_path: Path to the document file
            collection: Target collection/namespace
            force: Force re-ingestion even if unchanged
            on_progress: Optional callback(stage_name, current, total)
            trace: Optional trace context for observability
            
        Returns:
            IngestionResult with processing statistics
            
        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If file type is unsupported
            RuntimeError: If critical pipeline stage fails
        """
        start_time = time.time()
        source_path = str(Path(source_path).resolve())
        
        # Initialize trace
        if trace is None:
            trace = TraceContext(trace_type="ingestion")
        
        # Stage 1: Integrity Check
        if on_progress:
            on_progress("integrity_check", 0, 6)
        
        try:
            file_hash = self._compute_file_hash(source_path)
        except Exception as e:
            logger.error(f"Failed to compute file hash: {e}")
            return IngestionResult(
                success=False,
                source_path=source_path,
                file_hash="",
                collection=collection,
                error_message=f"File hash failed: {e}",
                trace_id=trace.trace_id
            )
        
        # Check if should skip
        if not force and self.integrity_checker.should_skip(file_hash):
            logger.info(f"Skipping unchanged file: {source_path}")
            return IngestionResult(
                success=True,
                source_path=source_path,
                file_hash=file_hash,
                collection=collection,
                chunks_processed=0,
                elapsed_seconds=time.time() - start_time,
                trace_id=trace.trace_id
            )
        
        if on_progress:
            on_progress("integrity_check", 1, 6)
        
        # Record start of processing
        self.integrity_checker.mark_processing(file_hash, source_path)
        
        try:
            # Stage 2: Load Document
            document = self._load_document(source_path, trace, collection)
            if on_progress:
                on_progress("load", 2, 6)
            
            # Stage 3: Split into Chunks
            chunks = self._split_document(document, trace)
            if on_progress:
                on_progress("split", 3, 6)
            
            # Stage 4: Transform
            chunks = self._transform_chunks(chunks, trace)
            if on_progress:
                on_progress("transform", 4, 6)
            
            # Stage 5: Encode
            dense_records, sparse_records = self._encode_chunks(chunks, trace)
            if on_progress:
                on_progress("encode", 5, 6)
            
            # Stage 6: Store
            self._store_vectors(
                dense_records, sparse_records, 
                source_path, collection, trace
            )
            if on_progress:
                on_progress("store", 6, 6)
            
            # Mark success
            self.integrity_checker.mark_success(
                file_hash, source_path, 
                metadata={"chunk_count": len(chunks)}
            )
            
            elapsed = time.time() - start_time
            
            # Record completion in trace
            trace.record_stage(
                name="ingestion_complete",
                method="pipeline",
                details={
                    "source_path": source_path,
                    "collection": collection,
                    "chunks_processed": len(chunks),
                    "dense_vectors": len(dense_records),
                    "sparse_vectors": len(sparse_records),
                    "elapsed_seconds": elapsed
                }
            )
            trace.finish()
            
            logger.info(
                f"Successfully ingested {source_path}: "
                f"{len(chunks)} chunks in {elapsed:.2f}s"
            )
            
            return IngestionResult(
                success=True,
                source_path=source_path,
                file_hash=file_hash,
                collection=collection,
                chunks_processed=len(chunks),
                images_extracted=len(document.metadata.images) if document.metadata and hasattr(document.metadata, 'images') else len(document.metadata.get('images', [])) if isinstance(document.metadata, dict) else 0,
                dense_vectors_stored=len(dense_records),
                sparse_vectors_stored=len(sparse_records),
                elapsed_seconds=elapsed,
                trace_id=trace.trace_id
            )
            
        except Exception as e:
            # Mark failure
            self.integrity_checker.mark_failed(file_hash, str(e))
            
            elapsed = time.time() - start_time
            logger.error(f"Ingestion failed for {source_path}: {e}")
            
            trace.record_stage(
                name="ingestion_failed",
                method="pipeline",
                details={"error": str(e), "source_path": source_path}
            )
            trace.finish()
            
            return IngestionResult(
                success=False,
                source_path=source_path,
                file_hash=file_hash,
                collection=collection,
                elapsed_seconds=elapsed,
                error_message=str(e),
                trace_id=trace.trace_id
            )
    
    def _compute_file_hash(self, source_path: str) -> str:
        """Compute SHA256 hash of file."""
        return self.integrity_checker.compute_sha256(source_path)
    
    def _load_document(
        self, 
        source_path: str, 
        trace: TraceContext,
        collection: str = "default"
    ) -> Document:
        """Load document using appropriate loader."""
        stage_start = time.time()
        
        path = Path(source_path)
        if path.suffix.lower() == ".pdf":
            document = self.pdf_loader.load(source_path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        # Index extracted images to SQLite
        images_indexed = 0
        if document.metadata and hasattr(document.metadata, 'images') and document.metadata.images:
            for i, img_info in enumerate(document.metadata.images):
                try:
                    # Get absolute path to the image
                    img_path = Path(img_info.path)
                    if not img_path.is_absolute():
                        # Path is relative to data directory
                        img_path = Path("data") / img_path
                    
                    self.image_storage.index_existing_image(
                        image_path=str(img_path),
                        image_id=img_info.id,
                        doc_hash=document.id,
                        collection=collection,
                        page_num=img_info.page
                    )
                    images_indexed += 1
                except Exception as e:
                    logger.warning(f"Failed to index image {img_info.id}: {e}")
        
        trace.record_stage(
            name="load",
            method="pdf_loader",
            provider="markitdown",
            details={
                "source_path": source_path,
                "text_length": len(document.text),
                "image_count": len(document.metadata.images) if document.metadata and hasattr(document.metadata, 'images') else len(document.metadata.get('images', [])) if isinstance(document.metadata, dict) else 0,
                "images_indexed": images_indexed,
                "elapsed_ms": (time.time() - stage_start) * 1000
            }
        )
        
        return document
    
    def _split_document(self, document: Document, trace: TraceContext) -> List[Chunk]:
        """Split document into chunks."""
        stage_start = time.time()
        
        chunks = self.chunker.split_document(document)
        
        trace.record_stage(
            name="split",
            method="document_chunker",
            provider=self.chunker.splitter.__class__.__name__,
            details={
                "input_length": len(document.text),
                "chunks_produced": len(chunks),
                "avg_chunk_length": sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0,
                "elapsed_ms": (time.time() - stage_start) * 1000
            }
        )
        
        return chunks
    
    def _transform_chunks(
        self, 
        chunks: List[Chunk], 
        trace: TraceContext
    ) -> List[Chunk]:
        """Apply transform pipeline to chunks."""
        stage_start = time.time()
        
        for transform in self.transforms:
            transform_start = time.time()
            transform_name = transform.__class__.__name__
            
            chunks = transform.transform(chunks, trace)
            
            trace.record_stage(
                name=f"transform_{transform_name.lower()}",
                method=transform_name,
                details={
                    "chunks_processed": len(chunks),
                    "elapsed_ms": (time.time() - transform_start) * 1000
                }
            )
        
        trace.record_stage(
            name="transform",
            method="pipeline",
            details={
                "transforms_applied": len(self.transforms),
                "chunks_output": len(chunks),
                "elapsed_ms": (time.time() - stage_start) * 1000
            }
        )
        
        return chunks
    
    def _encode_chunks(
        self, 
        chunks: List[Chunk], 
        trace: TraceContext
    ) -> Tuple[List[ChunkRecord], List[ChunkRecord]]:
        """Encode chunks to dense and sparse vectors."""
        stage_start = time.time()
        
        dense_records, sparse_records = self.batch_processor.process(chunks, trace)
        
        trace.record_stage(
            name="encode",
            method="batch_processor",
            details={
                "chunks_input": len(chunks),
                "dense_output": len(dense_records),
                "sparse_output": len(sparse_records),
                "elapsed_ms": (time.time() - stage_start) * 1000
            }
        )
        
        return dense_records, sparse_records
    
    def _store_vectors(
        self,
        dense_records: List[ChunkRecord],
        sparse_records: List[ChunkRecord],
        source_path: str,
        collection: str,
        trace: TraceContext
    ) -> None:
        """Store vectors to persistent storage."""
        stage_start = time.time()
        
        # Store dense vectors to ChromaDB
        if dense_records:
            self.vector_upserter.upsert_batch(dense_records, collection=collection)
        
        # Store sparse vectors to BM25 index
        if sparse_records:
            self.bm25_indexer.add_documents(sparse_records, source=source_path)
        
        trace.record_stage(
            name="store",
            method="storage",
            provider="chroma+bm25",
            details={
                "dense_stored": len(dense_records),
                "sparse_stored": len(sparse_records),
                "collection": collection,
                "elapsed_ms": (time.time() - stage_start) * 1000
            }
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "bm25_stats": self.bm25_indexer.get_stats(),
            "image_stats": self.image_storage.get_stats(),
            "transforms_enabled": len(self.transforms)
        }
