#!/usr/bin/env python3
"""
Batch Processor - Orchestrates dense and sparse encoding for chunks.

Provides batch processing capabilities for encoding chunks into both dense
and sparse vector representations. Handles batch splitting, parallel encoding
pipelines, and trace recording for observability.
"""

import time
from typing import Optional, List, Tuple

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseEncoder


class BatchProcessor:
    """
    Orchestrates batch processing of chunks through dense and sparse encoders.
    
    Splits chunks into batches, drives encoding through both dense and sparse
    encoders, and records timing information for tracing. Supports configuration
    switches to enable/disable either encoding type.
    
    Attributes:
        batch_size: Number of chunks to process in each batch
        encode_dense: Whether to generate dense vectors
        encode_sparse: Whether to generate sparse vectors
        dense_encoder: Encoder for dense embeddings
        sparse_encoder: Encoder for sparse (BM25) vectors
        
    Example:
        >>> settings = load_settings("config/settings.yaml")
        >>> processor = BatchProcessor(settings)
        >>> chunks = [Chunk(id="c1", text="Hello world")]
        >>> trace = TraceContext(trace_type="ingestion")
        >>> dense_records, sparse_records = processor.process(chunks, trace)
        >>> print(f"Dense: {len(dense_records)}, Sparse: {len(sparse_records)}")
    """
    
    def __init__(
        self,
        settings: Settings,
        dense_encoder: Optional[DenseEncoder] = None,
        sparse_encoder: Optional[SparseEncoder] = None
    ) -> None:
        """
        Initialize the batch processor.
        
        Args:
            settings: Application settings containing ingestion configuration
            dense_encoder: Optional pre-configured dense encoder instance.
                If provided, uses this instead of creating a new one.
                If None, creates a new DenseEncoder instance.
            sparse_encoder: Optional pre-configured sparse encoder instance.
                If provided, uses this instead of creating a new one.
                If None, creates a new SparseEncoder instance.
        """
        # Load configuration from settings
        ingestion_config = getattr(settings, 'ingestion', None)
        if ingestion_config:
            self.batch_size = getattr(ingestion_config, 'batch_size', 32)
            self.encode_dense = getattr(ingestion_config, 'encode_dense', True)
            self.encode_sparse = getattr(ingestion_config, 'encode_sparse', True)
        else:
            self.batch_size = 32
            self.encode_dense = True
            self.encode_sparse = True
        
        # Initialize encoders (can be injected for testing)
        if dense_encoder is not None:
            self.dense_encoder = dense_encoder
        elif self.encode_dense:
            self.dense_encoder = DenseEncoder(settings)
        else:
            self.dense_encoder = None
            
        if sparse_encoder is not None:
            self.sparse_encoder = sparse_encoder
        elif self.encode_sparse:
            self.sparse_encoder = SparseEncoder(settings)
        else:
            self.sparse_encoder = None
    
    def process(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> Tuple[List[ChunkRecord], List[ChunkRecord]]:
        """
        Process all chunks through dense and/or sparse encoding.
        
        Splits chunks into batches and processes each batch through the
        configured encoders. Returns two lists of ChunkRecords - one for
        dense vectors and one for sparse vectors. Both lists maintain the
        same order as the input chunks.
        
        Args:
            chunks: List of chunks to encode
            trace: Optional trace context for observability and timing
            
        Returns:
            Tuple of (dense_records, sparse_records):
            - dense_records: List of ChunkRecords with dense_vector populated
            - sparse_records: List of ChunkRecords with sparse_vector populated
            Both lists maintain the same index position for the same chunk.
            Returns empty lists if input is empty or encoding type is disabled.
        """
        # Handle empty input
        if not chunks:
            return [], []
        
        # Split into batches
        batches = self._split_batches(chunks, self.batch_size)
        
        # Initialize result lists
        all_dense_records: List[ChunkRecord] = []
        all_sparse_records: List[ChunkRecord] = []
        
        # Process each batch
        for batch_idx, batch in enumerate(batches):
            # Process dense encoding if enabled
            if self.encode_dense and self.dense_encoder is not None:
                batch_start = time.time()
                dense_batch_records = self._process_batch_dense(batch, trace)
                batch_elapsed_ms = (time.time() - batch_start) * 1000
                
                # Record batch timing to trace
                if trace:
                    trace.record_stage(
                        name="dense_batch_processing",
                        method="batch_encode",
                        provider=self.dense_encoder.__class__.__name__,
                        details={
                            "batch_index": batch_idx,
                            "batch_size": len(batch),
                            "elapsed_ms": batch_elapsed_ms
                        }
                    )
                
                all_dense_records.extend(dense_batch_records)
            
            # Process sparse encoding if enabled
            if self.encode_sparse and self.sparse_encoder is not None:
                batch_start = time.time()
                sparse_batch_records = self._process_batch_sparse(batch, trace)
                batch_elapsed_ms = (time.time() - batch_start) * 1000
                
                # Record batch timing to trace
                if trace:
                    trace.record_stage(
                        name="sparse_batch_processing",
                        method="tokenize_and_count",
                        provider=f"tokenizer:{self.sparse_encoder.tokenizer}",
                        details={
                            "batch_index": batch_idx,
                            "batch_size": len(batch),
                            "elapsed_ms": batch_elapsed_ms
                        }
                    )
                
                all_sparse_records.extend(sparse_batch_records)
        
        return all_dense_records, all_sparse_records
    
    def _split_batches(
        self,
        chunks: List[Chunk],
        batch_size: int
    ) -> List[List[Chunk]]:
        """
        Split chunks into batches of specified size.
        
        Maintains the order of chunks in the batches. The last batch may
        contain fewer chunks if the total count is not evenly divisible.
        
        Args:
            chunks: List of chunks to split
            batch_size: Maximum number of chunks per batch
            
        Returns:
            List of batches, where each batch is a list of chunks.
            Returns empty list if input is empty.
            
        Example:
            >>> chunks = [Chunk(id=f"c{i}") for i in range(5)]
            >>> batches = processor._split_batches(chunks, batch_size=2)
            >>> len(batches)  # 3 batches: [c0,c1], [c2,c3], [c4]
            3
        """
        if not chunks:
            return []
        
        batches: List[List[Chunk]] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def _process_batch_dense(
        self,
        batch: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[ChunkRecord]:
        """
        Process a single batch through the dense encoder.
        
        Args:
            batch: List of chunks to encode
            trace: Optional trace context for observability
            
        Returns:
            List of ChunkRecords with dense_vector populated.
            Returns empty list if batch is empty.
            
        Raises:
            RuntimeError: If dense encoding is not enabled but called.
        """
        if not batch:
            return []
        
        if self.dense_encoder is None:
            raise RuntimeError("Dense encoder not initialized")
        
        return self.dense_encoder.encode_batch(batch, trace)
    
    def _process_batch_sparse(
        self,
        batch: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[ChunkRecord]:
        """
        Process a single batch through the sparse encoder.
        
        Args:
            batch: List of chunks to encode
            trace: Optional trace context for observability
            
        Returns:
            List of ChunkRecords with sparse_vector populated.
            Returns empty list if batch is empty.
            
        Raises:
            RuntimeError: If sparse encoding is not enabled but called.
        """
        if not batch:
            return []
        
        if self.sparse_encoder is None:
            raise RuntimeError("Sparse encoder not initialized")
        
        return self.sparse_encoder.encode_batch(batch, trace)
