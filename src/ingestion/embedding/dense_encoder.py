#!/usr/bin/env python3
"""
Dense Encoder - Batch encoding for chunks using dense embeddings.

Converts chunks to ChunkRecords with dense vector representations
using a configurable embedding provider.
"""

from typing import Optional, List

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


class DenseEncoder:
    """
    Encoder for generating dense embeddings from chunks.
    
    Uses the configured embedding provider (via EmbeddingFactory) to
    generate dense vectors from chunk text. Supports batch processing
    to handle large numbers of chunks efficiently.
    
    Attributes:
        embedding_client: The underlying embedding provider instance
        
    Example:
        >>> settings = load_settings("config.yaml")
        >>> encoder = DenseEncoder(settings)
        >>> chunks = [Chunk(id="c1", text="Hello world")]
        >>> records = encoder.encode(chunks)
        >>> print(records[0].dense_vector)  # [0.1, 0.2, ...]
    """
    
    def __init__(
        self,
        settings: Settings,
        embedding_client: Optional[BaseEmbedding] = None
    ) -> None:
        """
        Initialize the dense encoder.
        
        Args:
            settings: Application settings containing embedding configuration
            embedding_client: Optional pre-configured embedding client.
                If provided, uses this instead of creating via EmbeddingFactory.
                If None, creates a new embedding instance via EmbeddingFactory.
        """
        if embedding_client is not None:
            self.embedding_client = embedding_client
        else:
            self.embedding_client = EmbeddingFactory.create(settings)
    
    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[ChunkRecord]:
        """
        Encode chunks into ChunkRecords with dense vectors.
        
        Batch processes all chunks using the configured batch size
        from the embedding client.
        
        Args:
            chunks: List of chunks to encode
            trace: Optional trace context for observability
            
        Returns:
            List of ChunkRecord objects with dense_vector populated.
            Returns empty list if input is empty.
            
        Note:
            - sparse_vector is always None (DenseEncoder doesn't generate sparse vectors)
            - Output list maintains same order as input chunks
        """
        if not chunks:
            return []
        
        return self.encode_batch(chunks, trace)
    
    def encode_batch(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[ChunkRecord]:
        """
        Encode a single batch of chunks into ChunkRecords.
        
        Processes all chunks in one batch, generating dense embeddings
        for each chunk's text content.
        
        Args:
            chunks: List of chunks to encode
            trace: Optional trace context for observability
            
        Returns:
            List of ChunkRecord objects with dense_vector populated.
            Returns empty list if input is empty.
            
        Raises:
            ValueError: If chunk text is empty and embedding model doesn't support it
            
        Note:
            This method handles the actual batch processing. The batch size
            is controlled by the embedding_client.batch_size configuration.
        """
        if not chunks:
            return []
        
        # Record trace if provided
        if trace:
            trace.record_stage(
                name="dense_encoding",
                method="batch_encode",
                provider=self.embedding_client.__class__.__name__,
                details={
                    "model": self.embedding_client.model,
                    "dimensions": self.embedding_client.dimensions,
                    "batch_size": self.embedding_client.batch_size,
                    "chunk_count": len(chunks)
                }
            )
        
        # Extract texts from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings using batch processing
        # The embed method handles batching internally based on batch_size
        embeddings = self.embedding_client.embed(texts, trace=trace)
        
        # Create ChunkRecords with embeddings
        records: List[ChunkRecord] = []
        for chunk, embedding in zip(chunks, embeddings):
            record = ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata={
                    **chunk.metadata,
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                    "source_ref": chunk.source_ref,
                },
                dense_vector=embedding,
                sparse_vector=None  # DenseEncoder doesn't generate sparse vectors
            )
            records.append(record)
        
        return records
