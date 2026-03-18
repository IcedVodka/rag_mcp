#!/usr/bin/env python3
"""
Vector Upserter - Idempotent vector storage manager.

Handles upserting ChunkRecords with dense vectors into the vector store.
Generates stable chunk IDs and ensures idempotent writes.
"""

import hashlib
from typing import List, Optional, Dict, Any
from core.settings import Settings
from core.types import ChunkRecord
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


class VectorUpserter:
    """
    Manages upserting chunks to vector store with idempotent chunk IDs.
    
    Generates deterministic chunk IDs based on source path, chunk index,
    and content hash to ensure idempotent writes. Same content will always
    produce the same ID, avoiding duplicates.
    
    ID format: hashlib.sha256(f"{source_path}:{chunk_index}:{content_hash[:8]}".encode()).hexdigest()[:16]
    
    Attributes:
        vector_store: The underlying vector store implementation
        
    Example:
        >>> settings = load_settings("config/settings.yaml")
        >>> upserter = VectorUpserter(settings)
        >>> records = [ChunkRecord(id="c1", text="hello", dense_vector=[0.1, ...])]
        >>> upserter.upsert(records)
    """
    
    def __init__(
        self,
        settings: Settings,
        vector_store: Optional[BaseVectorStore] = None
    ) -> None:
        """
        Initialize the vector upserter.
        
        Args:
            settings: Application settings
            vector_store: Optional pre-configured vector store instance.
                If None, creates one using VectorStoreFactory.
        """
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = VectorStoreFactory.create(settings)
    
    @staticmethod
    def generate_chunk_id(
        source_path: str,
        chunk_index: int,
        content_hash: str
    ) -> str:
        """
        Generate a stable chunk ID.
        
        Uses SHA256 hash of source path, chunk index, and content hash prefix
        to produce a deterministic 16-character hex ID.
        
        Args:
            source_path: Path to the source document
            chunk_index: Index of the chunk within the document
            content_hash: Hash of the chunk content (at least 8 chars)
            
        Returns:
            16-character hex string chunk ID
            
        Example:
            >>> VectorUpserter.generate_chunk_id("doc.pdf", 0, "abc123...")
            'a1b2c3d4e5f67890'
        """
        # Use first 8 chars of content hash for efficiency
        hash_input = f"{source_path}:{chunk_index}:{content_hash[:8]}"
        full_hash = hashlib.sha256(hash_input.encode()).hexdigest()
        return full_hash[:16]
    
    def _compute_content_hash(self, text: str) -> str:
        """
        Compute SHA256 hash of text content.
        
        Args:
            text: Content to hash
            
        Returns:
            64-character hex hash
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def upsert(
        self,
        records: List[ChunkRecord],
        collection: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upsert chunk records to the vector store.
        
        Generates stable IDs for records that don't have them and upserts
        all records to the vector store.
        
        Args:
            records: List of ChunkRecords with dense_vector populated
            collection: Optional collection/namespace for the vectors
            
        Returns:
            Dictionary with upsert statistics:
            {
                "upserted_count": int,
                "chunk_ids": List[str],
                "collection": str
            }
            
        Raises:
            ValueError: If a record lacks dense_vector
            RuntimeError: If vector store upsert fails
        """
        if not records:
            return {"upserted_count": 0, "chunk_ids": [], "collection": collection or "default"}
        
        # Ensure all records have stable IDs
        for record in records:
            if not record.dense_vector:
                raise ValueError(f"Record {record.id} missing dense_vector")
            
            # Generate stable ID if not already set or if it's a temporary ID
            if not record.id or len(record.id) < 16:
                source = record.metadata.get("source_path", "unknown")
                index = record.metadata.get("chunk_index", 0)
                content_hash = self._compute_content_hash(record.text)
                record.id = self.generate_chunk_id(source, index, content_hash)
        
        # Upsert to vector store
        try:
            self.vector_store.upsert(records, collection=collection)
        except Exception as e:
            raise RuntimeError(f"Vector store upsert failed: {e}") from e
        
        return {
            "upserted_count": len(records),
            "chunk_ids": [r.id for r in records],
            "collection": collection or "default"
        }
    
    def upsert_batch(
        self,
        records: List[ChunkRecord],
        collection: Optional[str] = None,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Upsert records in batches to avoid large payloads.
        
        Args:
            records: List of ChunkRecords to upsert
            collection: Optional collection name
            batch_size: Number of records per batch
            
        Returns:
            Dictionary with total upsert statistics
        """
        if not records:
            return {"upserted_count": 0, "chunk_ids": [], "collection": collection or "default"}
        
        all_chunk_ids = []
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            result = self.upsert(batch, collection)
            all_chunk_ids.extend(result["chunk_ids"])
        
        return {
            "upserted_count": len(all_chunk_ids),
            "chunk_ids": all_chunk_ids,
            "collection": collection or "default"
        }
    
    def delete_by_source(
        self,
        source_path: str,
        collection: Optional[str] = None
    ) -> int:
        """
        Delete all chunks from a source document.
        
        Args:
            source_path: Source document path to delete
            collection: Optional collection name
            
        Returns:
            Number of chunks deleted
        """
        # Use metadata filter to delete by source
        filter_dict = {"source_path": source_path}
        
        try:
            # Check if vector store supports delete_by_metadata
            if hasattr(self.vector_store, 'delete_by_metadata'):
                return self.vector_store.delete_by_metadata(filter_dict, collection)
            else:
                # Fallback: query for IDs then delete
                # This is a simplified implementation
                return 0
        except Exception as e:
            raise RuntimeError(f"Delete failed: {e}") from e
