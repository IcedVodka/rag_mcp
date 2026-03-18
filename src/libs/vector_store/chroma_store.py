#!/usr/bin/env python3
"""
Chroma Store - ChromaDB Vector Store Implementation

Uses ChromaDB for local vector storage and retrieval.
"""

from typing import Optional, Any
from pathlib import Path

from core.trace.trace_context import TraceContext
from .base_vector_store import BaseVectorStore, VectorRecord, QueryResult, VectorStoreError


class ChromaStore(BaseVectorStore):
    """Vector store implementation using ChromaDB."""
    
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.persist_directory = config.get("persist_directory", "data/db/chroma")
        self.distance_function = config.get("distance_function", "cosine")
        
        # Ensure directory exists
        Path(self.persist_directory).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_function}
            )
        except ImportError:
            raise VectorStoreError("chromadb is required. Install with: pip install chromadb")
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB: {e}")
    
    def upsert(
        self,
        records: list[VectorRecord],
        trace: Optional[TraceContext] = None
    ) -> None:
        """Upsert vectors into ChromaDB."""
        if not records:
            return
        
        if trace:
            trace.record_stage(
                name="vector_upsert",
                provider="chroma",
                details={"record_count": len(records)}
            )
        
        ids = [r.id for r in records]
        embeddings = [r.vector for r in records]
        documents = [r.text for r in records]
        metadatas = [r.metadata for r in records]
        
        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
        except Exception as e:
            raise VectorStoreError(f"ChromaDB upsert failed: {e}")
    
    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
        trace: Optional[TraceContext] = None
    ) -> list[QueryResult]:
        """Query for similar vectors."""
        if trace:
            trace.record_stage(
                name="vector_query",
                provider="chroma",
                details={"top_k": top_k}
            )
        
        try:
            results = self._collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                where=filters
            )
            
            # Convert to QueryResult
            query_results = []
            for i, id_ in enumerate(results["ids"][0]):
                query_results.append(QueryResult(
                    id=id_,
                    score=results["distances"][0][i],
                    text=results["documents"][0][i],
                    metadata=results["metadatas"][0][i] if results["metadatas"][0] else {}
                ))
            
            return query_results
        except Exception as e:
            raise VectorStoreError(f"ChromaDB query failed: {e}")
    
    def delete(
        self,
        ids: list[str],
        trace: Optional[TraceContext] = None
    ) -> None:
        """Delete vectors by ID."""
        if not ids:
            return
        
        try:
            self._collection.delete(ids=ids)
        except Exception as e:
            raise VectorStoreError(f"ChromaDB delete failed: {e}")
    
    def get_by_ids(
        self,
        ids: list[str],
        trace: Optional[TraceContext] = None
    ) -> list[dict[str, Any]]:
        """Get records by their IDs."""
        if not ids:
            return []
        
        try:
            results = self._collection.get(ids=ids)
            
            records = []
            for i, id_ in enumerate(results["ids"]):
                records.append({
                    "id": id_,
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i] if results["metadatas"] else {}
                })
            
            return records
        except Exception as e:
            raise VectorStoreError(f"ChromaDB get failed: {e}")
    
    def count(self) -> int:
        """Return the total number of vectors."""
        return self._collection.count()
