#!/usr/bin/env python3
"""
Core Data Types - Domain models for the RAG pipeline.

Provides shared data structures used across ingestion, retrieval, and MCP tools
to avoid coupling and duplication between submodules.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from datetime import datetime


@dataclass
class ImageInfo:
    """
    Image metadata for multimodal document support.
    
    Attributes:
        id: Global unique image identifier (format: {doc_hash}_{page}_{seq})
        path: Image file storage path (convention: data/images/{collection}/{image_id}.png)
        page: Page number in the original document (optional, for PDFs)
        text_offset: Start character position of placeholder in Document.text (0-indexed)
        text_length: Length of placeholder in characters (typically len("[IMAGE: {image_id}]"))
        position: Physical position info in original document (optional)
    """
    id: str
    path: str
    page: Optional[int] = None
    text_offset: int = 0
    text_length: int = 0
    position: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageInfo":
        """Create ImageInfo from dictionary."""
        return cls(**data)


@dataclass
class DocumentMetadata:
    """
    Standard metadata structure for documents.
    
    Attributes:
        source_path: Required field indicating the source file path
        created_at: Timestamp when the document was created
        updated_at: Timestamp when the document was last updated
        images: List of image metadata for multimodal support
        extra: Additional custom metadata fields
    """
    source_path: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    images: list[ImageInfo] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "source_path": self.source_path,
            "images": [img.to_dict() for img in self.images],
        }
        if self.created_at:
            result["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            result["updated_at"] = self.updated_at.isoformat()
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentMetadata":
        """Create DocumentMetadata from dictionary."""
        # Extract known fields
        source_path = data.pop("source_path", "")
        created_at = data.pop("created_at", None)
        updated_at = data.pop("updated_at", None)
        
        # Parse timestamps
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if updated_at and isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        # Parse images
        images_data = data.pop("images", [])
        images = [ImageInfo.from_dict(img) for img in images_data]
        
        # Remaining fields go to extra
        return cls(
            source_path=source_path,
            created_at=created_at,
            updated_at=updated_at,
            images=images,
            extra=data
        )


@dataclass
class Document:
    """
    Document type representing a source document in the RAG pipeline.
    
    In the text field, images are marked with placeholders in the format:
    `[IMAGE: {image_id}]`
    
    Attributes:
        id: Unique document identifier
        text: Document text content (may contain image placeholders)
        metadata: Document metadata (must include source_path)
    """
    id: str
    text: str
    metadata: DocumentMetadata

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """Create Document from dictionary."""
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            metadata = DocumentMetadata.from_dict(metadata.copy())
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=metadata
        )

    def get_image_placeholders(self) -> list[str]:
        """
        Extract all image placeholder IDs from the document text.
        
        Returns:
            List of image IDs found in placeholders
        """
        import re
        pattern = r'\[IMAGE:\s*([^\]]+)\]'
        matches = re.findall(pattern, self.text)
        return [match.strip() for match in matches]

    def validate_image_placeholders(self) -> bool:
        """
        Validate that all image placeholders have corresponding metadata.
        
        Returns:
            True if all placeholders have metadata entries
        """
        placeholder_ids = set(self.get_image_placeholders())
        
        # Handle both dict and Metadata object types
        images = []
        if self.metadata:
            if hasattr(self.metadata, 'images'):
                images = self.metadata.images
            elif isinstance(self.metadata, dict):
                images = self.metadata.get('images', [])
        
        metadata_ids = {img.id for img in images}
        return placeholder_ids.issubset(metadata_ids)


@dataclass
class Chunk:
    """
    Chunk type representing a text segment from a document.
    
    Attributes:
        id: Unique chunk identifier
        text: Chunk text content
        metadata: Chunk metadata (inherits from document)
        start_offset: Start character position in source document
        end_offset: End character position in source document
        source_ref: Reference to source document (e.g., document ID)
    """
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    start_offset: int = 0
    end_offset: int = 0
    source_ref: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata.copy(),
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "source_ref": self.source_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Create Chunk from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data.get("metadata", {}).copy(),
            start_offset=data.get("start_offset", 0),
            end_offset=data.get("end_offset", 0),
            source_ref=data.get("source_ref")
        )

    def get_char_count(self) -> int:
        """Get character count of the chunk text."""
        return len(self.text)

    def overlaps_with(self, other: "Chunk") -> bool:
        """
        Check if this chunk overlaps with another chunk.
        
        Args:
            other: Another chunk to check overlap with
            
        Returns:
            True if chunks overlap in source document
        """
        if self.source_ref != other.source_ref:
            return False
        return not (self.end_offset <= other.start_offset or self.start_offset >= other.end_offset)


@dataclass
class ChunkRecord:
    """
    ChunkRecord type for storage and retrieval in vector stores.
    
    This is the carrier that flows through the ingestion and retrieval pipeline,
    containing both the chunk content and its vector representations.
    
    Attributes:
        id: Unique record identifier
        text: Chunk text content
        metadata: Record metadata (includes chunk and document metadata)
        dense_vector: Dense embedding vector (optional)
        sparse_vector: Sparse embedding vector for hybrid search (optional)
    """
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_vector: Optional[list[float]] = None
    sparse_vector: Optional[dict[str, float]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation for serialization."""
        result = {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata.copy(),
        }
        if self.dense_vector is not None:
            result["dense_vector"] = self.dense_vector.copy()
        if self.sparse_vector is not None:
            result["sparse_vector"] = self.sparse_vector.copy()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkRecord":
        """Create ChunkRecord from dictionary."""
        dense_vector = data.get("dense_vector")
        if dense_vector is not None:
            dense_vector = list(dense_vector)
        
        sparse_vector = data.get("sparse_vector")
        if sparse_vector is not None:
            sparse_vector = dict(sparse_vector)
        
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data.get("metadata", {}).copy(),
            dense_vector=dense_vector,
            sparse_vector=sparse_vector
        )

    def has_vectors(self) -> bool:
        """Check if the record has any vector representations."""
        return self.dense_vector is not None or self.sparse_vector is not None

    def has_dense_vector(self) -> bool:
        """Check if the record has a dense vector."""
        return self.dense_vector is not None

    def has_sparse_vector(self) -> bool:
        """Check if the record has a sparse vector."""
        return self.sparse_vector is not None

    def get_vector_dimensions(self) -> dict[str, int]:
        """
        Get dimensions of vectors if present.
        
        Returns:
            Dictionary with 'dense' and/or 'sparse' keys and their dimensions
        """
        dims = {}
        if self.dense_vector is not None:
            dims["dense"] = len(self.dense_vector)
        if self.sparse_vector is not None:
            dims["sparse"] = len(self.sparse_vector)
        return dims


@dataclass
class RetrievalResult:
    """
    Retrieval result from vector search.
    
    Represents a single retrieved chunk with its similarity score,
    text content, and metadata.
    
    Attributes:
        chunk_id: Unique identifier of the retrieved chunk
        score: Similarity score (distance or cosine similarity)
        text: Retrieved text content
        metadata: Additional metadata associated with the chunk
    """
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalResult":
        """Create RetrievalResult from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            score=data["score"],
            text=data["text"],
            metadata=data.get("metadata", {}).copy()
        )
