#!/usr/bin/env python3
"""
Document Chunker - Business Adapter for Text Splitting

Transforms Document objects into Chunk objects using libs.splitter.
Acts as an adapter layer between raw text splitting and business objects.
"""

import hashlib
import re
from typing import Any, Optional

from core.settings import Settings
from core.types import Document, Chunk, ImageInfo
from libs.splitter.splitter_factory import SplitterFactory
from libs.splitter.base_splitter import BaseSplitter


class DocumentChunker:
    """
    Business adapter for splitting Document objects into Chunk objects.
    
    Responsibilities:
    1. Generate unique and deterministic Chunk IDs
    2. Inherit metadata from Document to each Chunk
    3. Add chunk_index for ordering and positioning
    4. Establish source_ref linking Chunk to parent Document
    5. Distribute image references to chunks based on placeholders
    6. Convert splitter output to Chunk type objects
    
    This class wraps libs.splitter (str → List[str]) and provides
    the business logic layer (Document → List[Chunk]).
    """
    
    def __init__(self, settings: Settings) -> None:
        """
        Initialize the DocumentChunker with settings.
        
        Args:
            settings: Application settings containing splitter configuration
        """
        self.settings = settings
        self._splitter: Optional[BaseSplitter] = None
    
    @property
    def splitter(self) -> BaseSplitter:
        """
        Lazy initialization of the splitter instance.
        
        Returns:
            Configured splitter instance from SplitterFactory
        """
        if self._splitter is None:
            self._splitter = SplitterFactory.create(self.settings)
        return self._splitter
    
    def split_document(self, document: Document) -> list[Chunk]:
        """
        Split a Document into Chunk objects.
        
        The complete transformation pipeline:
        1. Split document text using configured splitter
        2. Calculate character offsets for each chunk
        3. Generate unique chunk IDs
        4. Inherit and enhance metadata
        5. Create Chunk objects with source references
        
        Args:
            document: Document to split
            
        Returns:
            List of Chunk objects with complete metadata
        """
        if not document.text:
            return []
        
        # Get text chunks from splitter
        text_chunks = self.splitter.split_text(document.text)
        
        if not text_chunks:
            return []
        
        # Calculate offsets and create chunks
        chunks: list[Chunk] = []
        current_offset = 0
        doc_text = document.text
        doc_len = len(doc_text)
        
        for chunk_index, chunk_text in enumerate(text_chunks):
            if not chunk_text:
                continue
            
            # Find the actual position of this chunk in the document
            start_offset = doc_text.find(chunk_text, current_offset)
            if start_offset == -1:
                # Fallback: use current_offset if exact match not found
                start_offset = min(current_offset, doc_len)
            
            # Ensure end_offset doesn't exceed document length
            end_offset = min(start_offset + len(chunk_text), doc_len)
            
            # Generate unique chunk ID
            chunk_id = self._generate_chunk_id(document.id, chunk_index, chunk_text)
            
            # Build metadata with inheritance and image distribution
            metadata = self._inherit_metadata(document, chunk_index, chunk_text, start_offset)
            
            # Create Chunk object
            chunk = Chunk(
                id=chunk_id,
                text=chunk_text,
                metadata=metadata,
                start_offset=start_offset,
                end_offset=end_offset,
                source_ref=document.id
            )
            
            chunks.append(chunk)
            
            # Update offset for next chunk (move past this chunk)
            current_offset = start_offset + max(1, len(chunk_text) // 2)
        
        return chunks
    
    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """
        Generate a unique and deterministic Chunk ID.
        
        Format: {doc_id}_{index:04d}_{hash_8chars}
        
        The hash ensures uniqueness even if content changes,
        while the index ensures deterministic ordering.
        
        Args:
            doc_id: Parent document ID
            index: Chunk index within the document
            text: Chunk text content (used for hash)
            
        Returns:
            Unique chunk identifier
        """
        # Create hash from document ID, index, and text content
        hash_input = f"{doc_id}:{index}:{text}"
        hash_value = hashlib.md5(hash_input.encode('utf-8')).hexdigest()[:8]
        
        return f"{doc_id}_{index:04d}_{hash_value}"
    
    def _inherit_metadata(
        self,
        document: Document,
        chunk_index: int,
        chunk_text: str,
        start_offset: int
    ) -> dict[str, Any]:
        """
        Build chunk metadata by inheriting from document and adding chunk-specific fields.
        
        Includes:
        - All fields from Document.metadata
        - chunk_index: position in document
        - Image references relevant to this chunk
        
        Args:
            document: Source document
            chunk_index: Position of this chunk in the document
            chunk_text: Text content of this chunk
            start_offset: Start character position in document
            
        Returns:
            Complete metadata dictionary for the chunk
        """
        # Start with document metadata
        metadata = document.metadata.to_dict()
        
        # Remove images field from inherited metadata - will add back only if chunk references images
        # This ensures chunks without image placeholders don't have an images field
        if 'images' in metadata:
            del metadata['images']
        
        # Add chunk-specific fields
        metadata['chunk_index'] = chunk_index
        
        # Extract and distribute image references for this chunk
        image_refs = self._extract_image_references(chunk_text)
        
        if image_refs:
            # Get the subset of images referenced in this chunk
            chunk_images = self._get_chunk_images(document, image_refs)
            if chunk_images:
                metadata['images'] = [img.to_dict() for img in chunk_images]
                metadata['image_refs'] = image_refs
        
        return metadata
    
    def _extract_image_references(self, text: str) -> list[str]:
        """
        Extract image placeholder IDs from chunk text.
        
        Matches pattern: [IMAGE: {id}]
        
        Args:
            text: Chunk text content
            
        Returns:
            List of image IDs referenced in the text
        """
        pattern = r'\[IMAGE:\s*([^\]]+)\]'
        matches = re.findall(pattern, text)
        return [match.strip() for match in matches]
    
    def _get_chunk_images(self, document: Document, image_refs: list[str]) -> list[ImageInfo]:
        """
        Get ImageInfo objects for images referenced in the chunk.
        
        Args:
            document: Source document containing image metadata
            image_refs: List of image IDs to look up
            
        Returns:
            List of ImageInfo objects for the referenced images
        """
        if not document.metadata.images:
            return []
        
        # Build lookup from image ID to ImageInfo
        image_lookup = {img.id: img for img in document.metadata.images}
        
        # Get images in the order they appear in the references
        chunk_images = []
        for ref in image_refs:
            if ref in image_lookup:
                chunk_images.append(image_lookup[ref])
        
        return chunk_images
