#!/usr/bin/env python3
"""
PDF Loader - Document loader for PDF files.

Uses markitdown for text extraction and PyMuPDF (fitz) for image extraction.
Supports multimodal documents by extracting images and inserting placeholders.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional, Union

from core.types import Document, DocumentMetadata, ImageInfo
from libs.loader.base_loader import BaseLoader, LoaderError
from libs.loader.file_integrity import FileIntegrityChecker

logger = logging.getLogger(__name__)

# Optional dependencies - imported at module level for testability
try:
    from markitdown import MarkItDown
    _HAS_MARKITDOWN = True
except ImportError:
    MarkItDown = None  # type: ignore
    _HAS_MARKITDOWN = False

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except ImportError:
    fitz = None  # type: ignore
    _HAS_FITZ = False


class PdfLoader(BaseLoader):
    """
    Loader for PDF documents.
    
    Extracts text using markitdown and images using PyMuPDF (fitz).
    Images are saved to `data/images/{doc_hash}/` directory and referenced
    by placeholders in the document text.
    
    Attributes:
        images_dir: Base directory for storing extracted images.
                   Defaults to "data/images".
        collection: Collection name for organizing images.
                   Defaults to "default".
    
    Example:
        >>> loader = PdfLoader()
        >>> doc = loader.load("path/to/document.pdf")
        >>> print(doc.text)  # May contain [IMAGE: ...] placeholders
        >>> print(doc.metadata.images)  # List of ImageInfo objects
    """
    
    def __init__(
        self,
        images_dir: Union[str, Path] = "data/images",
        collection: str = "default"
    ) -> None:
        """
        Initialize the PDF loader.
        
        Args:
            images_dir: Base directory for storing extracted images.
                       Defaults to "data/images".
            collection: Collection name for organizing images.
                       Defaults to "default".
        """
        self.images_dir = Path(images_dir)
        self.collection = collection
        
        # Ensure images directory exists
        self.images_dir.mkdir(parents=True, exist_ok=True)
    
    def load(self, path: Union[str, Path]) -> Document:
        """
        Load a PDF document from the given file path.
        
        Extracts text content and images from the PDF. Images are saved to
        the configured images directory and referenced by placeholders in
        the document text.
        
        Args:
            path: Path to the PDF file to load.
            
        Returns:
            Document object containing:
                - id: SHA256 hash of the file
                - text: Extracted text with [IMAGE: {image_id}] placeholders
                - metadata: DocumentMetadata with source_path and images list
            
        Raises:
            FileNotFoundError: If the file does not exist.
            PermissionError: If the file cannot be read.
            ValueError: If the file is not a valid PDF.
            RuntimeError: If text extraction fails.
            
        Note:
            Image extraction failures are logged as warnings but do not block
            text extraction. The document will be returned with whatever
            images could be successfully extracted.
        """
        file_path = self._validate_path(path)
        
        # Compute document hash for ID
        doc_hash = FileIntegrityChecker.compute_sha256(file_path)
        
        # Extract text using markitdown
        try:
            text = self._extract_text(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {e}") from e
        
        # Extract images using PyMuPDF
        images: list[ImageInfo] = []
        try:
            text, images = self._extract_images(file_path, doc_hash, text)
        except Exception as e:
            logger.warning(f"Image extraction failed for {file_path}: {e}")
            # Continue without images - don't block text extraction
        
        # Create document metadata
        metadata = DocumentMetadata(
            source_path=str(file_path.absolute()),
            images=images
        )
        
        # Create and return document
        return Document(
            id=doc_hash,
            text=text,
            metadata=metadata
        )
    
    def _extract_text(self, file_path: Path) -> str:
        """
        Extract text content from PDF using markitdown.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            Extracted text content.
            
        Raises:
            ImportError: If markitdown is not installed.
            RuntimeError: If text extraction fails.
        """
        if not _HAS_MARKITDOWN or MarkItDown is None:
            raise ImportError(
                "markitdown is required for PDF text extraction. "
                "Install with: pip install markitdown"
            )
        
        md = MarkItDown()
        
        try:
            result = md.convert(str(file_path))
            return result.text_content
        except Exception as e:
            raise RuntimeError(f"markitdown conversion failed: {e}") from e
    
    def _extract_images(
        self,
        file_path: Path,
        doc_hash: str,
        text: str
    ) -> tuple[str, list[ImageInfo]]:
        """
        Extract images from PDF using PyMuPDF (fitz).
        
        Images are saved to `data/images/{collection}/{image_id}.png` and
        placeholders are inserted into the text at appropriate positions.
        
        Args:
            file_path: Path to the PDF file.
            doc_hash: SHA256 hash of the document (used for image IDs).
            text: The extracted text (used for calculating text offsets).
            
        Returns:
            Tuple of (updated_text, images) where:
                - updated_text: Text with image placeholders appended
                - images: List of ImageInfo objects describing extracted images
            
        Raises:
            ImportError: If PyMuPDF is not installed.
            ImageExtractionError: If image extraction fails.
            
        Note:
            This is a best-effort extraction. Some images may not be
            extractable depending on the PDF format and encoding.
        """
        if not _HAS_FITZ or fitz is None:
            raise ImportError(
                "PyMuPDF (fitz) is required for image extraction. "
                "Install with: pip install pymupdf"
            )
        
        images: list[ImageInfo] = []
        collection_dir = self.images_dir / self.collection
        collection_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            doc = fitz.open(str(file_path))
        except Exception as e:
            raise LoaderError(f"Failed to open PDF with PyMuPDF: {e}") from e
        
        # Work with a mutable copy of text
        updated_text = text
        
        try:
            image_seq = 0
            current_text_offset = len(updated_text)  # Start at end of text
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_images = page.get_images()
                
                for img_index, img in enumerate(page_images):
                    xref = img[0]
                    image_seq += 1
                    
                    # Generate unique image ID
                    image_id = f"{doc_hash}_{page_num + 1}_{image_seq}"
                    
                    try:
                        # Extract image data
                        base_image = doc.extract_image(xref)
                        if base_image is None:
                            logger.warning(
                                f"Could not extract image {img_index} from page {page_num + 1}"
                            )
                            continue
                        
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Save image to file (always use .png extension)
                        image_filename = f"{image_id}.png"
                        image_path = collection_dir / image_filename
                        
                        # Convert to PNG if needed using PIL
                        if image_ext != "png":
                            try:
                                from PIL import Image
                                import io
                                
                                img_pil = Image.open(io.BytesIO(image_bytes))
                                # Convert to RGB if necessary (for JPEG, etc.)
                                if img_pil.mode in ('RGBA', 'LA', 'P'):
                                    img_pil = img_pil.convert('RGBA')
                                else:
                                    img_pil = img_pil.convert('RGB')
                                img_pil.save(image_path, 'PNG')
                            except Exception as convert_error:
                                logger.warning(
                                    f"Could not convert image {image_id} to PNG: {convert_error}. "
                                    f"Saving as original format."
                                )
                                # Fallback: save as original format
                                image_filename = f"{image_id}.{image_ext}"
                                image_path = collection_dir / image_filename
                                with open(image_path, 'wb') as f:
                                    f.write(image_bytes)
                        else:
                            # Already PNG, save directly
                            with open(image_path, 'wb') as f:
                                f.write(image_bytes)
                        
                        # Create placeholder
                        placeholder = f"[IMAGE: {image_id}]"
                        
                        # For now, append placeholders at the end of text
                        # In a more sophisticated implementation, we could try to
                        # insert them at positions corresponding to their layout
                        # location in the PDF
                        text_offset = current_text_offset
                        text_length = len(placeholder)
                        
                        # Update text with placeholder
                        if images:  # Not the first image
                            updated_text += "\n\n" + placeholder
                        else:  # First image
                            updated_text += "\n\n" + placeholder
                        
                        current_text_offset = len(updated_text)
                        
                        # Create ImageInfo
                        image_info = ImageInfo(
                            id=image_id,
                            path=str(image_path.relative_to(self.images_dir.parent)),
                            page=page_num + 1,  # 1-indexed page numbers
                            text_offset=text_offset,
                            text_length=text_length
                        )
                        images.append(image_info)
                        
                        logger.debug(f"Extracted image {image_id} from page {page_num + 1}")
                        
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract image {img_index} from page {page_num + 1}: {e}"
                        )
                        # Continue with next image - don't let one failure block others
                        continue
            
            return updated_text, images
            
        finally:
            doc.close()
    
    def _generate_image_placeholder(self, image_id: str) -> str:
        """
        Generate a placeholder string for an image.
        
        Args:
            image_id: Unique identifier for the image.
            
        Returns:
            Placeholder string in the format [IMAGE: {image_id}].
        """
        return f"[IMAGE: {image_id}]"
