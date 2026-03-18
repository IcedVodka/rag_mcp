#!/usr/bin/env python3
"""
Base Loader - Abstract base class for document loaders.

Provides a common interface for loading documents from various file formats.
All specific loaders (PDF, DOCX, etc.) must inherit from this base class.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from src.core.types import Document


class BaseLoader(ABC):
    """
    Abstract base class for document loaders.
    
    Implementations must provide a `load` method that reads a file from the
    given path and returns a Document object containing the extracted text
    and metadata.
    
    Example:
        >>> loader = PdfLoader()
        >>> doc = loader.load("path/to/file.pdf")
        >>> print(doc.text)
        >>> print(doc.metadata.source_path)
    """
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> Document:
        """
        Load a document from the given file path.
        
        Args:
            path: Path to the file to load. Can be a string or Path object.
            
        Returns:
            Document object containing the extracted text and metadata.
            The metadata must include at least the `source_path` field.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            PermissionError: If the file cannot be read.
            ValueError: If the file format is invalid or unsupported.
            RuntimeError: If an error occurs during loading.
            
        Example:
            >>> loader = PdfLoader()
            >>> doc = loader.load("/path/to/document.pdf")
            >>> print(f"Loaded document with {len(doc.text)} characters")
        """
        pass
    
    def _validate_path(self, path: Union[str, Path]) -> Path:
        """
        Validate that the given path exists and is a file.
        
        Args:
            path: Path to validate.
            
        Returns:
            Path object representing the validated path.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            PermissionError: If the file cannot be accessed.
            ValueError: If the path is not a file.
        """
        file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        # Check read permission by attempting to open
        try:
            with open(file_path, 'rb') as _:
                pass
        except PermissionError:
            raise PermissionError(f"Cannot read file: {file_path}")
        
        return file_path


class LoaderError(Exception):
    """Base exception for loader errors."""
    pass


class FileFormatError(LoaderError):
    """Raised when a file format is invalid or unsupported."""
    pass


class ImageExtractionError(LoaderError):
    """Raised when image extraction fails but text extraction succeeded."""
    pass
