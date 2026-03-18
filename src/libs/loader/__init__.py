"""Loader Module - Document loading and parsing."""

from libs.loader.base_loader import (
    BaseLoader,
    LoaderError,
    FileFormatError,
    ImageExtractionError,
)
from libs.loader.file_integrity import (
    FileIntegrityChecker,
    SQLiteIntegrityChecker,
    IngestionRecord,
    FileIntegrityError,
    HashCalculationError,
    StorageError,
)
from libs.loader.pdf_loader import PdfLoader

__all__ = [
    # Base classes
    "BaseLoader",
    "LoaderError",
    "FileFormatError",
    "ImageExtractionError",
    # File integrity
    "FileIntegrityChecker",
    "SQLiteIntegrityChecker",
    "IngestionRecord",
    "FileIntegrityError",
    "HashCalculationError",
    "StorageError",
    # Specific loaders
    "PdfLoader",
]
