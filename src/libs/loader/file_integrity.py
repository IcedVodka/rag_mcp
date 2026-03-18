#!/usr/bin/env python3
"""
File Integrity Checker - SHA256-based file deduplication and tracking.

Provides functionality to:
- Calculate SHA256 hashes of files
- Track ingestion history in SQLite (with WAL mode for concurrency)
- Determine if a file should be skipped (already processed successfully)
- Mark files as success or failed with error messages

This module supports pluggable storage backends (SQLite by default,
Redis/PostgreSQL can be implemented by subclassing).
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Any


@dataclass
class IngestionRecord:
    """Record of a file ingestion attempt."""
    
    file_hash: str
    file_path: str
    status: str  # 'success' or 'failed'
    error_msg: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class FileIntegrityChecker(ABC):
    """
    Abstract base class for file integrity checking.
    
    Implementations must provide storage backends for tracking
    file ingestion history based on SHA256 hashes.
    """
    
    @staticmethod
    def compute_sha256(path: str | Path, chunk_size: int = 8192) -> str:
        """
        Calculate SHA256 hash of a file.
        
        Reads the file in chunks to handle large files efficiently.
        
        Args:
            path: Path to the file to hash
            chunk_size: Size of chunks to read (default 8KB)
            
        Returns:
            Hexadecimal string of the SHA256 hash
            
        Raises:
            FileNotFoundError: If the file does not exist
            PermissionError: If the file cannot be read
            IOError: On other IO errors
        """
        sha256_hash = hashlib.sha256()
        file_path = Path(path)
        
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """
        Check if a file should be skipped (already processed successfully).
        
        Args:
            file_hash: SHA256 hash of the file
            
        Returns:
            True if the file was previously processed successfully,
            False otherwise (new file or previously failed)
        """
        pass
    
    @abstractmethod
    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Mark a file as successfully processed.
        
        Args:
            file_hash: SHA256 hash of the file
            file_path: Original path of the file
            metadata: Optional additional metadata about the ingestion
        """
        pass
    
    @abstractmethod
    def mark_processing(self, file_hash: str, file_path: str) -> None:
        """
        Mark a file as currently being processed.
        
        Args:
            file_hash: SHA256 hash of the file
            file_path: Original path of the file
        """
        pass
    
    @abstractmethod
    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        """
        Mark a file as failed during processing.
        
        Args:
            file_hash: SHA256 hash of the file
            error_msg: Error message describing why it failed
        """
        pass
    
    @abstractmethod
    def get_record(self, file_hash: str) -> Optional[IngestionRecord]:
        """
        Retrieve the ingestion record for a file hash.
        
        Args:
            file_hash: SHA256 hash of the file
            
        Returns:
            IngestionRecord if found, None otherwise
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close any open resources (connections, etc.)."""
        pass
    
    def __enter__(self) -> FileIntegrityChecker:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """
    SQLite-based implementation of file integrity checker.
    
    Uses WAL (Write-Ahead Logging) mode for better concurrency support.
    Database is automatically created if it doesn't exist.
    
    Attributes:
        db_path: Path to the SQLite database file
        connection: Active SQLite connection
    """
    
    def __init__(self, db_path: str | Path = "data/db/ingestion_history.db") -> None:
        """
        Initialize the SQLite integrity checker.
        
        Args:
            db_path: Path to the SQLite database file.
                    Defaults to 'data/db/ingestion_history.db'
        """
        self.db_path = Path(db_path)
        self._connection: Optional[sqlite3.Connection] = None
        
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,  # Allow use across threads
                timeout=30.0,  # Wait up to 30s for locks
            )
            # Enable WAL mode for better concurrency
            self._connection.execute("PRAGMA journal_mode=WAL")
            # Enable foreign keys
            self._connection.execute("PRAGMA foreign_keys=ON")
        return self._connection
    
    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = self._get_connection()
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_history (
                file_hash TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('success', 'failed')),
                error_msg TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index on status for faster queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON ingestion_history(status)
        """)
        
        conn.commit()
    
    def should_skip(self, file_hash: str) -> bool:
        """
        Check if a file should be skipped.
        
        A file should be skipped if it was previously marked as 'success'.
        Failed files should be retried, so they return False.
        
        Args:
            file_hash: SHA256 hash of the file
            
        Returns:
            True if the file was successfully processed before
        """
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT 1 FROM ingestion_history WHERE file_hash = ? AND status = 'success'",
            (file_hash,)
        )
        return cursor.fetchone() is not None
    
    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Mark a file as successfully processed.
        
        Uses UPSERT (INSERT ... ON CONFLICT) to handle re-ingestion
        of previously failed files.
        
        Args:
            file_hash: SHA256 hash of the file
            file_path: Original path of the file
            metadata: Optional metadata (currently unused, reserved for future)
        """
        conn = self._get_connection()
        
        # UPSERT: Insert or update existing record
        conn.execute(
            """
            INSERT INTO ingestion_history (file_hash, file_path, status, error_msg)
            VALUES (?, ?, 'success', NULL)
            ON CONFLICT(file_hash) DO UPDATE SET
                status = 'success',
                error_msg = NULL,
                file_path = excluded.file_path,
                updated_at = CURRENT_TIMESTAMP
            """,
            (file_hash, str(file_path))
        )
        conn.commit()
    
    def mark_processing(self, file_hash: str, file_path: str) -> None:
        """
        Mark a file as currently being processed.
        
        This is optional and can be used to track in-progress files.
        By default, this is a no-op as we only track success/failure.
        
        Args:
            file_hash: SHA256 hash of the file
            file_path: Original path of the file
        """
        # Optional: Can be implemented to track in-progress state
        # For now, we only track success/failure
        pass
    
    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        """
        Mark a file as failed during processing.
        
        Uses UPSERT to handle both new failures and re-failures.
        
        Args:
            file_hash: SHA256 hash of the file
            error_msg: Error message describing why it failed
        """
        conn = self._get_connection()
        
        conn.execute(
            """
            INSERT INTO ingestion_history (file_hash, file_path, status, error_msg)
            VALUES (?, '', 'failed', ?)
            ON CONFLICT(file_hash) DO UPDATE SET
                status = 'failed',
                error_msg = excluded.error_msg,
                updated_at = CURRENT_TIMESTAMP
            """,
            (file_hash, error_msg)
        )
        conn.commit()
    
    def get_record(self, file_hash: str) -> Optional[IngestionRecord]:
        """
        Retrieve the ingestion record for a file hash.
        
        Args:
            file_hash: SHA256 hash of the file
            
        Returns:
            IngestionRecord if found, None otherwise
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT file_hash, file_path, status, error_msg, created_at, updated_at
            FROM ingestion_history
            WHERE file_hash = ?
            """,
            (file_hash,)
        )
        
        row = cursor.fetchone()
        if row is None:
            return None
        
        # Parse timestamps
        created_at = None
        updated_at = None
        
        if row[4]:
            try:
                created_at = datetime.fromisoformat(row[4])
            except (ValueError, TypeError):
                pass
        
        if row[5]:
            try:
                updated_at = datetime.fromisoformat(row[5])
            except (ValueError, TypeError):
                pass
        
        return IngestionRecord(
            file_hash=row[0],
            file_path=row[1],
            status=row[2],
            error_msg=row[3],
            created_at=created_at,
            updated_at=updated_at
        )
    
    def get_stats(self) -> dict[str, int]:
        """
        Get statistics about ingestion history.
        
        Returns:
            Dictionary with counts of success, failed, and total records
        """
        conn = self._get_connection()
        
        cursor = conn.execute(
            "SELECT status, COUNT(*) FROM ingestion_history GROUP BY status"
        )
        
        stats = {"success": 0, "failed": 0, "total": 0}
        for status, count in cursor.fetchall():
            if status in stats:
                stats[status] = count
            stats["total"] += count
        
        return stats
    
    def reset(self, confirm: bool = False) -> None:
        """
        Clear all ingestion history.
        
        Args:
            confirm: Must be True to actually clear data (safety measure)
            
        Raises:
            ValueError: If confirm is not True
        """
        if not confirm:
            raise ValueError("Must pass confirm=True to reset all data")
        
        conn = self._get_connection()
        conn.execute("DELETE FROM ingestion_history")
        conn.commit()
    
    def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None


class FileIntegrityError(Exception):
    """Base exception for file integrity errors."""
    pass


class HashCalculationError(FileIntegrityError):
    """Raised when hash calculation fails."""
    pass


class StorageError(FileIntegrityError):
    """Raised when storage operation fails."""
    pass
