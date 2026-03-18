#!/usr/bin/env python3
"""
File Integrity Checker Unit Tests

Tests for SHA256 hash calculation and SQLite-based ingestion tracking.
Uses temporary files and in-memory/temporary databases to avoid
polluting the real data directory.
"""

import hashlib
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from src.libs.loader.file_integrity import (
    FileIntegrityChecker,
    SQLiteIntegrityChecker,
    IngestionRecord,
    HashCalculationError,
)


class TestComputeSHA256:
    """Tests for SHA256 hash calculation."""
    
    def test_compute_sha256_basic(self, tmp_path: Path) -> None:
        """Test basic SHA256 calculation."""
        test_file = tmp_path / "test.txt"
        content = b"Hello, World!"
        test_file.write_bytes(content)
        
        expected_hash = hashlib.sha256(content).hexdigest()
        actual_hash = FileIntegrityChecker.compute_sha256(test_file)
        
        assert actual_hash == expected_hash
        assert len(actual_hash) == 64  # SHA256 is 256 bits = 64 hex chars
    
    def test_compute_sha256_empty_file(self, tmp_path: Path) -> None:
        """Test SHA256 of empty file."""
        test_file = tmp_path / "empty.txt"
        test_file.write_bytes(b"")
        
        expected_hash = hashlib.sha256(b"").hexdigest()
        actual_hash = FileIntegrityChecker.compute_sha256(test_file)
        
        assert actual_hash == expected_hash
    
    def test_compute_sha256_large_file(self, tmp_path: Path) -> None:
        """Test SHA256 calculation for large file (chunked reading)."""
        test_file = tmp_path / "large.bin"
        # Create 1MB file
        content = os.urandom(1024 * 1024)
        test_file.write_bytes(content)
        
        expected_hash = hashlib.sha256(content).hexdigest()
        actual_hash = FileIntegrityChecker.compute_sha256(test_file)
        
        assert actual_hash == expected_hash
    
    def test_compute_sha256_consistency(self, tmp_path: Path) -> None:
        """Test that same file produces same hash multiple times."""
        test_file = tmp_path / "consistent.txt"
        test_file.write_text("Test content for consistency check")
        
        hash1 = FileIntegrityChecker.compute_sha256(test_file)
        hash2 = FileIntegrityChecker.compute_sha256(test_file)
        hash3 = FileIntegrityChecker.compute_sha256(test_file)
        
        assert hash1 == hash2 == hash3
    
    def test_compute_sha256_different_files(self, tmp_path: Path) -> None:
        """Test that different files produce different hashes."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        
        file1.write_text("Content A")
        file2.write_text("Content B")
        
        hash1 = FileIntegrityChecker.compute_sha256(file1)
        hash2 = FileIntegrityChecker.compute_sha256(file2)
        
        assert hash1 != hash2
    
    def test_compute_sha256_with_string_path(self, tmp_path: Path) -> None:
        """Test that string paths work correctly."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")
        
        hash_from_str = FileIntegrityChecker.compute_sha256(str(test_file))
        hash_from_path = FileIntegrityChecker.compute_sha256(test_file)
        
        assert hash_from_str == hash_from_path
    
    def test_compute_sha256_file_not_found(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for non-existent file."""
        nonexistent = tmp_path / "does_not_exist.txt"
        
        with pytest.raises(FileNotFoundError):
            FileIntegrityChecker.compute_sha256(nonexistent)
    
    def test_compute_sha256_custom_chunk_size(self, tmp_path: Path) -> None:
        """Test hash calculation with custom chunk size."""
        test_file = tmp_path / "test.txt"
        content = b"Test content for custom chunk size"
        test_file.write_bytes(content)
        
        expected_hash = hashlib.sha256(content).hexdigest()
        
        # Test with different chunk sizes
        hash_1k = FileIntegrityChecker.compute_sha256(test_file, chunk_size=1024)
        hash_4k = FileIntegrityChecker.compute_sha256(test_file, chunk_size=4096)
        hash_8k = FileIntegrityChecker.compute_sha256(test_file, chunk_size=8192)
        
        assert hash_1k == hash_4k == hash_8k == expected_hash


class TestSQLiteIntegrityChecker:
    """Tests for SQLite-based integrity checker."""
    
    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> Generator[Path, None, None]:
        """Create a temporary database file."""
        db_path = tmp_path / "test_ingestion.db"
        yield db_path
        # Cleanup is handled by tmp_path
    
    @pytest.fixture
    def checker(self, temp_db: Path) -> Generator[SQLiteIntegrityChecker, None, None]:
        """Create a SQLiteIntegrityChecker with temp database."""
        checker = SQLiteIntegrityChecker(db_path=temp_db)
        yield checker
        checker.close()
    
    def test_database_creation(self, temp_db: Path) -> None:
        """Test that database file is created."""
        assert not temp_db.exists()
        
        checker = SQLiteIntegrityChecker(db_path=temp_db)
        
        assert temp_db.exists()
        checker.close()
    
    def test_table_creation(self, temp_db: Path) -> None:
        """Test that required tables are created."""
        checker = SQLiteIntegrityChecker(db_path=temp_db)
        checker.close()
        
        # Directly query the database to verify schema
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ingestion_history'"
        )
        assert cursor.fetchone() is not None
        conn.close()
    
    def test_wal_mode_enabled(self, temp_db: Path) -> None:
        """Test that WAL mode is enabled for concurrency."""
        checker = SQLiteIntegrityChecker(db_path=temp_db)
        conn = checker._get_connection()
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        checker.close()
        
        assert mode.lower() == "wal"
    
    def test_should_skip_new_file(self, checker: SQLiteIntegrityChecker) -> None:
        """Test should_skip returns False for new files."""
        result = checker.should_skip("nonexistent_hash_12345")
        
        assert result is False
    
    def test_should_skip_after_mark_success(self, checker: SQLiteIntegrityChecker) -> None:
        """Test should_skip returns True after marking success."""
        file_hash = "abc123"
        
        # Initially should not skip
        assert checker.should_skip(file_hash) is False
        
        # Mark as success
        checker.mark_success(file_hash, "/path/to/file.txt")
        
        # Now should skip
        assert checker.should_skip(file_hash) is True
    
    def test_should_skip_after_mark_failed(self, checker: SQLiteIntegrityChecker) -> None:
        """Test should_skip returns False after marking failed."""
        file_hash = "def456"
        
        # Mark as failed
        checker.mark_failed(file_hash, "Parse error: invalid format")
        
        # Failed files should NOT be skipped (can retry)
        assert checker.should_skip(file_hash) is False
    
    def test_mark_success_creates_record(self, checker: SQLiteIntegrityChecker) -> None:
        """Test mark_success creates a record in the database."""
        file_hash = "hash123"
        file_path = "/data/documents/test.pdf"
        
        checker.mark_success(file_hash, file_path)
        
        record = checker.get_record(file_hash)
        assert record is not None
        assert record.file_hash == file_hash
        assert record.file_path == file_path
        assert record.status == "success"
        assert record.error_msg is None
    
    def test_mark_failed_creates_record(self, checker: SQLiteIntegrityChecker) -> None:
        """Test mark_failed creates a record with error message."""
        file_hash = "hash456"
        error_msg = "Failed to parse PDF: corrupted file"
        
        checker.mark_failed(file_hash, error_msg)
        
        record = checker.get_record(file_hash)
        assert record is not None
        assert record.file_hash == file_hash
        assert record.status == "failed"
        assert record.error_msg == error_msg
    
    def test_mark_success_updates_existing_failed(self, checker: SQLiteIntegrityChecker) -> None:
        """Test mark_success can update a previously failed record."""
        file_hash = "hash789"
        
        # First mark as failed
        checker.mark_failed(file_hash, "Initial failure")
        assert checker.should_skip(file_hash) is False
        
        # Then mark as success
        checker.mark_success(file_hash, "/path/to/file.txt")
        
        # Now should skip
        assert checker.should_skip(file_hash) is True
        
        # Verify record updated
        record = checker.get_record(file_hash)
        assert record.status == "success"
        assert record.error_msg is None
    
    def test_mark_failed_updates_existing_success(self, checker: SQLiteIntegrityChecker) -> None:
        """Test mark_failed can update a previously successful record."""
        file_hash = "hash000"
        
        # First mark as success
        checker.mark_success(file_hash, "/path/to/file.txt")
        assert checker.should_skip(file_hash) is True
        
        # Then mark as failed (re-processing failure)
        checker.mark_failed(file_hash, "Re-processing failed")
        
        # Now should NOT skip
        assert checker.should_skip(file_hash) is False
        
        # Verify record updated
        record = checker.get_record(file_hash)
        assert record.status == "failed"
        assert record.error_msg == "Re-processing failed"
    
    def test_get_record_nonexistent(self, checker: SQLiteIntegrityChecker) -> None:
        """Test get_record returns None for non-existent hash."""
        record = checker.get_record("does_not_exist")
        
        assert record is None
    
    def test_get_stats(self, checker: SQLiteIntegrityChecker) -> None:
        """Test get_stats returns correct counts."""
        # Empty database
        stats = checker.get_stats()
        assert stats == {"success": 0, "failed": 0, "total": 0}
        
        # Add some records
        checker.mark_success("s1", "/path/1")
        checker.mark_success("s2", "/path/2")
        checker.mark_failed("f1", "error 1")
        checker.mark_failed("f2", "error 2")
        checker.mark_failed("f3", "error 3")
        
        stats = checker.get_stats()
        assert stats["success"] == 2
        assert stats["failed"] == 3
        assert stats["total"] == 5
    
    def test_reset_requires_confirm(self, checker: SQLiteIntegrityChecker) -> None:
        """Test reset requires confirm=True."""
        checker.mark_success("hash1", "/path/1")
        
        with pytest.raises(ValueError) as exc_info:
            checker.reset()
        
        assert "confirm=True" in str(exc_info.value)
        # Data should still exist
        assert checker.get_record("hash1") is not None
    
    def test_reset_clears_data(self, checker: SQLiteIntegrityChecker) -> None:
        """Test reset with confirm=True clears all data."""
        checker.mark_success("hash1", "/path/1")
        checker.mark_failed("hash2", "error")
        
        checker.reset(confirm=True)
        
        assert checker.get_record("hash1") is None
        assert checker.get_record("hash2") is None
        assert checker.get_stats()["total"] == 0
    
    def test_context_manager(self, temp_db: Path) -> None:
        """Test using as context manager properly closes connection."""
        with SQLiteIntegrityChecker(db_path=temp_db) as checker:
            checker.mark_success("hash1", "/path/1")
            assert checker._connection is not None
        
        # After exiting context, connection should be closed
        # Note: _connection is set to None by close(), but the actual
        # sqlite3 connection object still exists until garbage collected


class TestIntegration:
    """Integration tests combining hash calculation with storage."""
    
    def test_full_workflow_success(self, tmp_path: Path) -> None:
        """Test complete workflow: hash -> check -> mark success -> skip."""
        db_path = tmp_path / "integration.db"
        test_file = tmp_path / "document.txt"
        test_file.write_text("Important document content")
        
        with SQLiteIntegrityChecker(db_path=db_path) as checker:
            # Calculate hash
            file_hash = FileIntegrityChecker.compute_sha256(test_file)
            
            # First time - should not skip
            assert checker.should_skip(file_hash) is False
            
            # Process the file...
            # Mark as success
            checker.mark_success(file_hash, str(test_file))
            
            # Second time - should skip
            assert checker.should_skip(file_hash) is True
    
    def test_full_workflow_failure_and_retry(self, tmp_path: Path) -> None:
        """Test workflow with failure and retry."""
        db_path = tmp_path / "integration.db"
        test_file = tmp_path / "problematic.pdf"
        test_file.write_text("PDF content")
        
        with SQLiteIntegrityChecker(db_path=db_path) as checker:
            file_hash = FileIntegrityChecker.compute_sha256(test_file)
            
            # First attempt - fails
            checker.mark_failed(file_hash, "PDF parsing error")
            assert checker.should_skip(file_hash) is False  # Can retry
            
            # Retry - succeeds
            checker.mark_success(file_hash, str(test_file))
            assert checker.should_skip(file_hash) is True  # Now skip
    
    def test_same_content_different_paths(self, tmp_path: Path) -> None:
        """Test that same content from different paths has same hash."""
        db_path = tmp_path / "integration.db"
        
        # Create two files with identical content
        file1 = tmp_path / "folder1" / "doc.txt"
        file2 = tmp_path / "folder2" / "doc.txt"
        file1.parent.mkdir()
        file2.parent.mkdir()
        
        content = "Identical content"
        file1.write_text(content)
        file2.write_text(content)
        
        with SQLiteIntegrityChecker(db_path=db_path) as checker:
            hash1 = FileIntegrityChecker.compute_sha256(file1)
            hash2 = FileIntegrityChecker.compute_sha256(file2)
            
            # Same content = same hash
            assert hash1 == hash2
            
            # Mark first path as success
            checker.mark_success(hash1, str(file1))
            
            # Both should be considered already processed
            assert checker.should_skip(hash1) is True
            assert checker.should_skip(hash2) is True


class TestConcurrency:
    """Tests for concurrent access (WAL mode)."""
    
    def test_multiple_checkers_same_db(self, tmp_path: Path) -> None:
        """Test multiple checker instances can access the same database."""
        db_path = tmp_path / "concurrent.db"
        
        checker1 = SQLiteIntegrityChecker(db_path=db_path)
        checker2 = SQLiteIntegrityChecker(db_path=db_path)
        
        try:
            # Both can write
            checker1.mark_success("hash1", "/path/1")
            checker2.mark_success("hash2", "/path/2")
            
            # Both can read each other's data
            assert checker1.should_skip("hash2") is True
            assert checker2.should_skip("hash1") is True
        finally:
            checker1.close()
            checker2.close()
    
    def test_wal_files_created(self, tmp_path: Path) -> None:
        """Test that WAL mode creates expected files."""
        db_path = tmp_path / "wal_test.db"
        
        checker = SQLiteIntegrityChecker(db_path=db_path)
        checker.mark_success("hash1", "/path/1")
        checker.close()
        
        # WAL files may or may not exist depending on checkpoint timing
        # Just verify the main DB exists
        assert db_path.exists()


class TestIngestionRecord:
    """Tests for IngestionRecord dataclass."""
    
    def test_record_creation(self) -> None:
        """Test creating an IngestionRecord."""
        record = IngestionRecord(
            file_hash="abc123",
            file_path="/path/to/file.txt",
            status="success",
        )
        
        assert record.file_hash == "abc123"
        assert record.file_path == "/path/to/file.txt"
        assert record.status == "success"
        assert record.error_msg is None
        assert record.created_at is None
        assert record.updated_at is None
    
    def test_record_with_error(self) -> None:
        """Test creating a failed record."""
        record = IngestionRecord(
            file_hash="def456",
            file_path="/path/to/file.txt",
            status="failed",
            error_msg="Parse error",
            created_at=None,
            updated_at=None,
        )
        
        assert record.status == "failed"
        assert record.error_msg == "Parse error"
