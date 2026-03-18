#!/usr/bin/env python3
"""
Image Storage Tests

Tests the ImageStorage class for saving, retrieving, and managing extracted images.
"""

import tempfile
from pathlib import Path
from PIL import Image
import pytest

from ingestion.storage.image_storage import ImageStorage, ImageRecord


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img = Image.new("RGB", (100, 50), color="red")
        img.save(tmp.name)
        yield tmp.name
        Path(tmp.name).unlink(missing_ok=True)


class TestImageStorageInit:
    """Test ImageStorage initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(
                base_path=tmpdir,
                db_path=Path(tmpdir) / "image_index.db"
            )
            assert storage.base_path.exists()
            assert storage.db_path.exists()
    
    def test_creates_directories(self):
        """Test that storage creates necessary directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "images"
            db = Path(tmpdir) / "db" / "index.db"
            storage = ImageStorage(base_path=str(base), db_path=str(db))
            assert base.exists()
            assert db.parent.exists()


class TestSaveImage:
    """Test saving images."""
    
    def test_save_image(self, sample_image):
        """Test saving an image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(
                base_path=tmpdir,
                db_path=Path(tmpdir) / "image_index.db"
            )
            
            record = storage.save_image(
                source_path=sample_image,
                doc_hash="abc123",
                collection="default"
            )
            
            assert "abc123_" in record.image_id
            assert Path(record.file_path).exists()
            assert record.mime_type == "image/png"
    
    def test_save_image_with_page_num(self, sample_image):
        """Test saving image with page number."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(
                base_path=tmpdir,
                db_path=Path(tmpdir) / "image_index.db"
            )
            
            record = storage.save_image(
                source_path=sample_image,
                doc_hash="abc123",
                collection="default",
                page_num=5,
                seq=2
            )
            
            assert "p5" in record.image_id
            assert "002" in record.image_id
    
    def test_save_image_converts_to_png(self, sample_image):
        """Test that images are converted to PNG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(
                base_path=tmpdir,
                db_path=Path(tmpdir) / "image_index.db"
            )
            
            record = storage.save_image(
                source_path=sample_image,
                doc_hash="abc123",
                collection="default"
            )
            
            assert record.file_path.endswith(".png")
    
    def test_save_image_tracks_dimensions(self, sample_image):
        """Test that image dimensions are tracked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(
                base_path=tmpdir,
                db_path=Path(tmpdir) / "image_index.db"
            )
            
            record = storage.save_image(
                source_path=sample_image,
                doc_hash="abc123",
                collection="default"
            )
            
            assert record.width == 100
            assert record.height == 50
    
    def test_save_image_missing_file_raises(self):
        """Test that saving non-existent image raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(
                base_path=tmpdir,
                db_path=Path(tmpdir) / "image_index.db"
            )
            
            with pytest.raises(FileNotFoundError):
                storage.save_image(
                    source_path="/nonexistent/image.png",
                    doc_hash="abc123",
                    collection="default"
                )


class TestGetImagePath:
    """Test retrieving image paths."""
    
    def test_get_existing_image(self, sample_image):
        """Test getting path for existing image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(
                base_path=tmpdir,
                db_path=Path(tmpdir) / "image_index.db"
            )
            
            record = storage.save_image(
                source_path=sample_image,
                doc_hash="abc123",
                collection="default"
            )
            
            path = storage.get_image_path(record.image_id)
            assert path == record.file_path
    
    def test_get_nonexistent_image(self, sample_image):
        """Test getting path for non-existent image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(
                base_path=tmpdir,
                db_path=Path(tmpdir) / "image_index.db"
            )
            
            path = storage.get_image_path("nonexistent_id")
            assert path is None


class TestGetImageRecord:
    """Test retrieving full image records."""
    
    def test_get_existing_record(self, sample_image):
        """Test getting full record for existing image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(
                base_path=tmpdir,
                db_path=Path(tmpdir) / "image_index.db"
            )
            
            saved_record = storage.save_image(
                source_path=sample_image,
                doc_hash="abc123",
                collection="default",
                page_num=3
            )
            
            record = storage.get_image_record(saved_record.image_id)
            assert record is not None
            assert record.image_id == saved_record.image_id
            assert record.doc_hash == "abc123"
            assert record.collection == "default"
            assert record.page_num == 3
    
    def test_get_nonexistent_record(self, sample_image):
        """Test getting record for non-existent image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(
                base_path=tmpdir,
                db_path=Path(tmpdir) / "image_index.db"
            )
            
            record = storage.get_image_record("nonexistent_id")
            assert record is None


class TestListImages:
    """Test listing images."""
    
    def test_list_all_images(self, sample_image):
        """Test listing all images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(
                base_path=tmpdir,
                db_path=Path(tmpdir) / "image_index.db"
            )
            
            storage.save_image(sample_image, "doc1", "coll1")
            storage.save_image(sample_image, "doc2", "coll2")
            
            images = storage.list_images()
            assert len(images) == 2
    
    def test_list_by_collection(self, sample_image):
        """Test listing images by collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(
                base_path=tmpdir,
                db_path=Path(tmpdir) / "image_index.db"
            )
            
            storage.save_image(sample_image, "doc1", "coll1")
            storage.save_image(sample_image, "doc2", "coll1")
            storage.save_image(sample_image, "doc3", "coll2")
            
            images = storage.list_images(collection="coll1")
            assert len(images) == 2
            for img in images:
                assert img.collection == "coll1"
    
    def test_list_by_doc_hash(self, sample_image):
        """Test listing images by document hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(
                base_path=tmpdir,
                db_path=Path(tmpdir) / "image_index.db"
            )
            
            storage.save_image(sample_image, "doc1", "coll1", seq=0)
            storage.save_image(sample_image, "doc1", "coll1", seq=1)
            storage.save_image(sample_image, "doc2", "coll1", seq=0)
            
            images = storage.list_images(doc_hash="doc1")
            assert len(images) == 2
            for img in images:
                assert img.doc_hash == "doc1"
    
    def test_list_by_both_filters(self, sample_image):
        """Test listing images by both collection and doc_hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(
                base_path=tmpdir,
                db_path=Path(tmpdir) / "image_index.db"
            )
            
            storage.save_image(sample_image, "doc1", "coll1", seq=0)
            storage.save_image(sample_image, "doc1", "coll2", seq=0)
            storage.save_image(sample_image, "doc2", "coll1", seq=0)
            
            images = storage.list_images(collection="coll1", doc_hash="doc1")
            assert len(images) == 1


class TestDeleteImages:
    """Test deleting images."""
    
    def test_delete_by_collection(self, sample_image):
        """Test deleting images by collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(
                base_path=tmpdir,
                db_path=Path(tmpdir) / "image_index.db"
            )
            
            storage.save_image(sample_image, "doc1", "coll1")
            storage.save_image(sample_image, "doc2", "coll2")
            
            deleted = storage.delete_images(collection="coll1")
            assert deleted == 1
            
            remaining = storage.list_images()
            assert len(remaining) == 1
            assert remaining[0].collection == "coll2"
    
    def test_delete_by_doc_hash(self, sample_image):
        """Test deleting images by document hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(
                base_path=tmpdir,
                db_path=Path(tmpdir) / "image_index.db"
            )
            
            storage.save_image(sample_image, "doc1", "coll1", seq=0)
            storage.save_image(sample_image, "doc1", "coll1", seq=1)
            storage.save_image(sample_image, "doc2", "coll1", seq=0)
            
            deleted = storage.delete_images(doc_hash="doc1")
            assert deleted == 2
            
            remaining = storage.list_images()
            assert len(remaining) == 1
            assert remaining[0].doc_hash == "doc2"
    
    def test_delete_removes_files(self, sample_image):
        """Test that delete removes image files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(
                base_path=tmpdir,
                db_path=Path(tmpdir) / "image_index.db"
            )
            
            record = storage.save_image(sample_image, "doc1", "coll1")
            file_path = Path(record.file_path)
            
            assert file_path.exists()
            storage.delete_images(collection="coll1")
            assert not file_path.exists()


class TestStats:
    """Test statistics."""
    
    def test_get_stats(self, sample_image):
        """Test getting storage statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ImageStorage(
                base_path=tmpdir,
                db_path=Path(tmpdir) / "image_index.db"
            )
            
            storage.save_image(sample_image, "doc1", "coll1")
            storage.save_image(sample_image, "doc2", "coll2")
            
            stats = storage.get_stats()
            assert stats["total_images"] == 2
            assert stats["num_collections"] == 2
            assert stats["num_documents"] == 2
            assert stats["total_size_bytes"] > 0


class TestPersistence:
    """Test data persistence across instances."""
    
    def test_record_persists(self, sample_image):
        """Test that saved images persist across storage instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "image_index.db"
            
            # First instance
            storage1 = ImageStorage(base_path=tmpdir, db_path=db_path)
            record1 = storage1.save_image(sample_image, "doc1", "coll1")
            
            # Second instance
            storage2 = ImageStorage(base_path=tmpdir, db_path=db_path)
            record2 = storage2.get_image_record(record1.image_id)
            
            assert record2 is not None
            assert record2.image_id == record1.image_id
