#!/usr/bin/env python3
"""
Image Storage - Manages extracted image files and their index.

Stores extracted images to the filesystem and maintains an SQLite index
mapping image_ids to file paths. Supports retrieval by image_id, collection,
or document hash.
"""

import hashlib
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


@dataclass
class ImageRecord:
    """Record of a stored image."""
    image_id: str
    file_path: str
    collection: str
    doc_hash: str
    page_num: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    mime_type: str = "image/png"
    created_at: Optional[datetime] = None


class ImageStorage:
    """
    Manages storage and indexing of extracted document images.
    
    Images are stored in the filesystem at:
        data/images/{collection}/{doc_hash}/{image_id}.png
    
    An SQLite index at data/db/image_index.db maps image_ids to paths
    for fast lookup.
    
    Attributes:
        base_path: Root directory for image storage
        db_path: Path to SQLite index database
        
    Example:
        >>> storage = ImageStorage()
        >>> record = storage.save_image("path/to/input.png", "doc123", "default")
        >>> retrieved_path = storage.get_image_path(record.image_id)
    """
    
    DEFAULT_BASE_PATH = "data/images"
    DEFAULT_DB_PATH = "data/db/image_index.db"
    
    def __init__(
        self,
        base_path: str = DEFAULT_BASE_PATH,
        db_path: str = DEFAULT_DB_PATH
    ) -> None:
        """
        Initialize image storage.
        
        Args:
            base_path: Root directory for storing image files
            db_path: Path to SQLite index database
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the SQLite database with WAL mode."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Enable WAL mode for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        
        # Create image index table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_index (
                image_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                collection TEXT,
                doc_hash TEXT,
                page_num INTEGER,
                width INTEGER,
                height INTEGER,
                mime_type TEXT DEFAULT 'image/png',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for fast lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_collection ON image_index(collection)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON image_index(doc_hash)")
        
        conn.commit()
        conn.close()
    
    def _generate_image_id(
        self,
        doc_hash: str,
        page_num: Optional[int],
        seq: int,
        collection: str = "default"
    ) -> str:
        """
        Generate a unique image ID.
        
        Format: {collection}_{doc_hash}_{page}_{seq}
        
        Args:
            doc_hash: Hash of the source document
            page_num: Page number where image appears (optional)
            seq: Sequence number on the page
            collection: Collection name
            
        Returns:
            Unique image identifier
        """
        page_str = f"p{page_num}" if page_num is not None else "p0"
        return f"{collection}_{doc_hash}_{page_str}_{seq:03d}"
    
    def save_image(
        self,
        source_path: str,
        doc_hash: str,
        collection: str = "default",
        page_num: Optional[int] = None,
        seq: int = 0,
        convert_to_png: bool = True
    ) -> ImageRecord:
        """
        Save an image to storage and index it.
        
        Args:
            source_path: Path to the source image file
            doc_hash: Hash of the parent document
            collection: Collection/namespace for the image
            page_num: Page number in the source document
            seq: Sequence number for multiple images on same page
            convert_to_png: Whether to convert image to PNG format
            
        Returns:
            ImageRecord with storage details
            
        Raises:
            FileNotFoundError: If source image doesn't exist
            IOError: If image processing fails
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source image not found: {source_path}")
        
        # Generate image ID
        image_id = self._generate_image_id(doc_hash, page_num, seq, collection)
        
        # Determine target path
        target_dir = self.base_path / collection / doc_hash
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Process and save image
        if convert_to_png:
            target_path = target_dir / f"{image_id}.png"
            with Image.open(source) as img:
                # Convert to RGB if necessary (handles RGBA, palette, etc.)
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                img.save(target_path, 'PNG')
                width, height = img.size
                mime_type = "image/png"
        else:
            # Keep original format
            ext = source.suffix or ".png"
            target_path = target_dir / f"{image_id}{ext}"
            shutil.copy2(source, target_path)
            with Image.open(target_path) as img:
                width, height = img.size
                mime_type = f"image/{img.format.lower()}" if img.format else "image/png"
        
        # Create record
        record = ImageRecord(
            image_id=image_id,
            file_path=str(target_path),
            collection=collection,
            doc_hash=doc_hash,
            page_num=page_num,
            width=width,
            height=height,
            mime_type=mime_type,
            created_at=datetime.now()
        )
        
        # Save to index
        self._save_to_index(record)
        
        return record
    
    def index_existing_image(
        self,
        image_path: str,
        image_id: str,
        doc_hash: str,
        collection: str = "default",
        page_num: Optional[int] = None
    ) -> ImageRecord:
        """
        Index an already-existing image file without copying it.
        
        Use this when the image is already saved to the filesystem
        (e.g., by PDF loader) and just needs to be indexed.
        
        Args:
            image_path: Path to the existing image file
            image_id: Unique image identifier
            doc_hash: Hash of the parent document
            collection: Collection/namespace for the image
            page_num: Page number in the source document
            
        Returns:
            ImageRecord with storage details
            
        Raises:
            FileNotFoundError: If image doesn't exist
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Get image dimensions
        try:
            with Image.open(path) as img:
                width, height = img.size
                mime_type = f"image/{img.format.lower()}" if img.format else "image/png"
        except Exception:
            width, height = None, None
            mime_type = "image/png"
        
        # Create record
        record = ImageRecord(
            image_id=image_id,
            file_path=str(path.absolute()),
            collection=collection,
            doc_hash=doc_hash,
            page_num=page_num,
            width=width,
            height=height,
            mime_type=mime_type,
            created_at=datetime.now()
        )
        
        # Save to index
        self._save_to_index(record)
        
        return record
    
    def _save_to_index(self, record: ImageRecord) -> None:
        """Save image record to SQLite index."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT OR REPLACE INTO image_index 
            (image_id, file_path, collection, doc_hash, page_num, width, height, mime_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.image_id,
                record.file_path,
                record.collection,
                record.doc_hash,
                record.page_num,
                record.width,
                record.height,
                record.mime_type,
                record.created_at.isoformat() if record.created_at else None
            )
        )
        
        conn.commit()
        conn.close()
    
    def get_image_path(self, image_id: str) -> Optional[str]:
        """
        Get the file path for an image ID.
        
        Args:
            image_id: Unique image identifier
            
        Returns:
            File path if found, None otherwise
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT file_path FROM image_index WHERE image_id = ?",
            (image_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def get_image_record(self, image_id: str) -> Optional[ImageRecord]:
        """
        Get full record for an image ID.
        
        Args:
            image_id: Unique image identifier
            
        Returns:
            ImageRecord if found, None otherwise
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT image_id, file_path, collection, doc_hash, page_num, 
                   width, height, mime_type, created_at
            FROM image_index WHERE image_id = ?
            """,
            (image_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        return ImageRecord(
            image_id=result[0],
            file_path=result[1],
            collection=result[2],
            doc_hash=result[3],
            page_num=result[4],
            width=result[5],
            height=result[6],
            mime_type=result[7],
            created_at=datetime.fromisoformat(result[8]) if result[8] else None
        )
    
    def list_images(
        self,
        collection: Optional[str] = None,
        doc_hash: Optional[str] = None
    ) -> List[ImageRecord]:
        """
        List images matching filters.
        
        Args:
            collection: Filter by collection
            doc_hash: Filter by document hash
            
        Returns:
            List of matching ImageRecords
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        query = """
            SELECT image_id, file_path, collection, doc_hash, page_num,
                   width, height, mime_type, created_at
            FROM image_index
            WHERE 1=1
        """
        params = []
        
        if collection:
            query += " AND collection = ?"
            params.append(collection)
        
        if doc_hash:
            query += " AND doc_hash = ?"
            params.append(doc_hash)
        
        query += " ORDER BY created_at DESC"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return [
            ImageRecord(
                image_id=row[0],
                file_path=row[1],
                collection=row[2],
                doc_hash=row[3],
                page_num=row[4],
                width=row[5],
                height=row[6],
                mime_type=row[7],
                created_at=datetime.fromisoformat(row[8]) if row[8] else None
            )
            for row in results
        ]
    
    def delete_images(
        self,
        collection: Optional[str] = None,
        doc_hash: Optional[str] = None
    ) -> int:
        """
        Delete images matching filters.
        
        Args:
            collection: Filter by collection
            doc_hash: Filter by document hash
            
        Returns:
            Number of images deleted
        """
        # First get the files to delete
        images = self.list_images(collection, doc_hash)
        
        # Delete files from filesystem
        for img in images:
            path = Path(img.file_path)
            if path.exists():
                path.unlink()
        
        # Clean up empty directories
        if doc_hash and collection:
            doc_dir = self.base_path / collection / doc_hash
            if doc_dir.exists() and not any(doc_dir.iterdir()):
                doc_dir.rmdir()
                coll_dir = self.base_path / collection
                if coll_dir.exists() and not any(coll_dir.iterdir()):
                    coll_dir.rmdir()
        
        # Remove from index
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        query = "DELETE FROM image_index WHERE 1=1"
        params = []
        
        if collection:
            query += " AND collection = ?"
            params.append(collection)
        
        if doc_hash:
            query += " AND doc_hash = ?"
            params.append(doc_hash)
        
        cursor.execute(query, params)
        deleted = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return deleted
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with statistics
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM image_index")
        total_images = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT collection) FROM image_index")
        num_collections = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT doc_hash) FROM image_index")
        num_documents = cursor.fetchone()[0]
        
        conn.close()
        
        # Calculate total size
        total_size = 0
        for path in self.base_path.rglob("*"):
            if path.is_file():
                total_size += path.stat().st_size
        
        return {
            "total_images": total_images,
            "num_collections": num_collections,
            "num_documents": num_documents,
            "storage_path": str(self.base_path),
            "total_size_bytes": total_size
        }
