#!/usr/bin/env python3
"""
BM25 Indexer - Inverted index builder and querier for sparse retrieval.

Builds and maintains an inverted index from term statistics for BM25 ranking.
Supports index persistence, incremental updates, and document removal.

The index structure:
{
    "_meta": {
        "N": total_documents,
        "avgdl": average_document_length,
        "version": "1.0"
    },
    "terms": {
        "term1": {
            "idf": float,
            "postings": [
                {"chunk_id": str, "tf": int, "doc_length": int, "source": str}
            ]
        },
        ...
    }
}
"""

import json
import math
import os
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from core.types import ChunkRecord


@dataclass
class Posting:
    """A single posting in the inverted index."""
    chunk_id: str
    tf: int  # Term frequency in this document
    doc_length: int  # Total terms in document
    source: str  # Source document path


@dataclass
class TermInfo:
    """Information about a term in the index."""
    idf: float
    postings: List[Posting]


class BM25Indexer:
    """
    BM25 Inverted Index manager.
    
    Builds and maintains an inverted index from sparse vector records.
    Supports index persistence to SQLite or pickle files, incremental updates,
    and BM25 scoring for query retrieval.
    
    BM25 formula: score = IDF * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl/avgdl))
    
    Attributes:
        index_path: Path to store the index database
        k1: BM25 parameter for term frequency saturation (default 1.5)
        b: BM25 parameter for length normalization (default 0.75)
        
    Example:
        >>> indexer = BM25Indexer("data/db/bm25")
        >>> records = [ChunkRecord(id="c1", text="hello world", sparse_vector={...})]
        >>> indexer.add_documents(records, source="doc1.pdf")
        >>> results = indexer.query(["hello"], top_k=10)
    """
    
    DEFAULT_K1 = 1.5
    DEFAULT_B = 0.75
    
    def __init__(
        self,
        index_path: str = "data/db/bm25",
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        use_sqlite: bool = True
    ) -> None:
        """
        Initialize the BM25 indexer.
        
        Args:
            index_path: Directory to store index files
            k1: BM25 term frequency saturation parameter
            b: BM25 length normalization parameter
            use_sqlite: Whether to use SQLite backend (True) or pickle (False)
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.k1 = k1
        self.b = b
        self.use_sqlite = use_sqlite
        
        # In-memory cache of the index
        self._terms: Dict[str, TermInfo] = {}
        self._N = 0  # Total document count
        self._total_dl = 0  # Sum of all document lengths
        
        # Load existing index if present
        self._load_index()
    
    @property
    def N(self) -> int:
        """Total number of documents in the index."""
        return self._N
    
    @property
    def avgdl(self) -> float:
        """Average document length."""
        return self._total_dl / self._N if self._N > 0 else 0.0
    
    def _get_db_path(self) -> Path:
        """Get the database file path."""
        if self.use_sqlite:
            return self.index_path / "index.db"
        return self.index_path / "index.pkl"
    
    def _load_index(self) -> None:
        """Load existing index from storage."""
        db_path = self._get_db_path()
        if not db_path.exists():
            return
        
        if self.use_sqlite:
            self._load_from_sqlite()
        else:
            self._load_from_pickle()
    
    def _load_from_sqlite(self) -> None:
        """Load index from SQLite database."""
        db_path = self._get_db_path()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Load metadata
        cursor.execute("SELECT key, value FROM metadata")
        meta = {row[0]: row[1] for row in cursor.fetchall()}
        self._N = int(meta.get("N", 0))
        self._total_dl = int(meta.get("total_dl", 0))
        
        # Load terms and postings
        cursor.execute("SELECT term, idf FROM terms")
        for term, idf in cursor.fetchall():
            cursor.execute(
                "SELECT chunk_id, tf, doc_length, source FROM postings WHERE term = ?",
                (term,)
            )
            postings = [
                Posting(chunk_id=row[0], tf=row[1], doc_length=row[2], source=row[3])
                for row in cursor.fetchall()
            ]
            self._terms[term] = TermInfo(idf=idf, postings=postings)
        
        conn.close()
    
    def _load_from_pickle(self) -> None:
        """Load index from pickle file."""
        db_path = self._get_db_path()
        with open(db_path, "rb") as f:
            data = pickle.load(f)
        
        self._N = data["_meta"]["N"]
        self._total_dl = data["_meta"].get("total_dl", 0)
        
        for term, info in data["terms"].items():
            postings = [Posting(**p) for p in info["postings"]]
            self._terms[term] = TermInfo(idf=info["idf"], postings=postings)
    
    def _save_index(self) -> None:
        """Save index to storage."""
        if self.use_sqlite:
            self._save_to_sqlite()
        else:
            self._save_to_pickle()
    
    def _save_to_sqlite(self) -> None:
        """Save index to SQLite database."""
        db_path = self._get_db_path()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS terms (
                term TEXT PRIMARY KEY,
                idf REAL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS postings (
                term TEXT,
                chunk_id TEXT,
                tf INTEGER,
                doc_length INTEGER,
                source TEXT,
                PRIMARY KEY (term, chunk_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term)")
        
        # Clear existing data
        cursor.execute("DELETE FROM metadata")
        cursor.execute("DELETE FROM terms")
        cursor.execute("DELETE FROM postings")
        
        # Save metadata
        cursor.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ("N", str(self._N))
        )
        cursor.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ("total_dl", str(self._total_dl))
        )
        
        # Save terms and postings
        for term, info in self._terms.items():
            cursor.execute(
                "INSERT INTO terms (term, idf) VALUES (?, ?)",
                (term, info.idf)
            )
            for posting in info.postings:
                cursor.execute(
                    "INSERT INTO postings (term, chunk_id, tf, doc_length, source) VALUES (?, ?, ?, ?, ?)",
                    (term, posting.chunk_id, posting.tf, posting.doc_length, posting.source)
                )
        
        conn.commit()
        conn.close()
    
    def _save_to_pickle(self) -> None:
        """Save index to pickle file."""
        data = {
            "_meta": {
                "N": self._N,
                "total_dl": self._total_dl,
                "avgdl": self.avgdl,
                "version": "1.0"
            },
            "terms": {}
        }
        
        for term, info in self._terms.items():
            data["terms"][term] = {
                "idf": info.idf,
                "postings": [asdict(p) for p in info.postings]
            }
        
        db_path = self._get_db_path()
        with open(db_path, "wb") as f:
            pickle.dump(data, f)
    
    def _compute_idf(self, df: int) -> float:
        """
        Compute IDF (Inverse Document Frequency).
        
        Formula: IDF = log((N - df + 0.5) / (df + 0.5))
        
        Args:
            df: Document frequency (number of documents containing the term)
            
        Returns:
            IDF score
        """
        if self._N == 0:
            return 0.0
        return math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)
    
    def add_documents(
        self,
        records: List[ChunkRecord],
        source: str = "unknown"
    ) -> None:
        """
        Add documents to the index.
        
        Args:
            records: List of ChunkRecords with sparse_vector populated
            source: Source document path for all records
            
        Raises:
            ValueError: If a record lacks sparse_vector
        """
        # Group postings by term
        term_postings: Dict[str, List[Posting]] = {}
        
        for record in records:
            if not record.sparse_vector:
                raise ValueError(f"Record {record.id} missing sparse_vector")
            
            sv = record.sparse_vector
            terms = sv.get("terms", [])
            tfs = sv.get("tf", [])
            doc_length = sv.get("doc_length", 0)
            
            for term, tf in zip(terms, tfs):
                posting = Posting(
                    chunk_id=record.id,
                    tf=tf,
                    doc_length=doc_length,
                    source=source
                )
                if term not in term_postings:
                    term_postings[term] = []
                term_postings[term].append(posting)
            
            self._N += 1
            self._total_dl += doc_length
        
        # Update index with new postings
        for term, new_postings in term_postings.items():
            if term not in self._terms:
                self._terms[term] = TermInfo(idf=0.0, postings=[])
            
            # Add new postings
            self._terms[term].postings.extend(new_postings)
            
            # Recompute IDF
            df = len(self._terms[term].postings)
            self._terms[term].idf = self._compute_idf(df)
        
        self._save_index()
    
    def remove_document(self, source: str) -> int:
        """
        Remove all chunks from a source document from the index.
        
        Args:
            source: Source document path to remove
            
        Returns:
            Number of chunks removed
        """
        removed_count = 0
        terms_to_remove = []
        
        for term, info in self._terms.items():
            original_len = len(info.postings)
            info.postings = [p for p in info.postings if p.source != source]
            removed = original_len - len(info.postings)
            removed_count += removed
            
            if removed > 0:
                self._N -= removed
                self._total_dl -= sum(
                    p.doc_length for p in info.postings[:removed]
                )
            
            if not info.postings:
                terms_to_remove.append(term)
            else:
                # Recompute IDF
                df = len(info.postings)
                info.idf = self._compute_idf(df)
        
        # Remove empty terms
        for term in terms_to_remove:
            del self._terms[term]
        
        self._save_index()
        return removed_count
    
    def _compute_bm25_score(
        self,
        tf: int,
        doc_length: int,
        idf: float
    ) -> float:
        """
        Compute BM25 score for a term in a document.
        
        Args:
            tf: Term frequency in document
            doc_length: Total terms in document
            idf: Inverse document frequency
            
        Returns:
            BM25 score
        """
        if self.avgdl == 0:
            return 0.0
        
        # BM25 formula
        numerator = tf * (self.k1 + 1)
        denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avgdl))
        
        return idf * (numerator / denominator) if denominator > 0 else 0.0
    
    def query(
        self,
        keywords: List[str],
        top_k: int = 10,
        case_sensitive: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Query the index for top-k matching documents.
        
        Args:
            keywords: List of query terms
            top_k: Maximum number of results to return
            case_sensitive: Whether matching should be case-sensitive
            
        Returns:
            List of (chunk_id, score) tuples sorted by score descending
        """
        if not keywords:
            return []
        
        # Normalize keywords if case-insensitive
        if not case_sensitive:
            keywords = [k.lower() for k in keywords]
        
        # Accumulate scores per chunk
        scores: Dict[str, float] = {}
        
        for keyword in keywords:
            search_term = keyword if case_sensitive else keyword.lower()
            
            # Find matching terms (exact match or partial match)
            for term, info in self._terms.items():
                term_key = term if case_sensitive else term.lower()
                if search_term == term_key or search_term in term_key:
                    for posting in info.postings:
                        score = self._compute_bm25_score(
                            posting.tf,
                            posting.doc_length,
                            info.idf
                        )
                        scores[posting.chunk_id] = scores.get(posting.chunk_id, 0.0) + score
        
        # Sort by score and return top-k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            "N": self._N,
            "avgdl": self.avgdl,
            "num_terms": len(self._terms),
            "total_dl": self._total_dl,
            "storage": "sqlite" if self.use_sqlite else "pickle"
        }
    
    def rebuild(self) -> None:
        """Rebuild the index (clear and re-add all documents)."""
        self._terms = {}
        self._N = 0
        self._total_dl = 0
        self._save_index()
