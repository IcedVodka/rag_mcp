#!/usr/bin/env python3
"""
Sparse Encoder - BM25 term statistics generation for chunks.

Converts chunks to ChunkRecords with sparse vector representations
containing term frequency statistics needed for BM25 indexing.
Supports both Chinese (jieba) and English tokenization.
"""

import re
from collections import Counter
from typing import Optional, List, Dict, Any

from core.settings import Settings, load_settings
from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord


class SparseEncoder:
    """
    Encoder for generating BM25 term statistics from chunks.
    
    Generates sparse vector representations containing term frequencies
    and document length information required for BM25 indexing. Supports
    multiple tokenization strategies (jieba for Chinese, simple whitespace
    for English) and stopword filtering.
    
    The sparse_vector output follows this structure:
    {
        "terms": List[str],      # Unique terms in the document
        "tf": List[int],         # Term frequencies (aligned with terms)
        "doc_length": int        # Total term count in document
    }
    
    Attributes:
        tokenizer: Tokenization strategy ('jieba' or 'simple')
        stopwords: Set of stopwords to filter out
        use_stopwords: Whether to enable stopword filtering
        
    Example:
        >>> settings = load_settings("config/settings.yaml")
        >>> encoder = SparseEncoder(settings)
        >>> chunks = [Chunk(id="c1", text="Hello world")]
        >>> records = encoder.encode(chunks)
        >>> print(records[0].sparse_vector)
        {'terms': ['hello', 'world'], 'tf': [1, 1], 'doc_length': 2}
    """
    
    # Default English stopwords
    DEFAULT_STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
        'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
        'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
        'her', 'would', 'there', 'been', 'has', 'more', 'very', 'can',
        'him', 'his', 'did', 'get', 'come', 'made', 'may', 'say', 'or',
        'who', 'about', 'when', 'where', 'why', 'way', 'all', 'any',
        'both', 'were', 'we', 'you', 'your', 'me', 'my', 'mine', 'i',
        'over', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'again', 'further', 'once', 'here',
        'why', 'off', 'own', 'same', 'few', 'most', 'other', 'such',
        'only', 'just', 'now', 'than', 'too', 'also', 'back', 'still',
        'being', 'having', 'does', 'doing', 'done', 'am', 'being'
    }
    
    # Default Chinese stopwords (common function words)
    DEFAULT_CHINESE_STOPWORDS = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
        '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
        '你', '会', '着', '没有', '看', '好', '自己', '这', '那',
        '这些', '那些', '这个', '那个', '之', '与', '及', '等', '或',
        '但是', '而', '如果', '因为', '所以', '虽然', '但是', '可以',
        '被', '把', '给', '让', '向', '从', '到', '为', '以', '将'
    }
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        tokenizer: str = "jieba",
        stopwords: bool = True,
        custom_stopwords: Optional[set[str]] = None
    ) -> None:
        """
        Initialize the sparse encoder.
        
        Args:
            settings: Application settings. If provided and contains sparse_encoder
                config, those settings will be used as defaults.
            tokenizer: Tokenization strategy ('jieba' or 'simple').
                'jieba' uses jieba for Chinese text segmentation.
                'simple' uses whitespace splitting (good for English).
            stopwords: Whether to enable stopword filtering.
            custom_stopwords: Additional stopwords to use beyond defaults.
        """
        # Load config from settings if available
        config = self._load_config(settings)
        
        self.tokenizer = config.get('tokenizer', tokenizer)
        self.use_stopwords = config.get('stopwords', stopwords)
        
        # Initialize jieba if needed
        self._jieba = None
        if self.tokenizer == 'jieba':
            try:
                import jieba
                self._jieba = jieba
            except ImportError:
                raise ImportError(
                    "jieba is required for Chinese tokenization. "
                    "Install with: pip install jieba"
                )
        
        # Build stopword set
        self.stopwords: set[str] = set()
        if self.use_stopwords:
            self.stopwords.update(self.DEFAULT_STOPWORDS)
            self.stopwords.update(self.DEFAULT_CHINESE_STOPWORDS)
            if custom_stopwords:
                self.stopwords.update(custom_stopwords)
    
    def _load_config(self, settings: Optional[Settings]) -> Dict[str, Any]:
        """Load sparse encoder configuration from settings."""
        if settings is None:
            return {}
        
        # Try to get ingestion.sparse_encoder config
        ingestion_config = getattr(settings, 'ingestion', None)
        if ingestion_config:
            config = getattr(ingestion_config, 'sparse_encoder', None)
            if config:
                return config if isinstance(config, dict) else {}
        
        return {}
    
    def encode(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[ChunkRecord]:
        """
        Encode chunks into ChunkRecords with sparse vectors.
        
        Processes all chunks and generates term frequency statistics
        for BM25 indexing.
        
        Args:
            chunks: List of chunks to encode
            trace: Optional trace context for observability
            
        Returns:
            List of ChunkRecord objects with sparse_vector populated.
            Returns empty list if input is empty.
            
        Note:
            - dense_vector is always None (SparseEncoder doesn't generate dense vectors)
            - Output list maintains same order as input chunks
            - Empty text results in sparse_vector with empty terms and zero doc_length
        """
        if not chunks:
            return []
        
        return self.encode_batch(chunks, trace)
    
    def encode_batch(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[ChunkRecord]:
        """
        Encode a single batch of chunks into ChunkRecords.
        
        Generates sparse vector representations (term statistics) for
        each chunk's text content.
        
        Args:
            chunks: List of chunks to encode
            trace: Optional trace context for observability
            
        Returns:
            List of ChunkRecord objects with sparse_vector populated.
            Returns empty list if input is empty.
            
        Raises:
            ValueError: If chunk text is None
        """
        if not chunks:
            return []
        
        # Record trace if provided
        if trace:
            trace.record_stage(
                name="sparse_encoding",
                method="tokenize_and_count",
                provider=f"tokenizer:{self.tokenizer}",
                details={
                    "tokenizer": self.tokenizer,
                    "stopwords_enabled": self.use_stopwords,
                    "stopwords_count": len(self.stopwords) if self.use_stopwords else 0,
                    "chunk_count": len(chunks)
                }
            )
        
        # Process each chunk
        records: List[ChunkRecord] = []
        for chunk in chunks:
            # Compute term statistics
            term_stats = self._compute_term_stats_for_text(chunk.text)
            
            record = ChunkRecord(
                id=chunk.id,
                text=chunk.text,
                metadata={
                    **chunk.metadata,
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                    "source_ref": chunk.source_ref,
                },
                dense_vector=None,  # SparseEncoder doesn't generate dense vectors
                sparse_vector=term_stats
            )
            records.append(record)
        
        return records
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms.
        
        Uses the configured tokenizer strategy:
        - 'jieba': Uses jieba for Chinese text segmentation
        - 'simple': Uses whitespace and punctuation splitting
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens (lowercased, filtered for empty strings)
            
        Note:
            Tokens are lowercased and non-alphanumeric tokens
            are filtered out for 'simple' tokenizer.
        """
        if not text or not text.strip():
            return []
        
        if self.tokenizer == 'jieba':
            # Use jieba for Chinese tokenization
            tokens = list(self._jieba.cut(text.strip()))
            # Filter and normalize
            tokens = [t.lower().strip() for t in tokens if t.strip()]
        else:
            # Simple whitespace tokenization
            # Normalize whitespace and split
            text = text.lower().strip()
            # Replace non-alphanumeric (and non-Chinese) with space
            text = re.sub(r'[^\w\u4e00-\u9fff]+', ' ', text)
            tokens = [t.strip() for t in text.split() if t.strip()]
        
        return tokens
    
    def _compute_term_stats(self, tokens: List[str]) -> Dict[str, int]:
        """
        Compute term frequency statistics from tokens.
        
        Counts occurrences of each term after filtering stopwords.
        
        Args:
            tokens: List of tokens from tokenization
            
        Returns:
            Dictionary mapping term -> frequency count
        """
        if not tokens:
            return {}
        
        # Filter stopwords if enabled
        if self.use_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        
        # Count term frequencies
        return dict(Counter(tokens))
    
    def _compute_term_stats_for_text(self, text: str) -> Dict[str, Any]:
        """
        Compute complete term statistics for a text.
        
        This is the main method that combines tokenization and counting
        to produce the sparse vector structure for BM25.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with structure:
            {
                "terms": List[str],      # Unique terms
                "tf": List[int],         # Term frequencies
                "doc_length": int        # Total term count (after stopword filtering)
            }
        """
        tokens = self._tokenize(text)
        
        if not tokens:
            return {
                "terms": [],
                "tf": [],
                "doc_length": 0
            }
        
        # Filter stopwords and count
        if self.use_stopwords:
            filtered_tokens = [t for t in tokens if t not in self.stopwords]
        else:
            filtered_tokens = tokens
        
        term_freq = Counter(filtered_tokens)
        
        # Build output structure
        terms = list(term_freq.keys())
        tf = [term_freq[t] for t in terms]
        
        return {
            "terms": terms,
            "tf": tf,
            "doc_length": len(filtered_tokens)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get encoder configuration statistics.
        
        Returns:
            Dictionary with encoder configuration details
        """
        return {
            "tokenizer": self.tokenizer,
            "use_stopwords": self.use_stopwords,
            "stopword_count": len(self.stopwords),
            "jieba_available": self._jieba is not None
        }
