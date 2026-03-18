#!/usr/bin/env python3
"""
RRF Fusion Module - Reciprocal Rank Fusion for hybrid search results.

Implements the RRF algorithm to combine dense (vector) and sparse (BM25) 
retrieval results into a unified ranked list.

RRF Formula:
    Score = Σ 1 / (k + rank)

Where:
    - k: A constant to control the influence of rank (default: 60)
    - rank: The position of the document in the source ranking (1-indexed)

Reference:
    Cormack, G.V., Clarke, C.L.A., & Buettcher, S. (2009). 
    Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods.
"""

import logging
from typing import List
from dataclasses import replace

from core.types import RetrievalResult
from core.settings import Settings

logger = logging.getLogger(__name__)


class RRFFusion:
    """
    Reciprocal Rank Fusion implementation for combining dense and sparse search results.
    
    This class implements the RRF algorithm to fuse rankings from multiple retrieval
    sources (e.g., dense vector search and sparse BM25 search) into a single,
    unified ranking that leverages the strengths of each approach.
    
    Attributes:
        settings: Application settings container
        k: RRF constant controlling rank influence (default: 60)
    
    Example:
        >>> settings = load_settings("config.yaml")
        >>> fusion = RRFFusion(settings, k=60)
        >>> dense_results = [RetrievalResult(...), ...]
        >>> sparse_results = [RetrievalResult(...), ...]
        >>> fused = fusion.fuse(dense_results, sparse_results, top_k=10)
    """
    
    DEFAULT_K: int = 60
    """Default RRF constant value."""
    
    def __init__(self, settings: Settings, k: int = DEFAULT_K):
        """
        Initialize the RRF fusion component.
        
        Args:
            settings: Application settings container
            k: RRF constant to control rank influence (default: 60)
               Lower values give more weight to top-ranked items.
               Must be a positive integer.
        
        Raises:
            ValueError: If k is not a positive integer
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"RRF constant k must be a positive integer, got {k}")
        
        self.settings = settings
        self.k = k
        logger.debug(f"RRFFusion initialized with k={k}")
    
    def fuse(
        self, 
        dense_results: List[RetrievalResult], 
        sparse_results: List[RetrievalResult], 
        top_k: int
    ) -> List[RetrievalResult]:
        """
        Fuse dense and sparse retrieval results using RRF algorithm.
        
        The algorithm:
        1. Assigns each document a score of 1/(k + rank) for each source list
        2. Sums scores for documents appearing in multiple lists
        3. Sorts by RRF score in descending order
        4. Returns top_k results with updated scores
        
        Args:
            dense_results: Results from dense (vector) retrieval, ordered by relevance
            sparse_results: Results from sparse (BM25) retrieval, ordered by relevance
            top_k: Maximum number of results to return
        
        Returns:
            List of RetrievalResult sorted by RRF score (descending).
            Each result has its score replaced with the computed RRF score.
            Results are deterministic given the same input rankings.
        
        Example:
            >>> dense = [
            ...     RetrievalResult(chunk_id="A", score=0.9, text="...", metadata={}),
            ...     RetrievalResult(chunk_id="B", score=0.8, text="...", metadata={}),
            ... ]
            >>> sparse = [
            ...     RetrievalResult(chunk_id="B", score=1.2, text="...", metadata={}),
            ...     RetrievalResult(chunk_id="C", score=1.0, text="...", metadata={}),
            ... ]
            >>> fused = fusion.fuse(dense, sparse, top_k=3)
            >>> # Chunk B gets highest score (appears in both lists)
            >>> # Chunk A and C follow
        """
        # Handle edge cases
        if not dense_results and not sparse_results:
            logger.debug("Both result lists are empty, returning empty list")
            return []
        
        if top_k <= 0:
            logger.debug(f"top_k={top_k} is non-positive, returning empty list")
            return []
        
        # Dictionary to accumulate RRF scores by chunk_id
        # Key: chunk_id, Value: (rrf_score, RetrievalResult)
        rrf_scores: dict[str, tuple[float, RetrievalResult]] = {}
        
        # Process dense results
        for rank, result in enumerate(dense_results, start=1):
            rrf_score = 1.0 / (self.k + rank)
            chunk_id = result.chunk_id
            
            if chunk_id in rrf_scores:
                # Add to existing score
                existing_score, existing_result = rrf_scores[chunk_id]
                rrf_scores[chunk_id] = (existing_score + rrf_score, existing_result)
            else:
                # Store the result (we'll update the score later)
                rrf_scores[chunk_id] = (rrf_score, result)
        
        # Process sparse results
        for rank, result in enumerate(sparse_results, start=1):
            rrf_score = 1.0 / (self.k + rank)
            chunk_id = result.chunk_id
            
            if chunk_id in rrf_scores:
                # Add to existing score
                existing_score, existing_result = rrf_scores[chunk_id]
                rrf_scores[chunk_id] = (existing_score + rrf_score, existing_result)
            else:
                # Store the result
                rrf_scores[chunk_id] = (rrf_score, result)
        
        # Convert to list and sort by RRF score (descending)
        # Use chunk_id as tiebreaker for deterministic ordering
        scored_results = [
            (chunk_id, score, result) 
            for chunk_id, (score, result) in rrf_scores.items()
        ]
        scored_results.sort(key=lambda x: (-x[1], x[0]))
        
        # Take top_k and create new RetrievalResult objects with updated scores
        fused_results: List[RetrievalResult] = []
        for chunk_id, rrf_score, original_result in scored_results[:top_k]:
            # Create a new result with the RRF score
            fused_result = replace(original_result, score=rrf_score)
            fused_results.append(fused_result)
        
        logger.debug(
            f"RRF fusion: {len(dense_results)} dense + {len(sparse_results)} sparse "
            f"-> {len(fused_results)} fused results (top_k={top_k})"
        )
        
        return fused_results
