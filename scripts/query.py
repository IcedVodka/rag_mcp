#!/usr/bin/env python3
"""
Query Script - Command-line entry point for online search

Online query command-line entry that calls the complete HybridSearch + Reranker
pipeline and outputs retrieval results.

Usage:
    python scripts/query.py --query "如何配置 Azure？"
    python scripts/query.py --query "问题" --top-k 5 --verbose
    python scripts/query.py --query "问题" --collection my_collection --no-rerank
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Any

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from core.settings import load_settings, Settings
from core.types import RetrievalResult
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.sparse_retriever import SparseRetriever
from core.query_engine.fusion import RRFFusion
from core.query_engine.hybrid_search import HybridSearch
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.vector_store_factory import VectorStoreFactory
from libs.reranker.reranker_factory import RerankerFactory, RerankerProvider
from libs.reranker.base_reranker import RerankCandidate
from ingestion.storage.bm25_indexer import BM25Indexer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging level based on verbosity."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.getLogger().setLevel(level)


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to specified length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length].rstrip() + "..."


def format_result(result: RetrievalResult, index: int, verbose: bool = False) -> str:
    """Format a single retrieval result for display."""
    metadata = result.metadata or {}
    source = metadata.get("source", "未知")
    page = metadata.get("page")
    
    lines = [
        f"[{index}] Score: {result.score:.4f}",
        f"Source: {source}" + (f", Page: {page}" if page else ""),
        "-" * 50,
    ]
    
    if verbose:
        lines.append(f"Chunk ID: {result.chunk_id}")
    
    lines.append(truncate_text(result.text, 300))
    
    return "\n".join(lines)


def check_data_available(vector_store: Any, bm25_indexer: BM25Indexer) -> bool:
    """Check if there is any indexed data available."""
    # Check BM25 index
    bm25_stats = bm25_indexer.get_stats()
    if bm25_stats.get("N", 0) > 0:
        return True
    
    # Try to check vector store (may vary by implementation)
    try:
        if hasattr(vector_store, 'count'):
            return vector_store.count() > 0
    except Exception:
        pass
    
    return False


def convert_to_rerank_candidates(results: List[RetrievalResult]) -> List[RerankCandidate]:
    """Convert RetrievalResult to RerankCandidate."""
    return [
        RerankCandidate(
            id=r.chunk_id,
            text=r.text,
            score=r.score,
            metadata=r.metadata or {}
        )
        for r in results
    ]


def convert_from_rerank_results(
    rerank_results: List[Any], 
    original_results: List[RetrievalResult]
) -> List[RetrievalResult]:
    """Convert RerankResult back to RetrievalResult format."""
    # Create lookup for original results
    original_lookup = {r.chunk_id: r for r in original_results}
    
    new_results = []
    for rr in rerank_results:
        # Preserve original metadata if available
        orig = original_lookup.get(rr.id)
        metadata = orig.metadata if orig else rr.metadata
        
        new_results.append(RetrievalResult(
            chunk_id=rr.id,
            text=rr.text,
            score=rr.rerank_score,
            metadata=metadata
        ))
    
    return new_results


class QueryProcessorAdapter:
    """Adapter to make QueryProcessor return dict format for HybridSearch compatibility."""
    
    def __init__(self, processor: QueryProcessor) -> None:
        self.processor = processor
    
    def process(self, query: str) -> dict[str, Any]:
        """Process query and return dict format."""
        result = self.processor.process(query)
        return {
            'query': result.query,
            'keywords': result.keywords,
            'filters': result.filters
        }


def initialize_components(settings: Settings, verbose: bool = False):
    """
    Initialize all components for the search pipeline.
    
    Args:
        settings: Application settings
        verbose: Whether to enable verbose output
        
    Returns:
        Tuple of (hybrid_search, reranker_provider, vector_store, bm25_indexer)
    """
    if verbose:
        print("Initializing components...")
    
    # 1. Embedding Client
    embedding_client = EmbeddingFactory.create(settings)
    if verbose:
        print(f"  ✓ Embedding: {embedding_client.__class__.__name__}")
    
    # 2. Vector Store
    vector_store = VectorStoreFactory.create(settings)
    if verbose:
        print(f"  ✓ Vector Store: {vector_store.__class__.__name__}")
    
    # 3. BM25 Indexer
    bm25_indexer = BM25Indexer(
        index_path=settings.bm25.index_path,
        k1=settings.bm25.k1,
        b=settings.bm25.b
    )
    if verbose:
        stats = bm25_indexer.get_stats()
        print(f"  ✓ BM25 Index: {stats.get('N', 0)} documents, {stats.get('num_terms', 0)} terms")
    
    # 4. Query Processor (with adapter for dict compatibility)
    query_processor = QueryProcessorAdapter(QueryProcessor(settings))
    if verbose:
        print(f"  ✓ Query Processor initialized")
    
    # 5. Dense Retriever
    dense_retriever = DenseRetriever(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store
    )
    if verbose:
        print(f"  ✓ Dense Retriever initialized")
    
    # 6. Sparse Retriever
    sparse_retriever = SparseRetriever(
        settings={
            "bm25_index_path": settings.bm25.index_path,
            "bm25_k1": settings.bm25.k1,
            "bm25_b": settings.bm25.b
        },
        bm25_indexer=bm25_indexer,
        vector_store=vector_store
    )
    if verbose:
        print(f"  ✓ Sparse Retriever initialized")
    
    # 7. Fusion
    rrf_k = settings.retrieval.hybrid.get("rrf_k", 60)
    fusion = RRFFusion(settings, k=rrf_k)
    if verbose:
        print(f"  ✓ RRF Fusion (k={rrf_k}) initialized")
    
    # 8. Hybrid Search
    hybrid_search = HybridSearch(
        settings=settings,
        query_processor=query_processor,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        fusion=fusion
    )
    if verbose:
        print(f"  ✓ Hybrid Search initialized")
    
    # 9. Reranker
    reranker = RerankerFactory.create(settings)
    reranker_provider = RerankerProvider(reranker, fallback_on_error=True)
    if verbose:
        print(f"  ✓ Reranker: {reranker.__class__.__name__}")
    
    if verbose:
        print("All components initialized.\n")
    
    return hybrid_search, reranker_provider, vector_store, bm25_indexer


def main() -> int:
    """Main entry point for query script."""
    parser = argparse.ArgumentParser(
        description="Query the knowledge base using hybrid search + reranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --query "如何配置 Azure？"
  %(prog)s --query "问题" --top-k 5 --verbose
  %(prog)s --query "问题" --collection my_collection --no-rerank
  %(prog)s --query "问题" --verbose  # Show intermediate results
        """
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query text to search for"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Optional collection name to filter results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to configuration file (default: config/settings.yaml)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output (show intermediate results)"
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip the reranker stage"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Load settings
    try:
        settings = load_settings(args.config)
        if args.verbose:
            print(f"Loaded configuration from: {args.config}\n")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        print(f"Error: Failed to load configuration - {e}", file=sys.stderr)
        return 1
    
    # Initialize components
    try:
        hybrid_search, reranker_provider, vector_store, bm25_indexer = initialize_components(
            settings, args.verbose
        )
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        print(f"Error: Failed to initialize components - {e}", file=sys.stderr)
        return 1
    
    # Check if data is available
    if not check_data_available(vector_store, bm25_indexer):
        print("\n⚠️  No indexed data found in the knowledge base.")
        print("   Please run the following command first to ingest documents:")
        print(f"\n   python scripts/ingest.py --path /path/to/your/documents\n")
        return 1
    
    # Build filters if collection specified
    filters = {}
    if args.collection:
        filters["collection"] = args.collection
    
    # Execute search
    print(f"Query: {args.query}")
    if args.collection:
        print(f"Collection: {args.collection}")
    print(f"Top-K: {args.top_k}")
    if args.no_rerank:
        print("Reranker: Skipped (--no-rerank)")
    print()
    
    try:
        # Step 1: Hybrid Search
        candidates = hybrid_search.search(
            query=args.query,
            top_k=args.top_k * 2 if not args.no_rerank else args.top_k,
            filters=filters if filters else None
        )
        
        if args.verbose:
            print(f"Hybrid search returned {len(candidates)} candidates\n")
        
        if not candidates:
            print("No results found for your query.")
            print("Try rephrasing your question or ingest relevant documents.")
            return 0
        
        # Step 2: Rerank (if not disabled)
        final_results = candidates
        if not args.no_rerank and len(candidates) > 1:
            rerank_candidates = convert_to_rerank_candidates(candidates)
            
            if args.verbose:
                print(f"Running reranker on {len(rerank_candidates)} candidates...")
            
            rerank_results = reranker_provider.rerank_with_fallback(
                query=args.query,
                candidates=rerank_candidates,
                trace=None
            )
            
            final_results = convert_from_rerank_results(rerank_results, candidates)
            
            if args.verbose:
                print(f"Reranking complete\n")
        
        # Limit to top_k
        final_results = final_results[:args.top_k]
        
        # Display results
        print(f"Top-{len(final_results)} Results:")
        print("=" * 50)
        
        for i, result in enumerate(final_results, 1):
            print()
            print(format_result(result, i, args.verbose))
        
        print()
        print("=" * 50)
        
        # Verbose mode: show intermediate results
        if args.verbose:
            print("\n📊 Debug Information:")
            print("-" * 50)
            
            # Show query processing
            raw_processor = QueryProcessor(settings)
            processed_query = raw_processor.process(args.query)
            print(f"Keywords extracted: {processed_query.keywords}")
            
            # Show dense results (re-run with verbose)
            print("\n🔍 Dense Retrieval (top 5):")
            dense_retriever = DenseRetriever(settings)
            dense_results = dense_retriever.retrieve(args.query, top_k=5, filters=filters)
            for i, r in enumerate(dense_results[:5], 1):
                source = r.metadata.get("source", "未知") if r.metadata else "未知"
                print(f"  [{i}] {r.score:.4f} - {source}")
            
            # Show sparse results
            print("\n📝 Sparse Retrieval (top 5):")
            if processed_query.keywords:
                bm25_indexer = BM25Indexer(index_path=settings.bm25.index_path)
                sparse_retriever = SparseRetriever(
                    settings={"bm25_index_path": settings.bm25.index_path},
                    bm25_indexer=bm25_indexer,
                    vector_store=VectorStoreFactory.create(settings)
                )
                sparse_results = sparse_retriever.retrieve(processed_query.keywords, top_k=5)
                for i, r in enumerate(sparse_results[:5], 1):
                    source = r.metadata.get("source", "未知") if r.metadata else "未知"
                    print(f"  [{i}] {r.score:.4f} - {source}")
            else:
                print("  No keywords extracted")
            
            # Show rerank comparison
            if not args.no_rerank and len(candidates) > 1:
                print("\n🔄 Rerank Comparison (top 5):")
                for i, (orig, reranked) in enumerate(zip(candidates[:5], final_results[:5]), 1):
                    orig_source = orig.metadata.get("source", "未知") if orig.metadata else "未知"
                    reranked_source = reranked.metadata.get("source", "未知") if reranked.metadata else "未知"
                    print(f"  [{i}] Before: {orig.score:.4f} ({orig_source[:30]}...)")
                    print(f"       After:  {reranked.score:.4f} ({reranked_source[:30]}...)")
            
            print()
        
        return 0
        
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=args.verbose)
        print(f"\nError: Search failed - {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
