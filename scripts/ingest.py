#!/usr/bin/env python3
"""
Document Ingestion Script

Offline document processing pipeline for indexing documents into the knowledge base.
Supports single files or directories, with automatic file type detection and
incremental updates based on file hashes.

Usage:
    python scripts/ingest.py --path /path/to/document.pdf
    python scripts/ingest.py --path /path/to/documents/ --collection my_collection
    python scripts/ingest.py --path doc.pdf --force  # Force re-ingestion
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from core.settings import load_settings, Settings
from ingestion.pipeline import IngestionPipeline, IngestionResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging level based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)


def find_documents(path: Path) -> List[Path]:
    """
    Find all supported documents in a path.
    
    If path is a file, returns list containing just that file.
    If path is a directory, recursively finds all PDF files.
    
    Args:
        path: File or directory path to search
        
    Returns:
        List of paths to supported documents
    """
    supported_extensions = {".pdf"}
    
    if path.is_file():
        if path.suffix.lower() in supported_extensions:
            return [path]
        else:
            logger.warning(f"Unsupported file type: {path.suffix}")
            return []
    
    elif path.is_dir():
        documents = []
        for ext in supported_extensions:
            documents.extend(path.rglob(f"*{ext}"))
            documents.extend(path.rglob(f"*{ext.upper()}"))
        return sorted(documents)
    
    else:
        logger.error(f"Path does not exist: {path}")
        return []


def print_result(result: IngestionResult, index: int = 0, total: int = 1) -> None:
    """Print formatted ingestion result."""
    prefix = f"[{index}/{total}] " if total > 1 else ""
    
    if result.success:
        if result.chunks_processed == 0:
            print(f"{prefix}⏭️  Skipped (unchanged): {Path(result.source_path).name}")
        else:
            print(f"{prefix}✅ Ingested: {Path(result.source_path).name}")
            print(f"   Chunks: {result.chunks_processed}", end="")
            if result.images_extracted > 0:
                print(f" | Images: {result.images_extracted}", end="")
            print(f" | Time: {result.elapsed_seconds:.2f}s")
    else:
        print(f"{prefix}❌ Failed: {Path(result.source_path).name}")
        print(f"   Error: {result.error_message}")


def main() -> int:
    """Main entry point for document ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --path document.pdf
  %(prog)s --path ./documents/ --collection my_project
  %(prog)s --path doc.pdf --force --verbose
        """
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to document or directory to ingest"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="default",
        help="Collection name for the documents (default: default)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion even if file hasn't changed"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to configuration file (default: config/settings.yaml)"
    )
    parser.add_argument(
        "--no-transforms",
        action="store_true",
        help="Disable transform stages (refiner, enricher, captioner)"
    )
    parser.add_argument(
        "--no-captioning",
        action="store_true",
        help="Disable image captioning (only applies if transforms enabled)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Load settings
    try:
        settings = load_settings(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Find documents to ingest
    input_path = Path(args.path).resolve()
    documents = find_documents(input_path)
    
    if not documents:
        logger.error(f"No supported documents found at: {input_path}")
        return 1
    
    logger.info(f"Found {len(documents)} document(s) to ingest")
    
    # Initialize pipeline
    try:
        pipeline = IngestionPipeline(
            settings,
            enable_transforms=not args.no_transforms,
            enable_image_captioning=not args.no_captioning
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return 1
    
    # Process documents
    results = []
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for i, doc_path in enumerate(documents, 1):
        print(f"\nProcessing: {doc_path.name}")
        
        # Progress callback
        def on_progress(stage: str, current: int, total: int) -> None:
            if args.verbose:
                stage_names = {
                    "integrity_check": "Checking",
                    "load": "Loading",
                    "split": "Splitting",
                    "transform": "Transforming",
                    "encode": "Encoding",
                    "store": "Storing"
                }
                name = stage_names.get(stage, stage)
                percent = int((current / total) * 100)
                print(f"  {name}... {percent}%", end="\r")
        
        try:
            result = pipeline.run(
                source_path=str(doc_path),
                collection=args.collection,
                force=args.force,
                on_progress=on_progress if args.verbose else None
            )
            results.append(result)
            
            # Print result
            print_result(result, i, len(documents))
            
            # Update counters
            if result.success:
                if result.chunks_processed == 0:
                    skip_count += 1
                else:
                    success_count += 1
            else:
                fail_count += 1
                
        except Exception as e:
            logger.error(f"Unexpected error processing {doc_path}: {e}")
            fail_count += 1
    
    # Print summary
    print("\n" + "=" * 50)
    print("Ingestion Summary:")
    print(f"  Total: {len(documents)}")
    print(f"  Successful: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Failed: {fail_count}")
    
    # Print collection stats
    try:
        stats = pipeline.get_stats()
        bm25_stats = stats.get("bm25_stats", {})
        print(f"\nCollection Stats:")
        print(f"  BM25 documents: {bm25_stats.get('N', 0)}")
        print(f"  BM25 terms: {bm25_stats.get('num_terms', 0)}")
    except Exception:
        pass
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
