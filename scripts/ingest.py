#!/usr/bin/env python3
"""
Document Ingestion Script

Offline document processing pipeline for indexing documents into the knowledge base.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def main() -> int:
    """Main entry point for document ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the knowledge base"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="default",
        help="Collection name for the documents"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to document or directory to ingest"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-ingestion even if unchanged"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    print(f"Ingesting documents from: {args.path}")
    print(f"Collection: {args.collection}")
    print(f"Force: {args.force}")
    
    # TODO: Implement ingestion pipeline (C14)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
