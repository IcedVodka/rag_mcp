#!/usr/bin/env python3
"""
Query Script

Command line interface for querying the knowledge base.
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
    """Main entry point for querying."""
    parser = argparse.ArgumentParser(
        description="Query the knowledge base"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query text"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection to search in"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed results"
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip reranking"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    print(f"Query: {args.query}")
    print(f"Top-K: {args.top_k}")
    
    # TODO: Implement query logic (D7)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
