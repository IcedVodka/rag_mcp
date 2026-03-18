#!/usr/bin/env python3
"""
Smart Knowledge Hub - Main Entry Point

A modular RAG-based knowledge retrieval system with MCP protocol support.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def setup_logging() -> logging.Logger:
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr)
        ]
    )
    return logging.getLogger("smart_knowledge_hub")


def main() -> int:
    """
    Main entry point for the Smart Knowledge Hub.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    logger = setup_logging()
    logger.info("Smart Knowledge Hub starting...")
    
    # TODO: Load settings and initialize components (A3)
    # from core.settings import load_settings
    # settings = load_settings("config/settings.yaml")
    
    logger.info("Smart Knowledge Hub initialized successfully!")
    logger.info("Use 'python -m mcp_server.server' to start MCP server")
    logger.info("Use 'python scripts/ingest.py' to ingest documents")
    logger.info("Use 'python scripts/query.py' to query knowledge base")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
