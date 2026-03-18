#!/usr/bin/env python3
"""
MCP Server - Model Context Protocol Server Implementation

Implements the MCP protocol over stdio transport for integration with
Copilot, Claude Desktop, and other MCP clients.

CRITICAL CONSTRAINT: stdout is reserved for MCP protocol messages only.
All logging must go to stderr to avoid corrupting the protocol stream.
"""

import sys
import logging
import asyncio
import anyio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncIterator, Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    LoggingLevel,
)

# Import Protocol Handler
from .protocol_handler import (
    ProtocolHandler,
    ERROR_METHOD_NOT_FOUND,
    ERROR_INVALID_PARAMS,
)

# Configure logging to stderr ONLY (stdout is reserved for MCP protocol messages)
# This must be done before any other imports that might configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
    force=True,  # Force reconfiguration even if logging was already configured
)
logger = logging.getLogger("mcp_server")

# Ensure all loggers output to stderr
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
        handler.stream = sys.stderr
        logger.debug("Redirected logging handler from stdout to stderr")


@asynccontextmanager
async def app_lifespan(server: Server) -> AsyncIterator[dict[str, Any]]:
    """
    Application lifespan context manager.
    
    Handles startup and shutdown lifecycle events.
    Yields a context dictionary that can be accessed by tool handlers.
    """
    logger.info("MCP Server starting up...")
    
    # Startup: Initialize any resources needed
    try:
        # TODO: Initialize QueryEngine and other resources when E2/E3 is implemented
        context = {
            "server_name": "smart-knowledge-hub",
            "version": "0.1.0",
        }
        logger.info("MCP Server startup complete")
        yield context
    finally:
        # Shutdown: Clean up resources
        logger.info("MCP Server shutting down...")


# Create the MCP server instance
server = Server(
    name="smart-knowledge-hub",
    version="0.1.0",
    instructions="""
    Smart Knowledge Hub - A modular RAG-based knowledge retrieval system.
    
    This MCP server provides tools for querying documentation and managing
    knowledge collections using hybrid search (BM25 + Dense Embedding) with
    optional reranking.
    """.strip(),
    lifespan=app_lifespan,
)

# Create Protocol Handler instance
protocol_handler = ProtocolHandler(
    server_name="smart-knowledge-hub",
    version="0.1.0",
    instructions="Smart Knowledge Hub - A modular RAG-based knowledge retrieval system.",
)

# Register tools
from .tools import (
    query_knowledge_hub,
    QUERY_TOOL_NAME,
    QUERY_TOOL_DESCRIPTION,
    QUERY_TOOL_INPUT_SCHEMA,
    list_collections,
    LIST_COLLECTIONS_TOOL_NAME,
    LIST_COLLECTIONS_TOOL_DESCRIPTION,
    LIST_COLLECTIONS_TOOL_INPUT_SCHEMA,
    get_document_summary,
    GET_DOCUMENT_SUMMARY_TOOL_NAME,
    GET_DOCUMENT_SUMMARY_TOOL_DESCRIPTION,
    GET_DOCUMENT_SUMMARY_TOOL_INPUT_SCHEMA,
)

protocol_handler.register_tool(
    name=QUERY_TOOL_NAME,
    description=QUERY_TOOL_DESCRIPTION,
    inputSchema=QUERY_TOOL_INPUT_SCHEMA,
    handler=query_knowledge_hub,
)
logger.info(f"Registered tool: {QUERY_TOOL_NAME}")

protocol_handler.register_tool(
    name=LIST_COLLECTIONS_TOOL_NAME,
    description=LIST_COLLECTIONS_TOOL_DESCRIPTION,
    inputSchema=LIST_COLLECTIONS_TOOL_INPUT_SCHEMA,
    handler=list_collections,
)
logger.info(f"Registered tool: {LIST_COLLECTIONS_TOOL_NAME}")

protocol_handler.register_tool(
    name=GET_DOCUMENT_SUMMARY_TOOL_NAME,
    description=GET_DOCUMENT_SUMMARY_TOOL_DESCRIPTION,
    inputSchema=GET_DOCUMENT_SUMMARY_TOOL_INPUT_SCHEMA,
    handler=get_document_summary,
)
logger.info(f"Registered tool: {GET_DOCUMENT_SUMMARY_TOOL_NAME}")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    List available tools.
    
    Returns:
        List of Tool definitions exposed by this server.
    """
    # Get tools from ProtocolHandler
    schemas = protocol_handler.get_registered_tools()
    
    # Convert to MCP Tool objects
    tools = []
    for schema in schemas:
        tools.append(
            Tool(
                name=schema.name,
                description=schema.description,
                inputSchema=schema.inputSchema,
            )
        )
    
    logger.debug(f"Listing available tools (count: {len(tools)})")
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """
    Handle tool calls.
    
    Args:
        name: The name of the tool being called
        arguments: Tool-specific arguments
        
    Returns:
        List of content blocks representing the tool result
        
    Raises:
        ValueError: If the tool name is unknown
    """
    logger.info(f"Tool call received: {name}")
    
    # Check if tool exists
    if name not in protocol_handler._tools:
        logger.warning(f"Unknown tool called: {name}")
        raise ValueError(f"Unknown tool: {name}")
    
    # Call through ProtocolHandler for consistent error handling
    try:
        result = await protocol_handler.handle_tools_call({
            "name": name,
            "arguments": arguments
        })
        
        # Extract content from result
        content = result.get("content", [])
        
        # Convert to TextContent objects
        text_contents = []
        for item in content:
            if item.get("type") == "text":
                text_contents.append(TextContent(text=item.get("text", "")))
        
        return text_contents
        
    except ValueError as e:
        # Re-raise parameter errors
        logger.error(f"Tool call parameter error: {e}")
        raise
    except Exception as e:
        # Log internal errors but don't leak details
        logger.error(f"Tool execution error: {e}", exc_info=True)
        raise ValueError(f"Tool execution failed: {str(e)}")


async def run_server() -> None:
    """
    Run the MCP server using stdio transport.
    
    This is the main entry point that sets up the stdio transport
    and starts processing MCP protocol messages.
    """
    logger.info("Starting MCP Server with stdio transport...")
    
    # Use stdio_server context manager to handle stdin/stdout
    async with stdio_server() as (read_stream, write_stream):
        logger.info("Stdio transport initialized")
        
        # Create initialization options from server
        init_options = server.create_initialization_options()
        
        # Run the server
        await server.run(
            read_stream=read_stream,
            write_stream=write_stream,
            initialization_options=init_options,
            raise_exceptions=True,
        )


def main() -> int:
    """
    Main entry point for the MCP Server.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Run the async server
        asyncio.run(run_server())
        return 0
    except KeyboardInterrupt:
        logger.info("MCP Server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"MCP Server error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
