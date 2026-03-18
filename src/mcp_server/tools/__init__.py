"""MCP Tools - Available tool implementations."""

from mcp_server.tools.query_knowledge_hub import (
    query_knowledge_hub,
    TOOL_NAME as QUERY_TOOL_NAME,
    TOOL_DESCRIPTION as QUERY_TOOL_DESCRIPTION,
    TOOL_INPUT_SCHEMA as QUERY_TOOL_INPUT_SCHEMA,
)
from mcp_server.tools.list_collections import (
    list_collections,
    TOOL_NAME as LIST_COLLECTIONS_TOOL_NAME,
    TOOL_DESCRIPTION as LIST_COLLECTIONS_TOOL_DESCRIPTION,
    TOOL_INPUT_SCHEMA as LIST_COLLECTIONS_TOOL_INPUT_SCHEMA,
)
from mcp_server.tools.get_document_summary import (
    get_document_summary,
    TOOL_NAME as GET_DOCUMENT_SUMMARY_TOOL_NAME,
    TOOL_DESCRIPTION as GET_DOCUMENT_SUMMARY_TOOL_DESCRIPTION,
    TOOL_INPUT_SCHEMA as GET_DOCUMENT_SUMMARY_TOOL_INPUT_SCHEMA,
)

__all__ = [
    "query_knowledge_hub",
    "QUERY_TOOL_NAME",
    "QUERY_TOOL_DESCRIPTION",
    "QUERY_TOOL_INPUT_SCHEMA",
    "list_collections",
    "LIST_COLLECTIONS_TOOL_NAME",
    "LIST_COLLECTIONS_TOOL_DESCRIPTION",
    "LIST_COLLECTIONS_TOOL_INPUT_SCHEMA",
    "get_document_summary",
    "GET_DOCUMENT_SUMMARY_TOOL_NAME",
    "GET_DOCUMENT_SUMMARY_TOOL_DESCRIPTION",
    "GET_DOCUMENT_SUMMARY_TOOL_INPUT_SCHEMA",
]
