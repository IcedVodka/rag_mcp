"""MCP Server Layer - Model Context Protocol interface."""

from .server import main, run_server, server, protocol_handler
from .protocol_handler import (
    ProtocolHandler,
    ToolSchema,
    ERROR_INVALID_REQUEST,
    ERROR_METHOD_NOT_FOUND,
    ERROR_INVALID_PARAMS,
    ERROR_INTERNAL_ERROR,
    parse_json_rpc_message,
    build_error_response,
)

__all__ = [
    "main",
    "run_server",
    "server",
    "protocol_handler",
    "ProtocolHandler",
    "ToolSchema",
    "ERROR_INVALID_REQUEST",
    "ERROR_METHOD_NOT_FOUND",
    "ERROR_INVALID_PARAMS",
    "ERROR_INTERNAL_ERROR",
    "parse_json_rpc_message",
    "build_error_response",
]
