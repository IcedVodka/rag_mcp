#!/usr/bin/env python3
"""
MCP Protocol Handler - JSON-RPC 2.0 Protocol Parser and Handler

This module provides the ProtocolHandler class that encapsulates JSON-RPC 2.0
protocol parsing, handles core MCP methods (initialize, tools/list, tools/call),
and implements standardized error handling.

Error Codes (JSON-RPC 2.0 standard):
- -32600: Invalid Request - The JSON sent is not a valid Request object
- -32601: Method not found - The method does not exist / is not available
- -32602: Invalid params - Invalid method parameter(s)
- -32603: Internal error - Internal JSON-RPC error
"""

import json
import logging
from typing import Any, Callable, Awaitable
from dataclasses import dataclass, field

# Configure logger
logger = logging.getLogger(__name__)

# JSON-RPC 2.0 Error Codes
ERROR_INVALID_REQUEST = -32600
ERROR_METHOD_NOT_FOUND = -32601
ERROR_INVALID_PARAMS = -32602
ERROR_INTERNAL_ERROR = -32603


@dataclass
class ToolSchema:
    """
    Schema definition for an MCP Tool.
    
    Attributes:
        name: Unique tool identifier
        description: Human-readable tool description
        inputSchema: JSON Schema for tool parameters
        handler: Async callable that executes the tool
    """
    name: str
    description: str
    inputSchema: dict[str, Any] = field(default_factory=dict)
    handler: Callable[..., Awaitable[Any]] = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to MCP Tool schema dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.inputSchema,
        }


class ProtocolHandler:
    """
    MCP Protocol Handler - Manages JSON-RPC 2.0 message processing.
    
    This class handles:
    1. MCP protocol initialization and capability negotiation
    2. Tool registration and discovery (tools/list)
    3. Tool execution routing (tools/call)
    4. Standardized error responses
    
    Usage:
        handler = ProtocolHandler(server_name="my-server", version="1.0.0")
        
        # Register tools
        handler.register_tool(
            name="search",
            description="Search documents",
            inputSchema={"type": "object", "properties": {...}},
            handler=search_handler
        )
        
        # Process incoming JSON-RPC requests
        response = await handler.handle_request(json_rpc_request)
    """

    def __init__(
        self,
        server_name: str = "smart-knowledge-hub",
        version: str = "0.1.0",
        instructions: str = "",
    ):
        """
        Initialize the Protocol Handler.
        
        Args:
            server_name: Name of the MCP server
            version: Server version string
            instructions: Human-readable server instructions
        """
        self.server_name = server_name
        self.version = version
        self.instructions = instructions
        self._tools: dict[str, ToolSchema] = {}
        self._initialized = False
        
        logger.debug(f"ProtocolHandler initialized for {server_name} v{version}")

    def register_tool(
        self,
        name: str,
        description: str,
        inputSchema: dict[str, Any],
        handler: Callable[..., Awaitable[Any]],
    ) -> None:
        """
        Register a tool with the protocol handler.
        
        Args:
            name: Unique tool name
            description: Tool description
            inputSchema: JSON Schema for tool parameters
            handler: Async callable to execute when tool is called
        """
        self._tools[name] = ToolSchema(
            name=name,
            description=description,
            inputSchema=inputSchema,
            handler=handler,
        )
        logger.debug(f"Registered tool: {name}")

    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            name: Tool name to unregister
            
        Returns:
            True if tool was found and removed, False otherwise
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")
            return True
        return False

    def get_registered_tools(self) -> list[ToolSchema]:
        """
        Get list of all registered tools.
        
        Returns:
            List of ToolSchema objects
        """
        return list(self._tools.values())

    async def handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """
        Handle an incoming JSON-RPC request.
        
        Args:
            request: Parsed JSON-RPC request dictionary
            
        Returns:
            JSON-RPC response dictionary, or None for notifications
        """
        # Validate JSON-RPC structure
        if not isinstance(request, dict):
            return self._error_response(None, ERROR_INVALID_REQUEST, "Request must be an object")
        
        if request.get("jsonrpc") != "2.0":
            return self._error_response(
                request.get("id"), 
                ERROR_INVALID_REQUEST, 
                "Invalid JSON-RPC version"
            )
        
        method = request.get("method")
        if not method or not isinstance(method, str):
            return self._error_response(
                request.get("id"),
                ERROR_INVALID_REQUEST,
                "Method must be a non-empty string"
            )
        
        params = request.get("params", {})
        request_id = request.get("id")
        
        # Route to appropriate handler
        try:
            if method == "initialize":
                result = await self.handle_initialize(params)
            elif method == "tools/list":
                result = await self.handle_tools_list()
            elif method == "tools/call":
                result = await self.handle_tools_call(params)
            else:
                # Method not found
                return self._error_response(
                    request_id,
                    ERROR_METHOD_NOT_FOUND,
                    f"Method not found: {method}"
                )
            
            # Notifications (no id) don't return responses
            if request_id is None:
                return None
                
            return self._success_response(request_id, result)
            
        except ValueError as e:
            # Invalid parameters
            logger.warning(f"Invalid params for method {method}: {e}")
            return self._error_response(request_id, ERROR_INVALID_PARAMS, str(e))
        except Exception as e:
            # Internal error - don't leak stack traces
            logger.error(f"Internal error handling method {method}: {e}", exc_info=True)
            return self._error_response(
                request_id,
                ERROR_INTERNAL_ERROR,
                "Internal server error"
            )

    async def handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Handle initialize request - perform capability negotiation.
        
        Args:
            params: Initialize request parameters containing:
                - protocolVersion: Client protocol version
                - capabilities: Client capabilities
                - clientInfo: Client name and version
                
        Returns:
            Initialize result with server capabilities and info
        """
        client_info = params.get("clientInfo", {})
        client_name = client_info.get("name", "unknown")
        client_version = client_info.get("version", "unknown")
        
        logger.info(f"Initializing connection with client: {client_name} v{client_version}")
        
        # Mark as initialized
        self._initialized = True
        
        # Return server capabilities
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {
                    "listChanged": True  # Server supports tools/list_changed notifications
                }
            },
            "serverInfo": {
                "name": self.server_name,
                "version": self.version,
            },
            "instructions": self.instructions if self.instructions else None,
        }

    async def handle_tools_list(self) -> dict[str, Any]:
        """
        Handle tools/list request - return registered tool schemas.
        
        Returns:
            Dictionary containing list of tool schemas
        """
        tools_list = [tool.to_dict() for tool in self._tools.values()]
        logger.debug(f"Returning tools list with {len(tools_list)} tools")
        
        return {
            "tools": tools_list
        }

    async def handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Handle tools/call request - route to specific tool handler.
        
        Args:
            params: Tool call parameters containing:
                - name: Name of the tool to call
                - arguments: Arguments to pass to the tool
                
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If params are invalid or tool not found
        """
        # Validate params structure
        if not isinstance(params, dict):
            raise ValueError("Params must be an object")
        
        tool_name = params.get("name")
        if not tool_name or not isinstance(tool_name, str):
            raise ValueError("Tool name must be a non-empty string")
        
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("Arguments must be an object")
        
        # Find the tool
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool not found: {tool_name}")
        
        if tool.handler is None:
            raise ValueError(f"Tool '{tool_name}' has no handler registered")
        
        logger.info(f"Calling tool: {tool_name}")
        
        try:
            # Call the tool handler
            result = await tool.handler(**arguments)
            
            # Format result for MCP
            return {
                "content": [
                    {
                        "type": "text",
                        "text": str(result) if result is not None else ""
                    }
                ],
                "isError": False
            }
            
        except Exception as e:
            # Tool execution error
            logger.error(f"Tool '{tool_name}' execution failed: {e}", exc_info=True)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error executing tool: {str(e)}"
                    }
                ],
                "isError": True
            }

    def _success_response(self, request_id: Any, result: Any) -> dict[str, Any]:
        """
        Build a successful JSON-RPC response.
        
        Args:
            request_id: ID from the request
            result: Result data
            
        Returns:
            JSON-RPC response dictionary
        """
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }

    def _error_response(
        self,
        request_id: Any,
        code: int,
        message: str,
        data: Any = None
    ) -> dict[str, Any]:
        """
        Build an error JSON-RPC response.
        
        Args:
            request_id: ID from the request (can be None)
            code: Error code
            message: Error message
            data: Optional additional error data
            
        Returns:
            JSON-RPC error response dictionary
        """
        error = {
            "code": code,
            "message": message,
        }
        if data is not None:
            error["data"] = data
            
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": error
        }

    def is_initialized(self) -> bool:
        """Check if the handler has been initialized."""
        return self._initialized


def parse_json_rpc_message(line: str) -> dict[str, Any] | None:
    """
    Parse a JSON-RPC message from a line of text.
    
    Args:
        line: Raw input line (should be valid JSON)
        
    Returns:
        Parsed dictionary or None if parsing failed
    """
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON-RPC message: {e}")
        return None


def build_error_response(
    request_id: Any,
    code: int,
    message: str
) -> str:
    """
    Build a JSON-RPC error response string.
    
    Args:
        request_id: Request ID (can be null for parse errors)
        code: Error code
        message: Error message
        
    Returns:
        JSON string ready to be sent
    """
    response = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message
        }
    }
    return json.dumps(response)
