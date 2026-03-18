#!/usr/bin/env python3
"""
Protocol Handler Unit Tests

Tests for MCP Protocol Handler including:
- JSON-RPC 2.0 protocol parsing
- initialize, tools/list, tools/call method handling
- Error handling with standard JSON-RPC error codes
- Tool registration and management
"""

import sys
import json
import pytest
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from mcp_server.protocol_handler import (
    ProtocolHandler,
    ToolSchema,
    ERROR_INVALID_REQUEST,
    ERROR_METHOD_NOT_FOUND,
    ERROR_INVALID_PARAMS,
    ERROR_INTERNAL_ERROR,
    parse_json_rpc_message,
    build_error_response,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def handler():
    """Create a fresh ProtocolHandler instance."""
    return ProtocolHandler(
        server_name="test-server",
        version="1.0.0",
        instructions="Test server instructions"
    )


def sample_tool_handler_factory():
    """Factory for creating sample async tool handlers."""
    async def handler(query: str = "", limit: int = 10):
        return f"Results for: {query} (limit: {limit})"
    return handler


@pytest.fixture
def sample_tool_handler():
    """Sample async tool handler for testing."""
    return sample_tool_handler_factory()


@pytest.fixture
def handler_with_tools(handler, sample_tool_handler):
    """Create handler with some registered tools."""
    handler.register_tool(
        name="search",
        description="Search documents",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"}
            }
        },
        handler=sample_tool_handler
    )
    handler.register_tool(
        name="summarize",
        description="Summarize text",
        inputSchema={
            "type": "object",
            "properties": {
                "text": {"type": "string"}
            }
        },
        handler=sample_tool_handler
    )
    return handler


# =============================================================================
# ProtocolHandler Initialization Tests
# =============================================================================

class TestProtocolHandlerInit:
    """Test ProtocolHandler initialization."""

    def test_default_initialization(self):
        """Test ProtocolHandler with default parameters."""
        handler = ProtocolHandler()
        
        assert handler.server_name == "smart-knowledge-hub"
        assert handler.version == "0.1.0"
        assert handler.instructions == ""
        assert handler.is_initialized() is False
        assert handler.get_registered_tools() == []

    def test_custom_initialization(self):
        """Test ProtocolHandler with custom parameters."""
        handler = ProtocolHandler(
            server_name="custom-server",
            version="2.0.0",
            instructions="Custom instructions"
        )
        
        assert handler.server_name == "custom-server"
        assert handler.version == "2.0.0"
        assert handler.instructions == "Custom instructions"


# =============================================================================
# Tool Registration Tests
# =============================================================================

@pytest.mark.asyncio
class TestToolRegistration:
    """Test tool registration functionality."""

    async def test_register_single_tool(self, handler):
        """Test registering a single tool."""
        async def test_handler():
            return "test"
        
        handler.register_tool(
            name="test-tool",
            description="A test tool",
            inputSchema={"type": "object"},
            handler=test_handler
        )
        
        tools = handler.get_registered_tools()
        assert len(tools) == 1
        assert tools[0].name == "test-tool"
        assert tools[0].description == "A test tool"

    @pytest.mark.asyncio
    async def test_register_multiple_tools(self, handler):
        """Test registering multiple tools."""
        async def handler1():
            return "result1"
        
        async def handler2():
            return "result2"
        
        handler.register_tool("tool1", "Tool 1", {}, handler1)
        handler.register_tool("tool2", "Tool 2", {}, handler2)
        
        tools = handler.get_registered_tools()
        assert len(tools) == 2
        tool_names = {t.name for t in tools}
        assert tool_names == {"tool1", "tool2"}

    @pytest.mark.asyncio
    async def test_unregister_tool(self, handler):
        """Test unregistering a tool."""
        async def test_handler():
            return "test"
        
        handler.register_tool("tool1", "Tool 1", {}, test_handler)
        assert len(handler.get_registered_tools()) == 1
        
        # Unregister existing tool
        result = handler.unregister_tool("tool1")
        assert result is True
        assert len(handler.get_registered_tools()) == 0
        
        # Unregister non-existent tool
        result = handler.unregister_tool("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_tool_overwrite(self, handler):
        """Test that registering same name overwrites previous tool."""
        async def handler1():
            return "old"
        
        async def handler2():
            return "new"
        
        handler.register_tool("tool", "Old Tool", {}, handler1)
        handler.register_tool("tool", "New Tool", {"type": "object"}, handler2)
        
        tools = handler.get_registered_tools()
        assert len(tools) == 1
        assert tools[0].description == "New Tool"


# =============================================================================
# Initialize Method Tests
# =============================================================================

@pytest.mark.asyncio
class TestHandleInitialize:
    """Test handle_initialize method."""

    async def test_initialize_success(self, handler):
        """Test successful initialize request."""
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
        
        result = await handler.handle_initialize(params)
        
        # Verify result structure
        assert "protocolVersion" in result
        assert "capabilities" in result
        assert "serverInfo" in result
        
        # Verify capabilities
        assert "tools" in result["capabilities"]
        assert result["capabilities"]["tools"]["listChanged"] is True
        
        # Verify server info
        assert result["serverInfo"]["name"] == "test-server"
        assert result["serverInfo"]["version"] == "1.0.0"
        
        # Verify initialized flag is set
        assert handler.is_initialized() is True

    @pytest.mark.asyncio
    async def test_initialize_with_empty_params(self, handler):
        """Test initialize with minimal/empty params."""
        result = await handler.handle_initialize({})
        
        assert "protocolVersion" in result
        assert "capabilities" in result
        assert "serverInfo" in result
        assert handler.is_initialized() is True


# =============================================================================
# Tools/List Method Tests
# =============================================================================

@pytest.mark.asyncio
class TestHandleToolsList:
    """Test handle_tools_list method."""

    async def test_tools_list_empty(self, handler):
        """Test tools/list with no registered tools."""
        result = await handler.handle_tools_list()
        
        assert "tools" in result
        assert result["tools"] == []

    @pytest.mark.asyncio
    async def test_tools_list_with_tools(self, handler_with_tools):
        """Test tools/list with registered tools."""
        result = await handler_with_tools.handle_tools_list()
        
        assert "tools" in result
        assert len(result["tools"]) == 2
        
        # Verify tool schema structure
        tool_names = {t["name"] for t in result["tools"]}
        assert tool_names == {"search", "summarize"}
        
        for tool in result["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert "handler" not in tool  # Handler should not be serialized


# =============================================================================
# Tools/Call Method Tests
# =============================================================================

@pytest.mark.asyncio
class TestHandleToolsCall:
    """Test handle_tools_call method."""

    async def test_call_existing_tool(self, handler_with_tools):
        """Test calling an existing tool."""
        params = {
            "name": "search",
            "arguments": {"query": "test query", "limit": 5}
        }
        
        result = await handler_with_tools.handle_tools_call(params)
        
        assert "content" in result
        assert "isError" in result
        assert result["isError"] is False
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert "test query" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_call_nonexistent_tool(self, handler):
        """Test calling a non-existent tool raises error."""
        params = {
            "name": "nonexistent",
            "arguments": {}
        }
        
        with pytest.raises(ValueError, match="Tool not found: nonexistent"):
            await handler.handle_tools_call(params)

    @pytest.mark.asyncio
    async def test_call_tool_with_no_handler(self, handler):
        """Test calling a tool with no handler raises error."""
        # Register a tool with None handler
        handler._tools["broken"] = ToolSchema(
            name="broken",
            description="Broken tool",
            inputSchema={},
            handler=None
        )
        
        params = {
            "name": "broken",
            "arguments": {}
        }
        
        with pytest.raises(ValueError, match="has no handler registered"):
            await handler.handle_tools_call(params)

    @pytest.mark.asyncio
    async def test_call_tool_invalid_params_not_dict(self, handler):
        """Test calling tool with non-dict params raises error."""
        with pytest.raises(ValueError, match="Params must be an object"):
            await handler.handle_tools_call("not a dict")

    @pytest.mark.asyncio
    async def test_call_tool_missing_name(self, handler):
        """Test calling tool without name raises error."""
        params = {"arguments": {}}
        
        with pytest.raises(ValueError, match="Tool name must be a non-empty string"):
            await handler.handle_tools_call(params)

    @pytest.mark.asyncio
    async def test_call_tool_invalid_arguments(self, handler):
        """Test calling tool with invalid arguments raises error."""
        params = {
            "name": "test",
            "arguments": "not a dict"
        }
        
        with pytest.raises(ValueError, match="Arguments must be an object"):
            await handler.handle_tools_call(params)

    @pytest.mark.asyncio
    async def test_call_tool_handler_exception(self, handler):
        """Test tool handler exception is handled gracefully."""
        async def failing_handler():
            raise RuntimeError("Something went wrong")
        
        handler.register_tool("failing", "Failing tool", {}, failing_handler)
        
        params = {
            "name": "failing",
            "arguments": {}
        }
        
        # Should not raise, but return error response
        result = await handler.handle_tools_call(params)
        
        assert result["isError"] is True
        assert len(result["content"]) == 1
        assert "Error executing tool" in result["content"][0]["text"]


# =============================================================================
# Handle Request Tests
# =============================================================================

class TestHandleRequest:
    """Test main handle_request method."""

    @pytest.mark.asyncio
    async def test_handle_initialize_request(self, handler):
        """Test handling initialize request."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"}
            }
        }
        
        response = await handler.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert "error" not in response
        assert "serverInfo" in response["result"]

    @pytest.mark.asyncio
    async def test_handle_tools_list_request(self, handler_with_tools):
        """Test handling tools/list request."""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        response = await handler_with_tools.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 2
        assert "result" in response
        assert len(response["result"]["tools"]) == 2

    @pytest.mark.asyncio
    async def test_handle_tools_call_request(self, handler_with_tools):
        """Test handling tools/call request."""
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": {"query": "test"}
            }
        }
        
        response = await handler_with_tools.handle_request(request)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 3
        assert "result" in response
        assert response["result"]["isError"] is False

    @pytest.mark.asyncio
    async def test_handle_notification_no_id_key_returns_none(self, handler):
        """Test that notifications (no 'id' key at all) don't return responses."""
        # In JSON-RPC 2.0, a Notification is a Request object without an "id" member
        # Our handle_request returns None for notifications (id=None means no id key)
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {}
        }
        
        response = await handler.handle_request(request)
        
        # Notification should return None (no response for notifications)
        assert response is None
        
    @pytest.mark.asyncio
    async def test_handle_request_with_explicit_id_returns_response(self, handler):
        """Test that requests with explicit id return responses."""
        request = {
            "jsonrpc": "2.0",
            "id": 123,
            "method": "tools/list",
            "params": {}
        }
        
        response = await handler.handle_request(request)
        
        # Request with id should return response
        assert response is not None
        assert "result" in response
        assert response["id"] == 123


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test JSON-RPC error handling."""

    @pytest.mark.asyncio
    async def test_error_invalid_json_rpc_version(self, handler):
        """Test error for invalid JSON-RPC version."""
        request = {
            "jsonrpc": "1.0",  # Wrong version
            "id": 1,
            "method": "initialize",
            "params": {}
        }
        
        response = await handler.handle_request(request)
        
        assert "error" in response
        assert response["error"]["code"] == ERROR_INVALID_REQUEST
        assert "Invalid JSON-RPC version" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_error_missing_jsonrpc_field(self, handler):
        """Test error for missing jsonrpc field."""
        request = {
            "id": 1,
            "method": "initialize",
            "params": {}
        }
        
        response = await handler.handle_request(request)
        
        assert "error" in response
        assert response["error"]["code"] == ERROR_INVALID_REQUEST

    @pytest.mark.asyncio
    async def test_error_request_not_dict(self, handler):
        """Test error for request that is not a dict."""
        response = await handler.handle_request("not a dict")
        
        assert "error" in response
        assert response["error"]["code"] == ERROR_INVALID_REQUEST

    @pytest.mark.asyncio
    async def test_error_method_not_found(self, handler):
        """Test error for unknown method (-32601)."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "unknown/method",
            "params": {}
        }
        
        response = await handler.handle_request(request)
        
        assert "error" in response
        assert response["error"]["code"] == ERROR_METHOD_NOT_FOUND
        assert "Method not found" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_error_invalid_params_value_error(self, handler):
        """Test error for invalid params (-32602)."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "",  # Empty name should trigger ValueError
                "arguments": {}
            }
        }
        
        response = await handler.handle_request(request)
        
        assert "error" in response
        assert response["error"]["code"] == ERROR_INVALID_PARAMS

    @pytest.mark.asyncio
    async def test_error_internal_error(self, handler, monkeypatch):
        """Test error for internal exceptions (-32603)."""
        # Mock handle_initialize to raise unexpected exception
        async def mock_initialize(*args):
            raise RuntimeError("Unexpected crash")
        
        monkeypatch.setattr(handler, "handle_initialize", mock_initialize)
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {}
        }
        
        response = await handler.handle_request(request)
        
        assert "error" in response
        assert response["error"]["code"] == ERROR_INTERNAL_ERROR
        assert "Internal server error" in response["error"]["message"]
        # Ensure no stack trace is leaked
        assert "Unexpected crash" not in response["error"]["message"]


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilityFunctions:
    """Test utility functions."""

    def test_parse_json_rpc_message_valid(self):
        """Test parsing valid JSON-RPC message."""
        line = '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}'
        result = parse_json_rpc_message(line)
        
        assert result is not None
        assert result["jsonrpc"] == "2.0"
        assert result["method"] == "initialize"

    def test_parse_json_rpc_message_invalid(self):
        """Test parsing invalid JSON."""
        line = "not valid json"
        result = parse_json_rpc_message(line)
        
        assert result is None

    def test_build_error_response(self):
        """Test building error response string."""
        response_str = build_error_response(
            request_id=42,
            code=ERROR_METHOD_NOT_FOUND,
            message="Method not found"
        )
        
        response = json.loads(response_str)
        
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 42
        assert response["error"]["code"] == ERROR_METHOD_NOT_FOUND
        assert response["error"]["message"] == "Method not found"


# =============================================================================
# ToolSchema Tests
# =============================================================================

class TestToolSchema:
    """Test ToolSchema dataclass."""

    def test_tool_schema_creation(self):
        """Test creating ToolSchema."""
        schema = ToolSchema(
            name="test",
            description="Test tool",
            inputSchema={"type": "object"}
        )
        
        assert schema.name == "test"
        assert schema.description == "Test tool"
        assert schema.inputSchema == {"type": "object"}

    def test_tool_schema_to_dict(self):
        """Test ToolSchema serialization."""
        schema = ToolSchema(
            name="test",
            description="Test tool",
            inputSchema={"type": "object", "properties": {}}
        )
        
        data = schema.to_dict()
        
        assert data["name"] == "test"
        assert data["description"] == "Test tool"
        assert "inputSchema" in data
        assert "handler" not in data  # Handler should not be serialized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
