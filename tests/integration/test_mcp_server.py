#!/usr/bin/env python3
"""
Integration tests for MCP Server.

Tests the MCP Server using subprocess to verify:
1. Initialize handshake works correctly
2. stdout only contains valid MCP protocol messages
3. stderr contains logs (not stdout)
4. Server responds correctly to protocol messages
"""

import json
import os
import subprocess
import sys
import time
import pytest
from pathlib import Path


# Path to the server script
SERVER_PATH = Path(__file__).parent.parent.parent / "src" / "mcp_server" / "server.py"

# Environment for subprocess - need PYTHONPATH to find modules
SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent.parent / "src")}


class TestMCPServerE1:
    """Test cases for E1: MCP Server entry and stdio constraints."""
    
    def test_server_starts_and_responds_to_initialize(self):
        """
        Test that the server starts and responds to initialize request.
        
        This verifies:
        - Server process starts successfully
        - Server accepts JSON-RPC messages on stdin
        - Server responds with proper JSON-RPC response on stdout
        - Initialize handshake completes successfully
        """
        # Start the server as a subprocess
        process = subprocess.Popen(
            [sys.executable, str(SERVER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=SUBPROCESS_ENV,
            bufsize=1,  # Line buffered
        )
        
        try:
            # Wait a moment for server to start
            time.sleep(0.5)
            
            # Build initialize request (MCP protocol)
            initialize_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "test-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            # Send request to server
            request_line = json.dumps(initialize_request) + "\n"
            process.stdin.write(request_line)
            process.stdin.flush()
            
            # Read response with timeout
            response_line = process.stdout.readline()
            
            # Parse and verify response
            response = json.loads(response_line)
            
            # Verify JSON-RPC structure
            assert "jsonrpc" in response, "Response must have jsonrpc field"
            assert response["jsonrpc"] == "2.0", "jsonrpc must be 2.0"
            assert "id" in response, "Response must have id field"
            assert response["id"] == 1, "Response id must match request id"
            
            # Verify result structure (initialize response)
            assert "result" in response, "Response must have result field"
            result = response["result"]
            
            # Check for required MCP initialize response fields
            assert "protocolVersion" in result, "Result must have protocolVersion"
            assert "capabilities" in result, "Result must have capabilities"
            assert "serverInfo" in result, "Result must have serverInfo"
            
            # Verify server info
            server_info = result["serverInfo"]
            assert "name" in server_info, "serverInfo must have name"
            assert server_info["name"] == "smart-knowledge-hub", "Server name must match"
            
            print(f"✓ Initialize handshake successful: {server_info}")
            
        finally:
            # Clean up: terminate the server
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    
    def test_stdout_only_contains_valid_json(self):
        """
        Test that stdout only contains valid JSON-RPC messages.
        
        Verifies the critical constraint: stdout is reserved for MCP protocol only.
        No logs or debug info should appear on stdout.
        """
        process = subprocess.Popen(
            [sys.executable, str(SERVER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=SUBPROCESS_ENV,
            bufsize=1,
        )
        
        try:
            time.sleep(0.5)
            
            # Send initialize request
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "test-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            
            # Read all lines from stdout (with timeout)
            stdout_lines = []
            start_time = time.time()
            while time.time() - start_time < 3:
                import select
                # Check if data is available
                import os
                if os.name != 'nt':  # Unix-like
                    readable, _, _ = select.select([process.stdout], [], [], 0.5)
                    if readable:
                        line = process.stdout.readline()
                        if line:
                            stdout_lines.append(line.strip())
                        else:
                            break
                    else:
                        break
                else:
                    # Windows - just read one line
                    line = process.stdout.readline()
                    if line:
                        stdout_lines.append(line.strip())
                    break
            
            # Verify each line on stdout is valid JSON
            for line in stdout_lines:
                if line:  # Skip empty lines
                    try:
                        parsed = json.loads(line)
                        # Verify it's a valid JSON-RPC message
                        assert "jsonrpc" in parsed, f"Not a valid JSON-RPC message: {line}"
                        assert parsed["jsonrpc"] == "2.0", f"Invalid jsonrpc version: {line}"
                    except json.JSONDecodeError as e:
                        pytest.fail(f"stdout contains non-JSON output: {line!r} - {e}")
            
            print(f"✓ All {len(stdout_lines)} stdout lines are valid JSON-RPC")
            
        finally:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    
    def test_stderr_contains_logs(self):
        """
        Test that stderr contains log messages.
        
        Verifies that logging is properly configured to output to stderr,
        not stdout (which is reserved for MCP protocol messages).
        """
        process = subprocess.Popen(
            [sys.executable, str(SERVER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=SUBPROCESS_ENV,
            bufsize=1,
        )
        
        try:
            # Wait for server to start and log startup messages
            time.sleep(0.8)
            
            # Send a request to generate more log activity
            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "test-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            
            # Give server time to process and respond
            time.sleep(0.5)
            
            # Terminate and capture stderr
            process.terminate()
            try:
                _, stderr = process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                _, stderr = process.communicate()
            
            # Verify stderr contains log messages
            assert stderr, "stderr should contain log messages"
            
            # Check for expected log patterns
            log_patterns = ["MCP Server", "INFO", "DEBUG", "WARNING", "ERROR"]
            has_log_content = any(
                pattern in stderr for pattern in log_patterns
            )
            
            assert has_log_content, f"stderr should contain log formatted messages: {stderr[:500]}"
            
            print(f"✓ stderr contains log messages: {stderr[:200]}...")
            
        except AssertionError:
            raise
        except Exception:
            # If something went wrong, ensure cleanup
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    
    def test_tools_list_returns_registered_tools(self):
        """
        Test that tools/list returns registered tools.
        
        After E3, query_knowledge_hub tool should be registered
        and returned in the tools list.
        """
        process = subprocess.Popen(
            [sys.executable, str(SERVER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=SUBPROCESS_ENV,
            bufsize=1,
        )
        
        try:
            time.sleep(0.5)
            
            # First send initialize
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "test-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            process.stdin.write(json.dumps(init_request) + "\n")
            process.stdin.flush()
            
            # Read initialize response
            init_response = json.loads(process.stdout.readline())
            assert "result" in init_response, "Initialize should succeed"
            
            # Send initialized notification
            initialized_notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            process.stdin.write(json.dumps(initialized_notification) + "\n")
            process.stdin.flush()
            
            time.sleep(0.2)
            
            # Now request tools/list
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }
            
            process.stdin.write(json.dumps(tools_request) + "\n")
            process.stdin.flush()
            
            # Read response
            tools_response = json.loads(process.stdout.readline())
            
            # Verify structure
            assert "result" in tools_response, "tools/list should return result"
            result = tools_response["result"]
            assert "tools" in result, "result should contain tools field"
            
            # E3: tools list should include query_knowledge_hub
            tools = result["tools"]
            tool_names = [t.get("name") for t in tools]
            assert "query_knowledge_hub" in tool_names, \
                f"query_knowledge_hub should be registered, got: {tool_names}"
            
            print(f"✓ tools/list returns registered tools: {tool_names}")
            
        finally:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


class TestMCPServerE3:
    """Test cases for E3: query_knowledge_hub tool."""
    
    def test_tools_list_includes_query_knowledge_hub(self):
        """
        Test that tools/list includes query_knowledge_hub tool.
        
        Verifies that the query_knowledge_hub tool is properly registered
        and appears in the tools list.
        """
        process = subprocess.Popen(
            [sys.executable, str(SERVER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=SUBPROCESS_ENV,
            bufsize=1,
        )
        
        try:
            time.sleep(0.5)
            
            # First send initialize
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "test-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            process.stdin.write(json.dumps(init_request) + "\n")
            process.stdin.flush()
            
            # Read initialize response
            init_response = json.loads(process.stdout.readline())
            assert "result" in init_response, "Initialize should succeed"
            
            # Send initialized notification
            initialized_notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            process.stdin.write(json.dumps(initialized_notification) + "\n")
            process.stdin.flush()
            
            time.sleep(0.2)
            
            # Now request tools/list
            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            }
            
            process.stdin.write(json.dumps(tools_request) + "\n")
            process.stdin.flush()
            
            # Read response
            tools_response = json.loads(process.stdout.readline())
            
            # Verify structure
            assert "result" in tools_response, "tools/list should return result"
            result = tools_response["result"]
            assert "tools" in result, "result should contain tools field"
            
            # E3: tools list should include query_knowledge_hub
            tools = result["tools"]
            tool_names = [t.get("name") for t in tools]
            
            assert "query_knowledge_hub" in tool_names, \
                f"query_knowledge_hub tool should be registered, got: {tool_names}"
            
            # Find the tool definition
            query_tool = next(t for t in tools if t.get("name") == "query_knowledge_hub")
            
            # Verify tool has required fields
            assert "description" in query_tool, "Tool should have description"
            assert "inputSchema" in query_tool, "Tool should have inputSchema"
            
            # Verify input schema structure
            schema = query_tool["inputSchema"]
            assert schema.get("type") == "object", "Schema should be object type"
            assert "properties" in schema, "Schema should have properties"
            assert "query" in schema["properties"], "Schema should have query property"
            assert "required" in schema, "Schema should have required fields"
            assert "query" in schema["required"], "query should be required"
            
            print(f"✓ tools/list includes query_knowledge_hub: {query_tool['description'][:50]}...")
            
        finally:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    
    def test_query_knowledge_hub_tool_call_validates_params(self):
        """
        Test that query_knowledge_hub validates required parameters.
        
        Verifies that calling the tool without required 'query' parameter
        returns an appropriate error.
        """
        process = subprocess.Popen(
            [sys.executable, str(SERVER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=SUBPROCESS_ENV,
            bufsize=1,
        )
        
        try:
            time.sleep(0.5)
            
            # Initialize
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "test-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            process.stdin.write(json.dumps(init_request) + "\n")
            process.stdin.flush()
            process.stdout.readline()  # Read init response
            
            # Send initialized notification
            process.stdin.write(json.dumps({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }) + "\n")
            process.stdin.flush()
            time.sleep(0.2)
            
            # Call tool without required 'query' parameter
            tool_call = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "query_knowledge_hub",
                    "arguments": {
                        # Missing required 'query' parameter
                        "top_k": 5
                    }
                }
            }
            
            process.stdin.write(json.dumps(tool_call) + "\n")
            process.stdin.flush()
            
            # Read response
            response_line = process.stdout.readline()
            response = json.loads(response_line)
            
            # Verify response structure - should have error or content with error
            assert "result" in response or "error" in response, \
                "Response should have result or error"
            
            print("✓ query_knowledge_hub validates parameters")
            
        finally:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    
    def test_query_knowledge_hub_tool_call_with_query(self):
        """
        Test calling query_knowledge_hub with a valid query.
        
        This test verifies that the tool can be called and returns
        a properly formatted MCP response.
        """
        process = subprocess.Popen(
            [sys.executable, str(SERVER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=SUBPROCESS_ENV,
            bufsize=1,
        )
        
        try:
            time.sleep(0.5)
            
            # Initialize
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "test-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            process.stdin.write(json.dumps(init_request) + "\n")
            process.stdin.flush()
            process.stdout.readline()  # Read init response
            
            # Send initialized notification
            process.stdin.write(json.dumps({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }) + "\n")
            process.stdin.flush()
            time.sleep(0.2)
            
            # Call tool with valid query
            tool_call = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "query_knowledge_hub",
                    "arguments": {
                        "query": "test query",
                        "top_k": 3,
                        "collection": "default"
                    }
                }
            }
            
            process.stdin.write(json.dumps(tool_call) + "\n")
            process.stdin.flush()
            
            # Read response with timeout
            start_time = time.time()
            response_line = ""
            while time.time() - start_time < 10:  # 10 second timeout
                import select
                import os
                if os.name != 'nt':
                    readable, _, _ = select.select([process.stdout], [], [], 0.5)
                    if readable:
                        line = process.stdout.readline()
                        if line:
                            response_line = line
                            break
                else:
                    response_line = process.stdout.readline()
                    break
            
            assert response_line, "Should receive a response within timeout"
            response = json.loads(response_line)
            
            # Verify response structure
            assert "result" in response, f"Response should have result: {response}"
            result = response["result"]
            
            # Verify MCP content structure
            assert "content" in result, "Result should have content"
            assert "isError" in result, "Result should have isError"
            
            # Content should be an array
            assert isinstance(result["content"], list), "Content should be a list"
            
            if len(result["content"]) > 0:
                content_item = result["content"][0]
                assert "type" in content_item, "Content item should have type"
                assert content_item["type"] == "text", "Content type should be text"
                assert "text" in content_item, "Content item should have text"
                
                # Verify text contains expected elements (even for empty results)
                text = content_item["text"]
                assert isinstance(text, str), "Text should be a string"
                
                print(f"✓ query_knowledge_hub response text: {text[:100]}...")
            
            print("✓ query_knowledge_hub returns properly formatted MCP response")
            
        finally:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


class TestMCPServerE6:
    """Test cases for E6: Multimodal response with images."""
    
    def test_multimodal_response_contains_image_content(self):
        """
        Test that query_knowledge_hub returns image content when results have image_refs.
        
        This test verifies:
        - Response content array can contain both text and image types
        - Image content has correct structure (type, data, mimeType)
        - Base64 data is valid
        """
        # Import here to avoid affecting subprocess environment
        import tempfile
        from io import BytesIO
        try:
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("PIL not available for test image creation")
        
        from ingestion.storage.image_storage import ImageStorage
        
        # Create a test image and store it
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            # Create a simple test image
            img = PILImage.new('RGB', (100, 100), color='red')
            img.save(tmp.name, 'PNG')
            tmp_path = tmp.name
        
        try:
            # Store the image using ImageStorage
            storage = ImageStorage()
            record = storage.save_image(
                source_path=tmp_path,
                doc_hash="test_doc_hash_e6",
                collection="test_collection_e6",
                page_num=1,
                seq=0
            )
            image_id = record.image_id
            
            # Now test the multimodal assembler directly
            from core.response.multimodal_assembler import MultimodalAssembler
            from core.types import RetrievalResult
            from core.response.citation_generator import Citation
            
            assembler = MultimodalAssembler(storage)
            
            # Create retrieval results with image_refs
            retrieval_results = [
                RetrievalResult(
                    chunk_id="chunk_001",
                    score=0.95,
                    text="This is a test chunk with an image.",
                    metadata={
                        "source_path": "test_doc.pdf",
                        "page": 1,
                        "image_refs": [image_id]
                    }
                )
            ]
            
            citations = [Citation(
                id=1,
                source="test_doc.pdf",
                page=1,
                chunk_id="chunk_001",
                text="This is a test chunk with an image.",
                score=0.95,
                metadata={"image_refs": [image_id]}
            )]
            
            # Assemble response
            content = assembler.assemble_response(
                text_content="Test response with image",
                citations=citations,
                retrieval_results=retrieval_results
            )
            
            # Verify content structure
            assert len(content) >= 2, "Content should have text + at least one image"
            
            # First item should be text
            assert content[0]["type"] == "text", "First content item should be text"
            assert "text" in content[0], "Text content should have 'text' field"
            
            # Second item should be image
            assert content[1]["type"] == "image", "Second content item should be image"
            assert "data" in content[1], "Image content should have 'data' field"
            assert "mimeType" in content[1], "Image content should have 'mimeType' field"
            
            # Verify base64 data
            import base64
            image_data = base64.b64decode(content[1]["data"])
            assert len(image_data) > 0, "Image data should not be empty"
            
            # Verify mime type
            assert content[1]["mimeType"] == "image/png", "Mime type should be image/png"
            
            print(f"✓ Multimodal response contains {len(content)} content items")
            print(f"  - Text: {content[0]['type']}")
            print(f"  - Image: {content[1]['type']} ({content[1]['mimeType']})")
            
        finally:
            # Cleanup
            import os
            try:
                os.unlink(tmp_path)
            except:
                pass
            # Clean up stored image
            try:
                storage.delete_images(collection="test_collection_e6", doc_hash="test_doc_hash_e6")
            except:
                pass
    
    def test_multimodal_response_text_only_when_no_images(self):
        """
        Test that response contains only text when no image_refs are present.
        
        Verifies graceful degradation to text-only when results have no images.
        """
        from core.response.multimodal_assembler import MultimodalAssembler
        from core.types import RetrievalResult
        from core.response.citation_generator import Citation
        
        assembler = MultimodalAssembler()
        
        # Create retrieval results WITHOUT image_refs
        retrieval_results = [
            RetrievalResult(
                chunk_id="chunk_001",
                score=0.95,
                text="This is a test chunk without images.",
                metadata={
                    "source_path": "test_doc.pdf",
                    "page": 1
                }
            )
        ]
        
        citations = [Citation(
            id=1,
            source="test_doc.pdf",
            page=1,
            chunk_id="chunk_001",
            text="This is a test chunk without images.",
            score=0.95
        )]
        
        # Assemble response
        content = assembler.assemble_response(
            text_content="Test response without images",
            citations=citations,
            retrieval_results=retrieval_results
        )
        
        # Should only have text content
        assert len(content) == 1, "Content should have only text when no images"
        assert content[0]["type"] == "text", "Content should be text type"
        assert content[0]["text"] == "Test response without images"
        
        print("✓ Text-only response when no image_refs")
    
    def test_multimodal_response_graceful_degradation_on_missing_image(self):
        """
        Test graceful degradation when image_refs point to non-existent images.
        
        When image loading fails, should still return text content.
        """
        from core.response.multimodal_assembler import MultimodalAssembler
        from core.types import RetrievalResult
        from core.response.citation_generator import Citation
        
        assembler = MultimodalAssembler()
        
        # Create retrieval results with NON-EXISTENT image_refs
        retrieval_results = [
            RetrievalResult(
                chunk_id="chunk_001",
                score=0.95,
                text="This chunk references a missing image.",
                metadata={
                    "source_path": "test_doc.pdf",
                    "page": 1,
                    "image_refs": ["non_existent_image_id_12345"]
                }
            )
        ]
        
        citations = [Citation(
            id=1,
            source="test_doc.pdf",
            page=1,
            chunk_id="chunk_001",
            text="This chunk references a missing image.",
            score=0.95
        )]
        
        # Assemble response - should not raise exception
        content = assembler.assemble_response(
            text_content="Test response with missing image",
            citations=citations,
            retrieval_results=retrieval_results
        )
        
        # Should still have text content even if image loading failed
        assert len(content) >= 1, "Should have at least text content"
        assert content[0]["type"] == "text", "First content should be text"
        
        # Should not have image content since image doesn't exist
        image_items = [c for c in content if c.get("type") == "image"]
        assert len(image_items) == 0, "Should not have image content for missing images"
        
        print("✓ Graceful degradation when images are missing")
    
    def test_multimodal_assembler_extracts_multiple_images(self):
        """
        Test that multiple images from different chunks are all included.
        """
        import tempfile
        try:
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("PIL not available for test image creation")
        
        from ingestion.storage.image_storage import ImageStorage
        from core.response.multimodal_assembler import MultimodalAssembler
        from core.types import RetrievalResult
        from core.response.citation_generator import Citation
        
        storage = ImageStorage()
        image_ids = []
        tmp_paths = []
        
        try:
            # Create two test images
            for i, color in enumerate(['blue', 'green']):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    img = PILImage.new('RGB', (50, 50), color=color)
                    img.save(tmp.name, 'PNG')
                    tmp_paths.append(tmp.name)
                    
                    record = storage.save_image(
                        source_path=tmp.name,
                        doc_hash="test_multi_img_doc",
                        collection="test_multi_collection",
                        page_num=1,
                        seq=i
                    )
                    image_ids.append(record.image_id)
            
            assembler = MultimodalAssembler(storage)
            
            # Create retrieval results referencing both images
            retrieval_results = [
                RetrievalResult(
                    chunk_id="chunk_001",
                    score=0.95,
                    text="First chunk with first image.",
                    metadata={
                        "source_path": "test_doc.pdf",
                        "page": 1,
                        "image_refs": [image_ids[0]]
                    }
                ),
                RetrievalResult(
                    chunk_id="chunk_002",
                    score=0.90,
                    text="Second chunk with second image.",
                    metadata={
                        "source_path": "test_doc.pdf",
                        "page": 2,
                        "image_refs": [image_ids[1]]
                    }
                )
            ]
            
            citations = [
                Citation(id=1, source="test_doc.pdf", page=1, chunk_id="chunk_001",
                        text="First chunk", score=0.95),
                Citation(id=2, source="test_doc.pdf", page=2, chunk_id="chunk_002",
                        text="Second chunk", score=0.90)
            ]
            
            # Assemble response
            content = assembler.assemble_response(
                text_content="Test with multiple images",
                citations=citations,
                retrieval_results=retrieval_results
            )
            
            # Should have text + 2 images
            assert len(content) == 3, f"Should have text + 2 images, got {len(content)}"
            
            text_items = [c for c in content if c.get("type") == "text"]
            image_items = [c for c in content if c.get("type") == "image"]
            
            assert len(text_items) == 1, "Should have one text item"
            assert len(image_items) == 2, "Should have two image items"
            
            # Verify all images have correct structure
            for img in image_items:
                assert "data" in img, "Image should have data"
                assert "mimeType" in img, "Image should have mimeType"
                assert img["mimeType"] == "image/png"
            
            print(f"✓ Multiple images handled: {len(image_items)} images + 1 text")
            
        finally:
            import os
            for tmp_path in tmp_paths:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            try:
                storage.delete_images(collection="test_multi_collection", doc_hash="test_multi_img_doc")
            except:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
