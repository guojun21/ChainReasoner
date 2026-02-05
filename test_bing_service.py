#!/usr/bin/env python3
"""
Test script to verify bing-cn-mcp service directly
"""

import subprocess
import json
import time

def test_bing_service():
    """Test bing-cn-mcp service directly"""
    print("Testing bing-cn-mcp service directly...")
    
    try:
        # Start the bing-cn-mcp service
        process = subprocess.Popen(
            ["npx.cmd", "bing-cn-mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        
        # Wait for service to start
        time.sleep(1)
        
        # Check initial output
        initial_stdout = process.stdout.readline()
        initial_stderr = process.stderr.readline()
        
        print(f"  - Initial stdout: {initial_stdout.strip()}")
        print(f"  - Initial stderr: {initial_stderr.strip()}")
        
        # Send test request
        test_request = {
            "id": "test-1",
            "function": "bing_search",
            "arguments": {
                "query": "What is the capital of France?",
                "count": 10
            }
        }
        
        request_json = json.dumps(test_request) + '\n'
        print(f"  - Sending request: {request_json.strip()}")
        
        # Send request to stdin
        process.stdin.write(request_json)
        process.stdin.flush()
        
        # Read response
        print("  - Waiting for response...")
        response_lines = []
        
        # Set timeout
        start_time = time.time()
        timeout = 10
        
        while time.time() - start_time < timeout:
            # Check stdout
            stdout_line = process.stdout.readline()
            if stdout_line:
                response_lines.append(f"stdout: {stdout_line.strip()}")
                print(f"  - stdout: {stdout_line.strip()}")
                
                # Try to parse JSON
                try:
                    if stdout_line.strip():
                        response = json.loads(stdout_line.strip())
                        if "result" in response:
                            print(f"  ✓ Found valid MCP response")
                            print(f"  - Result: {json.dumps(response['result'], indent=2, ensure_ascii=False)}")
                            process.terminate()
                            return True
                except json.JSONDecodeError:
                    pass
            
            # Check stderr
            stderr_line = process.stderr.readline()
            if stderr_line:
                response_lines.append(f"stderr: {stderr_line.strip()}")
                print(f"  - stderr: {stderr_line.strip()}")
            
            # Check if process has exited
            if process.poll() is not None:
                break
            
            time.sleep(0.1)
        
        # Timeout or no response
        process.terminate()
        
        print("  ✗ No valid response received from bing-cn-mcp")
        print("  - All output:")
        for line in response_lines:
            print(f"    {line}")
        
        return False
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing bing-cn-mcp service directly")
    print("=" * 70)
    
    success = test_bing_service()
    
    print("\n" + "=" * 70)
    print(f"Test Result: {'✓ PASS' if success else '✗ FAIL'}")
