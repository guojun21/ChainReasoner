#!/usr/bin/env python3
"""
Simple searxng test with environment variable.
"""



import json
import subprocess
import os


def test_searxng():
    """Test searxng with proper environment."""
    print("\n" + "="*70)
    print("Testing: searxng")
    print("="*70)
    
    env = os.environ.copy()
    env["SEARXNG_URL"] = "https://searx.stream"
    
    print(f"SEARXNG_URL: {env['SEARXNG_URL']}")
    print(f"Command: npx.cmd -y mcp-searxng")
    
    try:
        process = subprocess.Popen(
            ["npx.cmd", "-y", "mcp-searxng"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            text=True,
            env=env
        )
        
        init_request = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "MCP Test Client",
                    "version": "1.0.0"
                }
            }
        })
        
        print("\nSending initialization request...")
        stdout, stderr = process.communicate(input=init_request + "\n", timeout=30)
        
        print(f"\nStdout length: {len(stdout) if stdout else 0}")
        print(f"Stderr length: {len(stderr) if stderr else 0}")
        
        if stdout:
            print(f"\nStdout (first 1000 chars):")
            print(stdout[:1000])
        
        if stderr:
            print(f"\nStderr:")
            print(stderr)
        
        if stdout:
            try:
                response = json.loads(stdout.strip())
                if "result" in response:
                    print(f"\n✅ searxng initialized successfully!")
                    print(f"Server Info: {response.get('result', {}).get('serverInfo', {})}")
                    return True
                else:
                    print(f"\n⚠ Response has no 'result' field")
                    print(f"Response: {response}")
            except json.JSONDecodeError as e:
                print(f"\n⚠ Could not parse as JSON: {e}")
                print(f"First 500 chars: {stdout[:500]}")
        
        process.terminate()
        return False
        
    except subprocess.TimeoutExpired:
        print(f"\n⚠ Timeout after 30 seconds")
        process.terminate()
        process.wait()
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = test_searxng()
    print(f"\nResult: {'PASS' if result else 'FAIL'}")