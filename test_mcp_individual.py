#!/usr/bin/env python3
"""
Individual MCP Server Test
Tests searxng and web-search separately.
"""



import json
import subprocess
import os


def test_searxng():
    """Test searxng MCP server."""
    print("\n" + "="*70)
    print("Testing: searxng")
    print("="*70)
    
    env = os.environ.copy()
    env["SEARXNG_URL"] = "https://searx.stream"
    
    print(f"SEARXNG_URL: {env['SEARXNG_URL']}")
    
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
        
        stdout, stderr = process.communicate(input=init_request + "\n", timeout=30)
        
        print(f"\nStdout (first 500 chars):")
        print(stdout[:500] if stdout else "None")
        
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
            except json.JSONDecodeError:
                print(f"\n⚠ Could not parse as JSON")
        
        process.terminate()
        return False
        
    except subprocess.TimeoutExpired:
        print(f"\n⚠ Timeout")
        process.terminate()
        process.wait()
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_web_search():
    """Test web-search MCP server."""
    print("\n" + "="*70)
    print("Testing: web-search")
    print("="*70)
    
    try:
        process = subprocess.Popen(
            ["npx.cmd", "open-websearch@1.1.5"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            text=True
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
        
        stdout, stderr = process.communicate(input=init_request + "\n", timeout=30)
        
        print(f"\nStdout (first 500 chars):")
        print(stdout[:500] if stdout else "None")
        
        if stderr:
            print(f"\nStderr:")
            print(stderr)
        
        if stdout:
            try:
                response = json.loads(stdout.strip())
                if "result" in response:
                    print(f"\n✅ web-search initialized successfully!")
                    print(f"Server Info: {response.get('result', {}).get('serverInfo', {})}")
                    return True
            except json.JSONDecodeError:
                print(f"\n⚠ Could not parse as JSON")
        
        process.terminate()
        return False
        
    except subprocess.TimeoutExpired:
        print(f"\n⚠ Timeout")
        process.terminate()
        process.wait()
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def main():
    """Main function."""
    print("\n" + "="*70)
    print("Individual MCP Server Tests")
    print("="*70)
    
    results = {}
    results["searxng"] = test_searxng()
    results["web-search"] = test_web_search()
    
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20s} {status}")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())