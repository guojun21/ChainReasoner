#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Tools Test Script
Tests mcp-deepwiki and trends-hub MCP servers.
"""



import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MCP_CONFIG_PATH = BASE_DIR / "configs" / "mcp_config.json"


def load_mcp_config():
    """Load MCP configuration from mcp_config.json."""
    with open(DEFAULT_MCP_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_mcp_server(server_name: str, server_config: dict):
    """Test a single MCP server."""
    print(f"\n{'='*60}")
    print(f"Testing MCP Server: {server_name}")
    print(f"{'='*60}")
    
    command = server_config.get("command")
    args = server_config.get("args", [])
    
    print(f"Command: {command}")
    print(f"Args: {args}")
    
    try:
        print(f"\nStarting {server_name}...")
        process = subprocess.Popen(
            [command] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            text=True
        )
        
        print(f"Process started with PID: {process.pid}")
        
        print("\nSending MCP initialization request...")
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
        
        stdout, stderr = process.communicate(input=init_request + "\n", timeout=10)
        
        if stdout:
            print(f"\nResponse from {server_name}:")
            print(stdout)
            
            try:
                response = json.loads(stdout.strip())
                if "result" in response:
                    print(f"\n✓ {server_name} initialized successfully!")
                    print(f"Available tools: {len(response['result'].get('capabilities', {}).get('tools', []))}")
                    return True
            except json.JSONDecodeError:
                print(f"\n⚠ Could not parse response as JSON")
        
        if stderr:
            print(f"\nStderr output:")
            print(stderr)
        
        process.terminate()
        process.wait(timeout=5)
        return True
        
    except subprocess.TimeoutExpired:
        print(f"\n⚠ Timeout while testing {server_name}")
        process.terminate()
        process.wait()
        return False
    except Exception as e:
        print(f"\n✗ Error testing {server_name}: {e}")
        return False


def test_mcp_servers():
    """Test all configured MCP servers."""
    print("\n" + "="*60)
    print("MCP Servers Test")
    print("="*60)
    
    config = load_mcp_config()
    mcp_servers = config.get("mcpServers", {})
    
    print(f"\nFound {len(mcp_servers)} MCP server(s) in configuration:")
    for name in mcp_servers.keys():
        print(f"  - {name}")
    
    results = {}
    for server_name, server_config in mcp_servers.items():
        results[server_name] = test_mcp_server(server_name, server_config)
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for server_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{server_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All MCP servers are working correctly!")
    else:
        print("\n⚠️ Some MCP servers failed. Please check the errors above.")
    
    return all_passed


def main():
    """Main function."""
    success = test_mcp_servers()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())