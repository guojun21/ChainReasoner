#!/usr/bin/env python3
"""
Detailed MCP Services Test Script
Tests each MCP server with specific functionality.
"""



import json
import subprocess
import sys
from pathlib import Path


class MCPTester:
    """Test individual MCP servers."""
    
    def __init__(self, config_path: str = "mcp_config.json"):
        self.config = self._load_config(config_path)
        self.mcp_servers = self.config.get("mcpServers", {})
    
    def _load_config(self, config_path: str):
        """Load MCP configuration."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _send_request(self, server_name: str, method: str, params: dict = None, timeout: int = 30) -> dict:
        """Send a request to MCP server."""
        server_config = self.mcp_servers.get(server_name)
        if not server_config:
            return {"error": f"Server {server_name} not found"}
        
        command = server_config.get("command")
        args = server_config.get("args", [])
        env = server_config.get("env", {})
        
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method
        }
        if params:
            request["params"] = params
        
        try:
            process_env = None
            if env:
                import os
                process_env = os.environ.copy()
                for key, value in env.items():
                    process_env[key] = value
            
            process = subprocess.Popen(
                [command] + args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
                env=process_env
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
            
            stdout, stderr = process.communicate(input=init_request + "\n", timeout=timeout)
            
            if stdout:
                try:
                    return json.loads(stdout.strip())
                except json.JSONDecodeError:
                    return {"error": "Invalid JSON response", "raw": stdout}
            
            if stderr:
                return {"error": stderr}
            
            return {"error": "No response"}
            
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait()
            return {"error": "Timeout"}
        except Exception as e:
            return {"error": str(e)}
    
    def test_deepwiki(self):
        """Test mcp-deepwiki service."""
        print("\n" + "="*70)
        print("Testing: mcp-deepwiki")
        print("="*70)
        print("Description: DeepWiki knowledge base search")
        
        result = self._send_request("mcp-deepwiki", "initialize")
        
        if "error" in result:
            print(f"❌ Failed: {result['error']}")
            return False
        
        print(f"✅ Server initialized successfully")
        print(f"   Server Info: {result.get('result', {}).get('serverInfo', {})}")
        print(f"   Capabilities: {list(result.get('result', {}).get('capabilities', {}).keys())}")
        
        return True
    
    def test_trends_hub(self):
        """Test trends-hub service."""
        print("\n" + "="*70)
        print("Testing: trends-hub")
        print("="*70)
        print("Description: Trend analysis and monitoring")
        
        result = self._send_request("trends-hub", "initialize")
        
        if "error" in result:
            print(f"❌ Failed: {result['error']}")
            return False
        
        print(f"✅ Server initialized successfully")
        print(f"   Server Info: {result.get('result', {}).get('serverInfo', {})}")
        print(f"   Capabilities: {list(result.get('result', {}).get('capabilities', {}).keys())}")
        
        return True
    
    def test_hotnews(self):
        """Test mcp-server-hotnews service."""
        print("\n" + "="*70)
        print("Testing: mcp-server-hotnews")
        print("="*70)
        print("Description: Hot news and trending topics")
        
        result = self._send_request("mcp-server-hotnews", "initialize")
        
        if "error" in result:
            print(f"❌ Failed: {result['error']}")
            return False
        
        print(f"✅ Server initialized successfully")
        print(f"   Server Info: {result.get('result', {}).get('serverInfo', {})}")
        print(f"   Capabilities: {list(result.get('result', {}).get('capabilities', {}).keys())}")
        
        return True
    
    def test_searxng(self):
        """Test searxng service."""
        print("\n" + "="*70)
        print("Testing: searxng")
        print("="*70)
        print("Description: Privacy-focused search engine")
        
        result = self._send_request("searxng", "initialize")
        
        if "error" in result:
            print(f"❌ Failed: {result['error']}")
            return False
        
        print(f"✅ Server initialized successfully")
        print(f"   Server Info: {result.get('result', {}).get('serverInfo', {})}")
        print(f"   Capabilities: {list(result.get('result', {}).get('capabilities', {}).keys())}")
        
        return True
    
    def test_arxiv(self):
        """Test arxiv-mcp-server service."""
        print("\n" + "="*70)
        print("Testing: arxiv-mcp-server")
        print("="*70)
        print("Description: Academic paper search from arXiv")
        
        result = self._send_request("arxiv-mcp-server", "initialize")
        
        if "error" in result:
            print(f"❌ Failed: {result['error']}")
            return False
        
        print(f"✅ Server initialized successfully")
        print(f"   Server Info: {result.get('result', {}).get('serverInfo', {})}")
        print(f"   Capabilities: {list(result.get('result', {}).get('capabilities', {}).keys())}")
        
        return True
    
    def test_worldbank(self):
        """Test worldbank-mcp service."""
        print("\n" + "="*70)
        print("Testing: worldbank-mcp")
        print("="*70)
        print("Description: World Bank open data access")
        
        result = self._send_request("worldbank-mcp", "initialize")
        
        if "error" in result:
            print(f"❌ Failed: {result['error']}")
            return False
        
        print(f"✅ Server initialized successfully")
        print(f"   Server Info: {result.get('result', {}).get('serverInfo', {})}")
        print(f"   Capabilities: {list(result.get('result', {}).get('capabilities', {}).keys())}")
        
        return True
    
    def test_web_search(self):
        """Test web-search service."""
        print("\n" + "="*70)
        print("Testing: web-search")
        print("="*70)
        print("Description: Open web search")
        
        result = self._send_request("web-search", "initialize")
        
        if "error" in result:
            print(f"❌ Failed: {result['error']}")
            return False
        
        print(f"✅ Server initialized successfully")
        print(f"   Server Info: {result.get('result', {}).get('serverInfo', {})}")
        print(f"   Capabilities: {list(result.get('result', {}).get('capabilities', {}).keys())}")
        
        return True
    
    def test_biomcp(self):
        """Test biomcp service."""
        print("\n" + "="*70)
        print("Testing: biomcp")
        print("="*70)
        print("Description: Biological data and research")
        
        result = self._send_request("biomcp", "initialize")
        
        if "error" in result:
            print(f"❌ Failed: {result['error']}")
            return False
        
        print(f"✅ Server initialized successfully")
        print(f"   Server Info: {result.get('result', {}).get('serverInfo', {})}")
        print(f"   Capabilities: {list(result.get('result', {}).get('capabilities', {}).keys())}")
        
        return True
    
    def test_pozansky_stock(self):
        """Test pozansky-stock-server service."""
        print("\n" + "="*70)
        print("Testing: pozansky-stock-server")
        print("="*70)
        print("Description: Stock market data")
        
        result = self._send_request("pozansky-stock-server", "initialize")
        
        if "error" in result:
            print(f"❌ Failed: {result['error']}")
            return False
        
        print(f"✅ Server initialized successfully")
        print(f"   Server Info: {result.get('result', {}).get('serverInfo', {})}")
        print(f"   Capabilities: {list(result.get('result', {}).get('capabilities', {}).keys())}")
        
        return True
    
    def run_all_tests(self):
        """Run all MCP server tests."""
        print("\n" + "="*70)
        print("MCP Services Detailed Test")
        print("="*70)
        print(f"\nTotal servers to test: {len(self.mcp_servers)}")
        
        results = {}
        
        tests = [
            ("mcp-deepwiki", self.test_deepwiki),
            ("trends-hub", self.test_trends_hub),
            ("mcp-server-hotnews", self.test_hotnews),
            ("searxng", self.test_searxng),
            ("arxiv-mcp-server", self.test_arxiv),
            ("worldbank-mcp", self.test_worldbank),
            ("web-search", self.test_web_search),
            ("biomcp", self.test_biomcp),
            ("pozansky-stock-server", self.test_pozansky_stock)
        ]
        
        for server_name, test_func in tests:
            if server_name in self.mcp_servers:
                results[server_name] = test_func()
            else:
                print(f"\n⚠️  Server {server_name} not in configuration")
                results[server_name] = False
        
        self._print_summary(results)
        return results
    
    def _print_summary(self, results: dict):
        """Print test summary."""
        print("\n" + "="*70)
        print("Test Summary")
        print("="*70)
        
        passed = sum(1 for v in results.values() if v)
        failed = sum(1 for v in results.values() if not v)
        
        for server_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{server_name:30s} {status}")
        
        print(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed}")
        
        if passed == len(results):
            print("\n🎉 All MCP servers tested successfully!")
        elif passed > 0:
            print(f"\n⚠️  {passed} server(s) working, {failed} failed")
        else:
            print("\n❌ All MCP servers failed")


def main():
    """Main function."""
    tester = MCPTester()
    results = tester.run_all_tests()
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())