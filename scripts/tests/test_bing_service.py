#!/usr/bin/env python3
"""
Test script to verify brave-search via API server
"""

import json
import requests

def test_brave_service():
    """Test brave-search service via API server"""
    print("Testing brave-search service via API server...")
    
    try:
        url = "http://localhost:5000/api/v1/mcp/call"
        headers = {
            "Authorization": "Bearer multihop_agent_token_2024",
            "Content-Type": "application/json"
        }
        payload = {
            "service": "brave-search",
            "query": "What is the capital of France?"
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"  - Status code: {response.status_code}")
        result = response.json()
        print(f"  - Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        if response.status_code == 200 and "error" not in result:
            print("  ✓ Brave search succeeded")
            return True
        print("  ✗ Brave search failed")
        return False
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing brave-search service via API server")
    print("=" * 70)
    
    success = test_brave_service()
    
    print("\n" + "=" * 70)
    print(f"Test Result: {'✓ PASS' if success else '✗ FAIL'}")
