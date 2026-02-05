#!/usr/bin/env python3
"""
Test script to verify API server endpoints
"""

import requests
import json

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check endpoint...")
    try:
        url = "http://127.0.0.1:5000/health"
        response = requests.get(url, timeout=10)
        
        print(f"  - Status code: {response.status_code}")
        print(f"  - Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
        if response.status_code == 200:
            print("  ✓ Health check passed")
            return True
        else:
            print("  ✗ Health check failed")
            return False
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return False

def test_mcp_list():
    """Test MCP service list endpoint"""
    print("\nTesting MCP service list endpoint...")
    try:
        url = "http://127.0.0.1:5000/api/v1/mcp/list"
        headers = {
            "Authorization": "Bearer multihop_agent_token_2024"
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        print(f"  - Status code: {response.status_code}")
        print(f"  - Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
        if response.status_code == 200:
            print("  ✓ MCP service list passed")
            return True
        else:
            print("  ✗ MCP service list failed")
            return False
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return False

def test_bing_search():
    """Test bing-search MCP service"""
    print("\nTesting bing-search MCP service...")
    try:
        url = "http://127.0.0.1:5000/api/v1/mcp/call"
        headers = {
            "Authorization": "Bearer multihop_agent_token_2024",
            "Content-Type": "application/json"
        }
        data = {
            "service": "bing-search",
            "query": "What is the capital of France?"
        }
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        print(f"  - Status code: {response.status_code}")
        
        try:
            response_data = response.json()
            print(f"  - Response: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
            
            if response.status_code == 200:
                if "error" not in response_data:
                    print(f"  ✓ Bing search passed with {response_data.get('count', 0)} results")
                    return True
                else:
                    print(f"  ✗ Bing search failed: {response_data['error']}")
                    return False
            else:
                print(f"  ✗ Bing search failed with status code: {response.status_code}")
                return False
        except json.JSONDecodeError:
            print(f"  ✗ Error: Invalid JSON response")
            print(f"  - Response content: {response.text}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return False

def test_searxng():
    """Test searxng MCP service"""
    print("\nTesting searxng MCP service...")
    try:
        url = "http://127.0.0.1:5000/api/v1/mcp/call"
        headers = {
            "Authorization": "Bearer multihop_agent_token_2024",
            "Content-Type": "application/json"
        }
        data = {
            "service": "searxng",
            "query": "What is the capital of Germany?"
        }
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        print(f"  - Status code: {response.status_code}")
        
        try:
            response_data = response.json()
            print(f"  - Response: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
            
            if response.status_code == 200:
                if "error" not in response_data:
                    print(f"  ✓ SearXNG search passed with {response_data.get('count', 0)} results")
                    return True
                else:
                    print(f"  ✗ SearXNG search failed: {response_data['error']}")
                    return False
            else:
                print(f"  ✗ SearXNG search failed with status code: {response.status_code}")
                return False
        except json.JSONDecodeError:
            print(f"  ✗ Error: Invalid JSON response")
            print(f"  - Response content: {response.text}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Enhanced MultiHop Agent API Server")
    print("=" * 70)
    
    # Test health check
    health_ok = test_health_check()
    
    # Test MCP service list
    mcp_list_ok = test_mcp_list()
    
    # Test bing-search
    bing_ok = test_bing_search()
    
    # Test searxng
    searxng_ok = test_searxng()
    
    print("\n" + "=" * 70)
    print("Test Results Summary:")
    print(f"  - Health Check: {'✓ PASS' if health_ok else '✗ FAIL'}")
    print(f"  - MCP Service List: {'✓ PASS' if mcp_list_ok else '✗ FAIL'}")
    print(f"  - Bing Search: {'✓ PASS' if bing_ok else '✗ FAIL'}")
    print(f"  - SearXNG: {'✓ PASS' if searxng_ok else '✗ FAIL'}")
    
    all_passed = health_ok and mcp_list_ok and bing_ok and searxng_ok
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
