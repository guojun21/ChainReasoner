#!/usr/bin/env python3
"""
Test script to verify the newly added bing-search MCP service.
"""

import json
import requests
import time

# Configuration
API_URL = "http://localhost:5000/api/v1/mcp/list"
API_TOKEN = "multihop_agent_token_2024"

def test_mcp_list():
    """Test if the bing-search MCP service is available."""
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        print("Testing MCP service list...")
        print(f"Sending request to: {API_URL}")
        
        response = requests.get(API_URL, headers=headers, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        print("\n✅ MCP service list retrieved successfully!")
        
        # Check for both possible key names
        if 'mcp_services' in result:
            services_dict = result.get('mcp_services', {})
            services = list(services_dict.keys())
            print(f"Available MCP services: {len(services)}")
            print("\nServices:")
            
            bing_search_available = False
            
            for service in services:
                print(f"  - {service}")
                if 'bing-search' in service:
                    bing_search_available = True
        elif 'services' in result:
            services = result.get('services', [])
            print(f"Available MCP services: {len(services)}")
            print("\nServices:")
            
            bing_search_available = False
            
            for service in services:
                print(f"  - {service}")
                if 'bing-search' in service:
                    bing_search_available = True
        else:
            print(f"Unexpected response structure: {result}")
            services = []
            bing_search_available = False
        
        if bing_search_available:
            print("\n✅ bing-search service is available!")
        else:
            print("\n❌ bing-search service is not available!")
        
        return bing_search_available
    except Exception as e:
        print(f"\n❌ Error testing MCP services: {e}")
        return False

def test_bing_search():
    """Test the bing-search MCP service with a sample query."""
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    data = {
        "service": "bing-search",
        "query": "测试搜索"
    }
    
    try:
        print("\nTesting bing-search service...")
        response = requests.post("http://localhost:5000/api/v1/mcp/call", headers=headers, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        print("✅ bing-search service test successful!")
        print(f"Response: {json.dumps(result, ensure_ascii=False, indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Error testing bing-search service: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("Testing bing-search MCP Service")
    print("=" * 70)
    
    # Test if bing-search is in the service list
    service_available = test_mcp_list()
    
    # Test the bing-search service if available
    if service_available:
        test_bing_search()
    
    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)
