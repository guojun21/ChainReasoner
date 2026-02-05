#!/usr/bin/env python3
"""
Test script to check if the base model can connect properly.
"""

import requests
import json

# Configuration
API_URL = "http://localhost:5000/api/v1/answer"
API_TOKEN = "multihop_agent_token_2024"

def test_model_connection():
    """Test the model connection by sending a simple request."""
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    data = {
        "question": "测试模型连接",
        "use_mcp": False
    }
    
    try:
        print("Testing model connection...")
        print(f"Sending request to: {API_URL}")
        
        response = requests.post(API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        print("\n✅ Connection successful!")
        print(f"Status code: {response.status_code}")
        print(f"Answer: {result.get('answer', '')}")
        print(f"Question: {result.get('question', '')}")
        print(f"Use MCP: {result.get('use_mcp', '')}")
        print(f"Reasoning steps: {result.get('reasoning_steps', 0)}")
        print(f"MCP results: {result.get('mcp_results', 0)}")
        
        return True
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Base Model Connection")
    print("=" * 70)
    
    success = test_model_connection()
    
    print("\n" + "=" * 70)
    print(f"Test {'PASSED' if success else 'FAILED'}")
    print("=" * 70)
