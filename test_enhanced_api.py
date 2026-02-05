#!/usr/bin/env python3
"""
Test script for Enhanced MultiHop Agent API
Tests all endpoints with proper encoding.
"""

import requests
import json

BASE_URL = "http://127.0.0.1:5000"
API_TOKEN = "multihop_agent_token_2024"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json; charset=utf-8"
}

def test_health():
    """Test health check endpoint."""
    print("\n" + "="*70)
    print("测试1：健康检查端点")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_mcp_list():
    """Test MCP list endpoint."""
    print("\n" + "="*70)
    print("测试2：MCP服务列表端点")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/api/v1/mcp/list", headers=headers)
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"MCP服务数量: {result.get('count', 0)}")
    print(f"MCP服务列表: {list(result.get('mcp_services', {}).keys())}")

def test_answer_without_mcp():
    """Test answer endpoint without MCP."""
    print("\n" + "="*70)
    print("测试3：问答端点（不使用MCP）")
    print("="*70)
    
    data = {
        "question": "中国的首都是哪里？",
        "use_mcp": False
    }
    
    response = requests.post(f"{BASE_URL}/api/v1/answer", headers=headers, json=data)
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"问题: {result.get('question', '')}")
    print(f"答案: {result.get('answer', '')}")
    print(f"使用MCP: {result.get('use_mcp', False)}")
    print(f"推理步骤数量: {len(result.get('reasoning_steps', []))}")

def test_answer_with_mcp():
    """Test answer endpoint with MCP."""
    print("\n" + "="*70)
    print("测试4：问答端点（使用MCP）")
    print("="*70)
    
    data = {
        "question": "中国的首都是哪里？",
        "use_mcp": True
    }
    
    response = requests.post(f"{BASE_URL}/api/v1/answer", headers=headers, json=data, timeout=120)
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"问题: {result.get('question', '')}")
    print(f"答案: {result.get('answer', '')}")
    print(f"使用MCP: {result.get('use_mcp', False)}")
    print(f"MCP结果: {result.get('mcp_results', [])}")
    print(f"推理步骤数量: {len(result.get('reasoning_steps', []))}")

def test_mcp_call():
    """Test MCP call endpoint."""
    print("\n" + "="*70)
    print("测试5：MCP服务调用端点")
    print("="*70)
    
    data = {
        "service": "searxng",
        "query": "人工智能"
    }
    
    response = requests.post(f"{BASE_URL}/api/v1/mcp/call", headers=headers, json=data, timeout=60)
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"服务: {data.get('service', '')}")
    print(f"查询: {data.get('query', '')}")
    print(f"结果: {json.dumps(result, indent=2, ensure_ascii=False)}")

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("增强版MultiHop Agent API功能测试")
    print("="*70)
    
    try:
        test_health()
        test_mcp_list()
        test_answer_without_mcp()
        test_answer_with_mcp()
        test_mcp_call()
        
        print("\n" + "="*70)
        print("所有测试完成！")
        print("="*70)
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
