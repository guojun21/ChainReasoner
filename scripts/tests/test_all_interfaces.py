#!/usr/bin/env python3
"""
Comprehensive test script for all three interaction ends.
Tests console interface, API server, and web interface.
"""



import sys
import time
from datetime import datetime
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

from apps.console.console_interface import MultiHopConsoleEnhanced

def print_section(title):
    """Print a section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def test_console_interface():
    """Test enhanced console interface."""
    print_section("测试1: 控制台接口 (增强版)")
    
    console = MultiHopConsoleEnhanced()
    
    print("\n测试1.1: 普通提问（不使用MCP）")
    print("-"*70)
    console.process_question("中国的首都是哪里？", use_mcp=False)
    
    print("\n\n测试1.2: 使用MCP服务的提问")
    print("-"*70)
    console.process_question("什么是人工智能？", use_mcp=True)
    
    print("\n✅ 控制台接口测试完成")

def test_api_server():
    """Test enhanced API server."""
    print_section("测试2: API服务器 (增强版)")
    
    BASE_URL = "http://localhost:5000"
    API_TOKEN = "multihop_agent_token_2024"
    
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    print("\n测试2.1: 健康检查")
    print("-"*70)
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"状态: {result.get('status', 'unknown')}")
        print(f"模型: {result.get('model', 'unknown')}")
        print("✅ 健康检查通过")
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return
    
    print("\n测试2.2: 获取MCP服务列表")
    print("-"*70)
    try:
        response = requests.get(f"{BASE_URL}/api/v1/mcp/list", headers=headers, timeout=10)
        print(f"状态码: {response.status_code}")
        result = response.json()
        services = result.get("services", [])
        print(f"可用MCP服务数量: {len(services)}")
        for service in services[:5]:
            print(f"  - {service}")
        print("✅ MCP服务列表获取成功")
    except Exception as e:
        print(f"❌ MCP服务列表获取失败: {e}")
    
    print("\n测试2.3: 问答端点（不使用MCP）")
    print("-"*70)
    try:
        data = {
            "question": "法国的首都是哪里？",
            "use_mcp": False
        }
        response = requests.post(f"{BASE_URL}/api/v1/answer", headers=headers, json=data, timeout=120)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"问题: {result.get('question', '')}")
        print(f"答案: {result.get('answer', '')}")
        print(f"使用MCP: {result.get('use_mcp', False)}")
        print(f"推理步骤数量: {len(result.get('reasoning_steps', []))}")
        print("✅ 问答端点测试成功（不使用MCP）")
    except Exception as e:
        print(f"❌ 问答端点测试失败: {e}")
    
    print("\n测试2.4: 问答端点（使用MCP）")
    print("-"*70)
    try:
        data = {
            "question": "什么是机器学习？",
            "use_mcp": True
        }
        response = requests.post(f"{BASE_URL}/api/v1/answer", headers=headers, json=data, timeout=120)
        print(f"状态码: {response.status_code}")
        result = response.json()
        print(f"问题: {result.get('question', '')}")
        print(f"答案: {result.get('answer', '')}")
        print(f"使用MCP: {result.get('use_mcp', False)}")
        mcp_results = result.get('mcp_results', [])
        print(f"MCP结果数量: {len(mcp_results)}")
        for mcp_result in mcp_results:
            service = mcp_result.get("service", "")
            if "error" not in mcp_result:
                count = mcp_result.get("count", 0)
                print(f"  - {service}: {count} 条结果")
            else:
                error = mcp_result.get("error", "")
                print(f"  - {service}: 错误 - {error}")
        print(f"推理步骤数量: {len(result.get('reasoning_steps', []))}")
        print("✅ 问答端点测试成功（使用MCP）")
    except Exception as e:
        print(f"❌ 问答端点测试失败: {e}")
    
    print("\n✅ API服务器测试完成")

def test_web_interface():
    """Test enhanced web interface."""
    print_section("测试3: Web界面 (增强版)")
    
    BASE_URL = "http://localhost:8080"
    
    print("\n测试3.1: 访问主页")
    print("-"*70)
    try:
        response = requests.get(BASE_URL, timeout=10)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print("✅ 主页访问成功")
        else:
            print(f"❌ 主页访问失败: 状态码 {response.status_code}")
    except Exception as e:
        print(f"❌ 主页访问失败: {e}")
        return
    
    print("\n测试3.2: 访问问答端点（GET）")
    print("-"*70)
    try:
        response = requests.get(f"{BASE_URL}/ask", timeout=10)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print("✅ 问答端点访问成功")
        else:
            print(f"❌ 问答端点访问失败: 状态码 {response.status_code}")
    except Exception as e:
        print(f"❌ 问答端点访问失败: {e}")
    
    print("\n测试3.3: API问答端点（POST，不使用MCP）")
    print("-"*70)
    try:
        data = {
            "question": "英国的首都是哪里？",
            "use_mcp": False
        }
        response = requests.post(f"{BASE_URL}/api/ask", json=data, timeout=120)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"问题: {result.get('question', '')}")
            print(f"答案: {result.get('answer', '')}")
            print(f"使用MCP: {result.get('use_mcp', False)}")
            print("✅ API问答端点测试成功（不使用MCP）")
        else:
            print(f"❌ API问答端点测试失败: 状态码 {response.status_code}")
    except Exception as e:
        print(f"❌ API问答端点测试失败: {e}")
    
    print("\n测试3.4: API问答端点（POST，使用MCP）")
    print("-"*70)
    try:
        data = {
            "question": "什么是深度学习？",
            "use_mcp": True
        }
        response = requests.post(f"{BASE_URL}/api/ask", json=data, timeout=120)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"问题: {result.get('question', '')}")
            print(f"答案: {result.get('answer', '')}")
            print(f"使用MCP: {result.get('use_mcp', False)}")
            mcp_results = result.get('mcp_results', [])
            print(f"MCP结果数量: {len(mcp_results)}")
            for mcp_result in mcp_results:
                service = mcp_result.get("service", "")
                if "error" not in mcp_result:
                    count = mcp_result.get("count", 0)
                    print(f"  - {service}: {count} 条结果")
                else:
                    error = mcp_result.get("error", "")
                    print(f"  - {service}: 错误 - {error}")
            print("✅ API问答端点测试成功（使用MCP）")
        else:
            print(f"❌ API问答端点测试失败: 状态码 {response.status_code}")
    except Exception as e:
        print(f"❌ API问答端点测试失败: {e}")
    
    print("\n✅ Web界面测试完成")

def main():
    """Main function."""
    print("\n" + "="*70)
    print("  多跳推理系统 - 全方位测试")
    print(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    try:
        test_console_interface()
        test_api_server()
        test_web_interface()
        
        print_section("测试总结")
        print("\n✅ 所有测试完成！")
        print("\n三个交互端均已启用多跳推理和MCP集成：")
        print("  1. 控制台接口 (apps/console/console_interface.py)")
        print("  2. API服务器 (apps/api/api_server.py)")
        print("  3. Web界面 (apps/web/web_interface.py)")
        print("\n服务状态：")
        print("  - API服务器: http://localhost:5000")
        print("  - Web界面: http://localhost:8080")
        print("\n" + "="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n\n❌ 测试过程中发生错误: {e}")

if __name__ == "__main__":
    main()
