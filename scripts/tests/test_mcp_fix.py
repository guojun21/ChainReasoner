import requests
import json

def test_mcp_service(service_name, query="北京天气"):
    """测试单个MCP服务是否能够正常调用"""
    url = "http://localhost:5000/api/v1/mcp/call"
    api_token = "multihop_agent_token_2024"
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "service": service_name,
        "query": query
    }
    
    print(f"测试MCP服务: {service_name}")
    
    try:
        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
    except Exception as e:
        print(f"错误: {e}")

def main():
    """测试MCP服务修复"""
    print("测试MCP服务修复...")
    print("=" * 70)
    
    # 测试几个关键服务
    test_mcp_service("brave-search")
    print("-" * 70)
    test_mcp_service("mcp-deepwiki")
    print("-" * 70)
    test_mcp_service("trends-hub")
    print("=" * 70)

if __name__ == "__main__":
    main()
