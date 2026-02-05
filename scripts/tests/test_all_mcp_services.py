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
    
    print(f"\n测试MCP服务: {service_name}")
    print(f"查询: {query}")
    print(f"请求URL: {url}")
    
    try:
        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if "error" not in result:
                print(f"✅ 服务调用成功!")
                print(f"结果: {json.dumps(result, indent=2, ensure_ascii=False)[:300]}...")
                return True
            else:
                print(f"❌ 服务调用失败: {result['error']}")
                return False
        else:
            print(f"❌ 服务返回错误状态码: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ 连接错误: {e}")
        print("API服务器可能未启动")
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

def main():
    """测试所有MCP服务"""
    print("测试所有MCP服务...")
    print("=" * 70)
    
    # 所有MCP服务列表
    mcp_services = [
        "searxng",
        "web-search",
        "brave-search",
        "mcp-deepwiki",
        "trends-hub",
        "arxiv-mcp-server",
        "pozansky-stock-server",
        "worldbank-mcp",
        "mcp-server-hotnews",
        "biomcp"
    ]
    
    # 测试每个服务
    success_count = 0
    total_count = len(mcp_services)
    
    for service in mcp_services:
        if test_mcp_service(service):
            success_count += 1
        print("-" * 70)
    
    # 打印测试结果
    print("=" * 70)
    print(f"测试结果: {success_count}/{total_count} 个服务测试成功")
    print("=" * 70)

if __name__ == "__main__":
    main()
