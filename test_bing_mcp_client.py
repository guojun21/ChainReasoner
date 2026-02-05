import requests
import json

def test_mcp_service():
    """使用MCP客户端方式测试bing-cn-mcp服务"""
    # MCP服务通常使用8080端口
    url = "http://localhost:8080"
    
    # MCP服务的标准请求格式
    test_query = "北京天气"
    mcp_request = {
        "id": "test-1",
        "function": "search",
        "arguments": {
            "query": test_query
        }
    }
    
    print(f"测试bing-cn-mcp服务，查询: {test_query}")
    print(f"请求URL: {url}")
    print(f"请求数据: {json.dumps(mcp_request, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(
            url,
            json=mcp_request,
            timeout=30
        )
        
        print(f"\n响应状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n服务正常运行!")
            print(f"响应结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"\n服务返回错误状态码: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"\n连接错误: {e}")
        print("服务可能未启动或未在8080端口监听")
        return False
    except Exception as e:
        print(f"\n其他错误: {e}")
        return False

if __name__ == "__main__":
    test_mcp_service()
