import requests
import json

def test_mcp_service():
    """使用API服务器测试brave-search服务"""
    url = "http://localhost:5000/api/v1/mcp/call"
    test_query = "北京天气"
    payload = {
        "service": "brave-search",
        "query": test_query
    }
    
    print(f"测试brave-search服务，查询: {test_query}")
    print(f"请求URL: {url}")
    print(f"请求数据: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(
            url,
            headers={"Authorization": "Bearer multihop_agent_token_2024"},
            json=payload,
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
        print("服务可能未启动或API端口不可用")
        return False
    except Exception as e:
        print(f"\n其他错误: {e}")
        return False

if __name__ == "__main__":
    test_mcp_service()
