import requests
import json

def test_api_brave_search():
    """测试API服务器是否能够正常调用brave-search服务"""
    url = "http://localhost:5000/api/v1/mcp/call"
    api_token = "multihop_agent_token_2024"
    test_query = "北京天气"
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "service": "brave-search",
        "query": test_query
    }
    
    print(f"测试API服务器调用brave-search服务，查询: {test_query}")
    print(f"请求URL: {url}")
    
    try:
        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        
        if response.status_code == 200:
            result = response.json()
            if "error" not in result:
                print("API服务器调用brave-search服务成功!")
                return True
            else:
                print(f"API服务器调用brave-search服务失败: {result['error']}")
                return False
        else:
            print(f"API服务器返回错误状态码: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"连接错误: {e}")
        print("API服务器可能未启动")
        return False
    except Exception as e:
        print(f"其他错误: {e}")
        return False

if __name__ == "__main__":
    test_api_brave_search()
