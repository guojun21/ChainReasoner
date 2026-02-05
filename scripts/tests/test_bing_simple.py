import json
import requests

# 简单测试brave-search服务
def test_brave_simple():
    print("=== 简单测试brave-search服务 ===")
    url = "http://localhost:5000/api/v1/mcp/call"
    headers = {
        "Authorization": "Bearer multihop_agent_token_2024",
        "Content-Type": "application/json"
    }
    payload = {
        "service": "brave-search",
        "query": "北京天气"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    test_brave_simple()