import subprocess
import json
import time

# 简单测试bing-cn-mcp服务
def test_bing_simple():
    print("=== 简单测试bing-cn-mcp服务 ===")
    
    # 构建MCP请求
    mcp_request = {
        "id": "test-1",
        "function": "bing_search",
        "arguments": {
            "query": "北京天气",
            "count": 10
        }
    }
    
    request_json = json.dumps(mcp_request)
    print(f"请求: {request_json}")
    
    # 使用echo命令管道输入
    try:
        # 构建命令
        command = f"echo {request_json} | npx bing-cn-mcp"
        print(f"执行命令: {command}")
        
        # 运行命令
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15
        )
        
        print(f"返回码: {result.returncode}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    test_bing_simple()