import subprocess
import json
import time

# 测试直接调用bing-cn-mcp服务
def test_bing_mcp_direct():
    print("测试直接调用bing-cn-mcp服务...")
    
    # 构建MCP请求
    mcp_request = {
        "id": "test-1",
        "function": "bing_search",
        "arguments": {
            "query": "北京天气",
            "count": 10
        }
    }
    
    # 启动bing-cn-mcp服务
    process = subprocess.Popen(
        ["npx.cmd", "bing-cn-mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8'
    )
    
    # 发送请求
    request_json = json.dumps(mcp_request) + '\n'
    print(f"发送请求: {request_json}")
    
    # 发送输入
    process.stdin.write(request_json)
    process.stdin.flush()
    process.stdin.close()
    
    # 读取输出
    print("\n读取服务输出:")
    start_time = time.time()
    timeout = 15
    
    while time.time() - start_time < timeout:
        # 检查进程是否仍在运行
        if process.poll() is not None:
            break
        
        # 读取标准输出
        line = process.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
        
        print(f"输出: {line.strip()}")
        
        # 检查是否包含JSON响应
        try:
            if line.strip():
                response = json.loads(line.strip())
                print(f"\n解析到JSON响应:")
                print(json.dumps(response, indent=2, ensure_ascii=False))
                break
        except json.JSONDecodeError:
            # 不是JSON，继续读取
            pass
    
    # 读取错误输出
    stderr_output = process.stderr.read()
    if stderr_output:
        print(f"\n错误输出: {stderr_output}")
    
    # 终止进程
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()

if __name__ == "__main__":
    test_bing_mcp_direct()