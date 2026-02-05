#!/usr/bin/env python3
"""
Simple web-search test.
"""



import json
import subprocess


def test_web_search():
    """Test web-search."""
    print("\n" + "="*70)
    print("Testing: web-search")
    print("="*70)
    
    print(f"Command: npx.cmd open-websearch@1.1.5")
    
    try:
        process = subprocess.Popen(
            ["npx.cmd", "open-websearch@1.1.5"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            text=True
        )
        
        init_request = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "MCP Test Client",
                    "version": "1.0.0"
                }
            }
        })
        
        print("\nSending initialization request...")
        stdout, stderr = process.communicate(input=init_request + "\n", timeout=30)
        
        print(f"\nStdout length: {len(stdout) if stdout else 0}")
        print(f"Stderr length: {len(stderr) if stderr else 0}")
        
        if stdout:
            print(f"\nStdout (first 1500 chars):")
            print(stdout[:1500])
        
        if stderr:
            print(f"\nStderr:")
            print(stderr)
        
        if stdout:
            try:
                lines = stdout.strip().split('\n')
                print(f"\nTotal lines: {len(lines)}")
                
                for i, line in enumerate(lines):
                    if line.strip():
                        try:
                            parsed = json.loads(line)
                            if "result" in parsed:
                                print(f"\n✅ web-search initialized successfully!")
                                print(f"Server Info: {parsed.get('result', {}).get('serverInfo', {})}")
                                return True
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                print(f"\n⚠ Error parsing output: {e}")
        
        process.terminate()
        return False
        
    except subprocess.TimeoutExpired:
        print(f"\n⚠ Timeout after 30 seconds")
        process.terminate()
        process.wait()
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = test_web_search()
    print(f"\nResult: {'PASS' if result else 'FAIL'}")