#!/usr/bin/env python3
"""
Test script for enhanced console interface.
"""



import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from console_interface_enhanced import MultiHopConsoleEnhanced

def test_console():
    """Test enhanced console interface."""
    print("="*70)
    print("测试增强版控制台接口")
    print("="*70)
    
    console = MultiHopConsoleEnhanced()
    
    print("\n测试1: 普通提问（不使用MCP）")
    print("-"*70)
    console.process_question("中国的首都是哪里？", use_mcp=False)
    
    print("\n\n测试2: 使用MCP服务的提问")
    print("-"*70)
    console.process_question("什么是人工智能？", use_mcp=True)
    
    print("\n\n测试完成！")
    print("="*70)

if __name__ == "__main__":
    test_console()
