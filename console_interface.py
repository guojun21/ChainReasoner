#!/usr/bin/env python3
"""
MultiHop Agent Console Interface
Provides an interactive console for question answering.
"""



import yaml
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


class MultiHopConsole:
    """Console interface for MultiHop Agent."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.base_model = self.config.get("base_model", {})
        self.history = []
        self._load_history()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_history(self):
        """Load command history."""
        history_file = Path("console_history.json")
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
            except:
                self.history = []
    
    def _save_history(self):
        """Save command history."""
        with open("console_history.json", 'w', encoding='utf-8') as f:
            json.dump(self.history[-50:], f, indent=2, ensure_ascii=False)
    
    def _call_llm(self, question: str) -> Dict[str, Any]:
        """Call LLM API with reasoning."""
        api_url = self.base_model.get("api_url")
        api_key = self.base_model.get("api_key")
        model_id = self.base_model.get("model_id")
        temperature = self.base_model.get("temperature", 0.7)
        max_tokens = self.base_model.get("max_tokens", 2048)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        system_prompt = """You are a helpful AI assistant that answers questions accurately and concisely.
Please provide your reasoning process step by step before giving the final answer.
Format your response as:
REASONING PROCESS:
[Step 1: ...]
[Step 2: ...]
...

FINAL ANSWER:
[Your final answer here]"""
        
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            reasoning_steps = []
            final_answer = ""
            
            if "REASONING PROCESS:" in content:
                parts = content.split("REASONING PROCESS:")[1].split("FINAL ANSWER:")
                reasoning_text = parts[0].strip()
                final_answer = parts[1].strip() if len(parts) > 1 else ""
                
                for line in reasoning_text.split('\n'):
                    if line.strip():
                        reasoning_steps.append(line.strip())
            else:
                final_answer = content
            
            return {
                "reasoning_steps": reasoning_steps,
                "answer": final_answer
            }
        except Exception as e:
            return {
                "reasoning_steps": [f"Error: {str(e)}"],
                "answer": f"Error: {str(e)}"
            }
    
    def process_question(self, question: str) -> None:
        """Process a single question."""
        print(f"\n{'='*70}")
        print(f"问题: {question}")
        print(f"{'='*70}")
        
        print("\n🧠 推理过程...")
        
        result = self._call_llm(question)
        
        reasoning_steps = result.get("reasoning_steps", [])
        answer = result.get("answer", "")
        
        for i, step in enumerate(reasoning_steps, 1):
            print(f"  步骤 {i}: {step}")
        
        print(f"\n✅ 最终答案:")
        print(f"  {answer}")
        
        self.history.insert(0, {
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
        self._save_history()
    
    def show_history(self, limit: int = 5):
        """Show recent history."""
        print(f"\n{'='*70}")
        print(f"最近 {limit} 条历史记录")
        print(f"{'='*70}")
        
        for i, item in enumerate(self.history[:limit], 1):
            print(f"\n[{i}] 问题: {item['question']}")
            print(f"    答案: {item['answer'][:100]}...")
    
    def show_help(self):
        """Show help information."""
        print(f"\n{'='*70}")
        print("MultiHop Agent 控制台 - 帮助")
        print(f"{'='*70}")
        print("\n命令:")
        print("  输入问题 - 直接提问")
        print("  /h 或 /help  - 显示帮助")
        print("  /history [n]  - 显示最近 n 条历史记录（默认 5 条）")
        print("  /clear         - 清空屏幕")
        print("  /quit 或 /exit - 退出程序")
        print(f"\n模型: {self.base_model.get('model_id', 'unknown')}")
        print(f"{'='*70}")
    
    def clear_screen(self):
        """Clear console screen."""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def run(self):
        """Run console interface."""
        print("\n" + "="*70)
        print("🤖 MultiHop Agent - 控制台交互端")
        print("="*70)
        print(f"\n模型: {self.base_model.get('model_id', 'unknown')}")
        print(f"API: {self.base_model.get('api_url', 'unknown')}")
        
        self.show_help()
        
        print("\n请输入您的问题（输入 /help 查看帮助）:")
        
        import sys
        
        if not sys.stdin.isatty():
            print("\n检测到管道输入，处理单次提问...")
            for line in sys.stdin:
                user_input = line.strip()
                if user_input:
                    self.process_question(user_input)
            return
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['/quit', '/exit', 'q']:
                    print("\n👋 再见！")
                    break
                
                if user_input.lower() in ['/clear', '/cls']:
                    self.clear_screen()
                    continue
                
                if user_input.lower() in ['/help', '/h']:
                    self.show_help()
                    continue
                
                if user_input.lower().startswith('/history'):
                    parts = user_input.split()
                    limit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
                    self.show_history(limit)
                    continue
                
                self.process_question(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 程序已中断。再见！")
                break
            except EOFError:
                print("\n\n👋 输入结束。再见！")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")


def main():
    """Main function."""
    console = MultiHopConsole()
    console.run()


if __name__ == "__main__":
    main()