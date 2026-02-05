#!/usr/bin/env python3
"""
MultiHop Agent Console Interface (Enhanced)
Provides an interactive console for question answering with multi-hop reasoning and MCP integration.
"""



import yaml
import json
import requests
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from logger_config import get_logger, MultiHopLogger


class MultiHopConsoleEnhanced:
    """Enhanced console interface for MultiHop Agent with MCP integration."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = get_logger("console", "console.log")
        self.logger.info("="*70)
        self.logger.info("MultiHop Console Interface (Enhanced) - Starting")
        self.logger.info("="*70)
        
        self.config = self._load_config(config_path)
        self.base_model = self.config.get("base_model", {})
        self.mcp_config = self._load_mcp_config()
        self.history = []
        self._load_history()
        
        self.logger.info(f"Model: {self.base_model.get('model_id', 'unknown')}")
        self.logger.info(f"MCP Services: {len(self.mcp_config.get('mcpServers', {}))} available")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_mcp_config(self) -> Dict[str, Any]:
        """Load MCP configuration from JSON file."""
        mcp_config_file = Path("mcp_config.json")
        if mcp_config_file.exists():
            with open(mcp_config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_history(self):
        """Load command history."""
        history_file = Path("console_history_enhanced.json")
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
            except:
                self.history = []
    
    def _save_history(self):
        """Save command history."""
        with open("console_history_enhanced.json", 'w', encoding='utf-8') as f:
            json.dump(self.history[-50:], f, indent=2, ensure_ascii=False)
    
    def _call_llm(self, question: str, context: str = "") -> Dict[str, Any]:
        """Call LLM API with reasoning."""
        start_time = time.time()
        
        self.logger.info("LLM API Call - Starting")
        self.logger.debug(f"Question: {question[:100]}...")
        
        api_url = self.base_model.get("api_url")
        api_key = self.base_model.get("api_key")
        model_id = self.base_model.get("model_id")
        temperature = self.base_model.get("temperature", 0.7)
        max_tokens = self.base_model.get("max_tokens", 2048)
        
        self.logger.debug(f"Model: {model_id}")
        self.logger.debug(f"API URL: {api_url}")
        
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
        
        user_content = question
        if context:
            user_content = f"Context information:\n{context}\n\nQuestion: {question}"
            self.logger.debug(f"Context length: {len(context)} characters")
        
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
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
            
            duration = time.time() - start_time
            self.logger.info(f"LLM API Call - Success (Duration: {duration:.2f}s)")
            self.logger.debug(f"Reasoning steps: {len(reasoning_steps)}")
            self.logger.debug(f"Answer length: {len(final_answer)} characters")
            
            return {
                "reasoning_steps": reasoning_steps,
                "answer": final_answer
            }
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"LLM API Call - Failed (Duration: {duration:.2f}s)")
            MultiHopLogger.log_error(self.logger, e, "LLM API call")
            
            return {
                "reasoning_steps": [f"Error: {str(e)}"],
                "answer": f"Error: {str(e)}"
            }
    
    def _call_mcp_service(self, service_name: str, query: str) -> Dict[str, Any]:
        """Call MCP service."""
        start_time = time.time()
        
        self.logger.info(f"MCP Service Call - {service_name}")
        self.logger.debug(f"Query: {query[:100]}...")
        
        mcp_services = self.mcp_config.get("mcpServers", {})
        
        if service_name not in mcp_services:
            self.logger.error(f"MCP Service - {service_name} not found in configuration")
            return {
                "service": service_name,
                "error": f"Service {service_name} not found in configuration"
            }
        
        service_config = mcp_services[service_name]
        service_url = service_config.get("url", "")
        
        if not service_url:
            self.logger.error(f"MCP Service - No URL configured for {service_name}")
            return {
                "service": service_name,
                "error": f"No URL configured for service {service_name}"
            }
        
        self.logger.debug(f"Service URL: {service_url}")
        
        try:
            if service_name == "searxng":
                search_url = f"{service_url}/search"
                params = {
                    "q": query,
                    "format": "json",
                    "engines": "google,bing,duckduckgo"
                }
                
                headers = {
                    "Accept": "application/json"
                }
                
                time.sleep(1)
                
                response = requests.get(search_url, params=params, headers=headers, timeout=30)
                
                if response.status_code == 429:
                    duration = time.time() - start_time
                    self.logger.warning(f"MCP Service - {service_name} rate limit exceeded (Duration: {duration:.2f}s)")
                    return {
                        "service": service_name,
                        "error": "Rate limit exceeded (429)",
                        "suggestion": "Please wait a moment before making another request"
                    }
                
                response.raise_for_status()
                results = response.json()
                
                search_results = results.get("results", [])
                formatted_results = []
                
                for result in search_results[:5]:
                    formatted_results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("content", "")
                    })
                
                duration = time.time() - start_time
                self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s, Results: {len(formatted_results)})")
                
                return {
                    "service": service_name,
                    "count": len(formatted_results),
                    "results": formatted_results
                }
            
            elif service_name == "web-search":
                search_url = f"{service_url}/api/search"
                payload = {
                    "query": query,
                    "limit": 5
                }
                
                time.sleep(1)
                
                response = requests.post(search_url, json=payload, timeout=30)
                
                if response.status_code == 429:
                    duration = time.time() - start_time
                    self.logger.warning(f"MCP Service - {service_name} rate limit exceeded (Duration: {duration:.2f}s)")
                    return {
                        "service": service_name,
                        "error": "Rate limit exceeded (429)",
                        "suggestion": "Please wait a moment before making another request"
                    }
                
                response.raise_for_status()
                results = response.json()
                
                search_results = results.get("results", [])
                formatted_results = []
                
                for result in search_results:
                    formatted_results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("snippet", "")
                    })
                
                duration = time.time() - start_time
                self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s, Results: {len(formatted_results)})")
                
                return {
                    "service": service_name,
                    "count": len(formatted_results),
                    "results": formatted_results
                }
            
            else:
                self.logger.warning(f"MCP Service - {service_name} not supported")
                return {
                    "service": service_name,
                    "error": f"Service {service_name} is not supported"
                }
        
        except requests.exceptions.Timeout:
            duration = time.time() - start_time
            self.logger.error(f"MCP Service - {service_name} timeout (Duration: {duration:.2f}s)")
            return {
                "service": service_name,
                "error": "Request timeout",
                "suggestion": "The service took too long to respond"
            }
        except requests.exceptions.ConnectionError:
            duration = time.time() - start_time
            self.logger.error(f"MCP Service - {service_name} connection error (Duration: {duration:.2f}s)")
            return {
                "service": service_name,
                "error": "Connection error",
                "suggestion": "Could not connect to the service"
            }
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"MCP Service - {service_name} exception (Duration: {duration:.2f}s)")
            MultiHopLogger.log_error(self.logger, e, f"MCP service call: {service_name}")
            return {
                "service": service_name,
                "error": str(e)
            }
    
    def _multi_hop_reasoning(self, question: str, use_mcp: bool = False) -> Dict[str, Any]:
        """Perform multi-hop reasoning with optional MCP integration."""
        start_time = time.time()
        
        self.logger.info("="*70)
        self.logger.info("Multi-Hop Reasoning - Starting")
        self.logger.info(f"Question: {question[:100]}...")
        self.logger.info(f"MCP Enabled: {use_mcp}")
        self.logger.info("="*70)
        
        reasoning_steps = []
        mcp_results = []
        context = ""
        
        if use_mcp:
            self.logger.info("Multi-Hop Step 1: Using MCP services to gather information")
            reasoning_steps.append("Step 1: 使用MCP服务收集信息")
            
            mcp_services = ["searxng", "web-search"]
            for service in mcp_services:
                self.logger.debug(f"Calling MCP service: {service}")
                mcp_result = self._call_mcp_service(service, question)
                mcp_results.append(mcp_result)
                
                if "error" not in mcp_result:
                    count = mcp_result.get("count", 0)
                    reasoning_steps.append(f"  - 调用 {service}: 获取 {count} 条结果")
                    self.logger.info(f"MCP Result: {service} returned {count} results")
                    
                    results = mcp_result.get("results", [])
                    for i, result in enumerate(results[:3], 1):
                        title = result.get("title", "")
                        snippet = result.get("snippet", "")
                        context += f"\n[{i}] {title}\n{snippet}\n"
                else:
                    error = mcp_result.get('error', 'failed')
                    reasoning_steps.append(f"  - 调用 {service}: {error}")
                    self.logger.warning(f"MCP Result: {service} failed - {error}")
        
        self.logger.info("Multi-Hop Step 2: Analyzing gathered information")
        reasoning_steps.append("Step 2: 分析收集的信息")
        
        llm_result = self._call_llm(question, context)
        reasoning_steps.extend(llm_result.get("reasoning_steps", []))
        
        self.logger.info("Multi-Hop Step 3: Synthesizing final answer")
        reasoning_steps.append("Step 3: 综合最终答案")
        
        duration = time.time() - start_time
        self.logger.info(f"Multi-Hop Reasoning - Completed (Duration: {duration:.2f}s)")
        self.logger.info(f"Total reasoning steps: {len(reasoning_steps)}")
        self.logger.info(f"MCP results: {len(mcp_results)}")
        self.logger.debug(f"Context length: {len(context)} characters")
        
        return {
            "question": question,
            "answer": llm_result.get("answer", ""),
            "reasoning_steps": reasoning_steps,
            "mcp_results": mcp_results if use_mcp else [],
            "use_mcp": use_mcp,
            "timestamp": datetime.now().isoformat()
        }
    
    def process_question(self, question: str, use_mcp: bool = False) -> None:
        """Process a single question."""
        self.logger.info("="*70)
        self.logger.info("Processing Question")
        self.logger.info(f"Question: {question}")
        self.logger.info(f"MCP: {'Enabled' if use_mcp else 'Disabled'}")
        self.logger.info("="*70)
        
        print(f"\n{'='*70}")
        print(f"问题: {question}")
        print(f"{'='*70}")
        
        print(f"\n🧠 推理过程 (MCP: {'启用' if use_mcp else '禁用'})...")
        
        result = self._multi_hop_reasoning(question, use_mcp)
        
        reasoning_steps = result.get("reasoning_steps", [])
        answer = result.get("answer", "")
        mcp_results = result.get("mcp_results", [])
        
        for i, step in enumerate(reasoning_steps, 1):
            print(f"  步骤 {i}: {step}")
        
        if mcp_results:
            print(f"\n📊 MCP 服务结果:")
            for mcp_result in mcp_results:
                service = mcp_result.get("service", "")
                if "error" not in mcp_result:
                    count = mcp_result.get("count", 0)
                    print(f"  - {service}: {count} 条结果")
                else:
                    error = mcp_result.get("error", "")
                    print(f"  - {service}: 错误 - {error}")
        
        print(f"\n✅ 最终答案:")
        print(f"  {answer}")
        
        self.logger.info(f"Final Answer: {answer[:100]}...")
        self.logger.info(f"Answer Length: {len(answer)} characters")
        
        self.history.insert(0, {
            "question": question,
            "answer": answer,
            "use_mcp": use_mcp,
            "timestamp": datetime.now().isoformat()
        })
        self._save_history()
        
        self.logger.info("Question processing completed")
        self.logger.info("="*70)
    
    def show_history(self, limit: int = 5):
        """Show recent history."""
        print(f"\n{'='*70}")
        print(f"最近 {limit} 条历史记录")
        print(f"{'='*70}")
        
        for i, item in enumerate(self.history[:limit], 1):
            mcp_status = "MCP: 启用" if item.get("use_mcp", False) else "MCP: 禁用"
            print(f"\n[{i}] 问题: {item['question']}")
            print(f"    {mcp_status}")
            print(f"    答案: {item['answer'][:100]}...")
    
    def show_help(self):
        """Show help information."""
        print(f"\n{'='*70}")
        print("MultiHop Agent 控制台 (增强版) - 帮助")
        print(f"{'='*70}")
        print("\n命令:")
        print("  输入问题 - 直接提问")
        print("  /mcp <问题> - 使用MCP服务提问")
        print("  /h 或 /help  - 显示帮助")
        print("  /history [n]  - 显示最近 n 条历史记录（默认 5 条）")
        print("  /clear         - 清空屏幕")
        print("  /quit 或 /exit - 退出程序")
        print(f"\n模型: {self.base_model.get('model_id', 'unknown')}")
        print(f"MCP 服务: {len(self.mcp_config.get('mcpServers', {}))} 个")
        print(f"{'='*70}")
    
    def clear_screen(self):
        """Clear console screen."""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def run(self):
        """Run console interface."""
        print("\n" + "="*70)
        print("🤖 MultiHop Agent - 控制台交互端 (增强版)")
        print("="*70)
        print(f"\n模型: {self.base_model.get('model_id', 'unknown')}")
        print(f"API: {self.base_model.get('api_url', 'unknown')}")
        print(f"MCP 服务: {len(self.mcp_config.get('mcpServers', {}))} 个")
        
        self.show_help()
        
        print("\n请输入您的问题（输入 /help 查看帮助）:")
        
        import sys
        
        if not sys.stdin.isatty():
            print("\n检测到管道输入，处理单次提问...")
            for line in sys.stdin:
                user_input = line.strip()
                if user_input:
                    self.process_question(user_input, use_mcp=False)
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
                
                if user_input.lower().startswith('/mcp'):
                    question = user_input[4:].strip()
                    if question:
                        self.process_question(question, use_mcp=True)
                    else:
                        print("❌ 请提供问题")
                    continue
                
                self.process_question(user_input, use_mcp=False)
                
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
    console = MultiHopConsoleEnhanced()
    console.run()


if __name__ == "__main__":
    main()
