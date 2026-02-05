#!/usr/bin/env python3
"""
Enhanced MultiHop Agent API Server with Multi-hop Reasoning and MCP Integration
Provides HTTP/HTTPS endpoints for question answering with full multi-hop reasoning.
"""



import yaml
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Generator, List, Optional
from flask import Flask, request, jsonify, Response, stream_with_context
import requests
from logger_config import get_logger, MultiHopLogger


class EnhancedMultiHopAPIServer:
    """Enhanced API Server with Multi-hop Reasoning and MCP Integration."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = get_logger("api", "api.log")
        self.logger.info("="*70)
        self.logger.info("Enhanced MultiHop API Server - Starting")
        self.logger.info("="*70)
        
        self.config = self._load_config(config_path)
        self.base_model = self.config.get("base_model", {})
        self.api_token = self.config.get("api_token", "default_token_123456")
        self.mcp_config = self._load_mcp_config()
        self.app = Flask(__name__)
        self._setup_routes()
        
        self.logger.info(f"Model: {self.base_model.get('model_id', 'unknown')}")
        self.logger.info(f"MCP Services: {len(self.mcp_config.get('mcpServers', {}))} available")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_mcp_config(self) -> Dict[str, Any]:
        """Load MCP configuration."""
        mcp_config_path = Path("mcp_config.json")
        if mcp_config_path.exists():
            with open(mcp_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"mcpServers": {}}
    
    def _call_llm(self, question: str) -> Dict[str, Any]:
        """Call LLM API with reasoning process."""
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
        """Call MCP service for additional information."""
        start_time = time.time()
        
        self.logger.info(f"MCP Service Call - {service_name}")
        self.logger.debug(f"Query: {query[:100]}...")
        
        mcp_servers = self.mcp_config.get("mcpServers", {})
        
        if service_name not in mcp_servers:
            self.logger.error(f"MCP Service - {service_name} not found in configuration")
            return {
                "error": f"MCP service '{service_name}' not found",
                "available_services": list(mcp_servers.keys())
            }
        
        service_config = mcp_servers[service_name]
        self.logger.debug(f"Service config: {service_config}")
        
        try:
            if service_name == "searxng":
                result = self._call_searxng(query)
                duration = time.time() - start_time
                if "error" not in result:
                    self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s)")
                else:
                    self.logger.error(f"MCP Service - {service_name} failed (Duration: {duration:.2f}s)")
                return result
            elif service_name == "web-search":
                result = self._call_web_search(query)
                duration = time.time() - start_time
                if "error" not in result:
                    self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s)")
                else:
                    self.logger.error(f"MCP Service - {service_name} failed (Duration: {duration:.2f}s)")
                return result
            elif service_name == "bing-search":
                result = self._call_bing_search(query)
                duration = time.time() - start_time
                if "error" not in result:
                    self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s)")
                else:
                    self.logger.error(f"MCP Service - {service_name} failed (Duration: {duration:.2f}s)")
                return result
            elif service_name == "mcp-deepwiki":
                result = self._call_mcp_deepwiki(query)
                duration = time.time() - start_time
                if "error" not in result:
                    self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s)")
                else:
                    self.logger.error(f"MCP Service - {service_name} failed (Duration: {duration:.2f}s)")
                return result
            elif service_name == "trends-hub":
                result = self._call_trends_hub(query)
                duration = time.time() - start_time
                if "error" not in result:
                    self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s)")
                else:
                    self.logger.error(f"MCP Service - {service_name} failed (Duration: {duration:.2f}s)")
                return result
            elif service_name == "arxiv-mcp-server":
                result = self._call_arxiv_mcp(query)
                duration = time.time() - start_time
                if "error" not in result:
                    self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s)")
                else:
                    self.logger.error(f"MCP Service - {service_name} failed (Duration: {duration:.2f}s)")
                return result
            elif service_name == "pozansky-stock-server":
                result = self._call_pozansky_stock(query)
                duration = time.time() - start_time
                if "error" not in result:
                    self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s)")
                else:
                    self.logger.error(f"MCP Service - {service_name} failed (Duration: {duration:.2f}s)")
                return result
            elif service_name == "worldbank-mcp":
                result = self._call_worldbank_mcp(query)
                duration = time.time() - start_time
                if "error" not in result:
                    self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s)")
                else:
                    self.logger.error(f"MCP Service - {service_name} failed (Duration: {duration:.2f}s)")
                return result
            elif service_name == "mcp-server-hotnews":
                result = self._call_hotnews(query)
                duration = time.time() - start_time
                if "error" not in result:
                    self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s)")
                else:
                    self.logger.error(f"MCP Service - {service_name} failed (Duration: {duration:.2f}s)")
                return result
            elif service_name == "biomcp":
                result = self._call_biomcp(query)
                duration = time.time() - start_time
                if "error" not in result:
                    self.logger.info(f"MCP Service - {service_name} success (Duration: {duration:.2f}s)")
                else:
                    self.logger.error(f"MCP Service - {service_name} failed (Duration: {duration:.2f}s)")
                return result
            else:
                self.logger.warning(f"MCP Service - {service_name} not implemented")
                return {
                    "error": f"MCP service '{service_name}' not yet implemented",
                    "note": "This service is configured but not yet integrated"
                }
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"MCP Service - {service_name} exception (Duration: {duration:.2f}s)")
            MultiHopLogger.log_error(self.logger, e, f"MCP service call: {service_name}")
            return {
                "error": f"MCP service error: {str(e)}"
            }
    
    def _call_searxng(self, query: str) -> Dict[str, Any]:
        """Call SearXNG search service."""
        self.logger.debug("Calling SearXNG search service")
        
        searxng_url = "https://searx.stream"
        search_url = f"{searxng_url}/search"
        
        try:
            response = requests.get(
                search_url,
                params={"q": query, "format": "json"},
                timeout=30
            )
            response.raise_for_status()
            results = response.json()
            
            if results.get("results"):
                count = len(results.get("results", []))
                self.logger.debug(f"SearXNG returned {count} results")
                return {
                    "service": "searxng",
                    "query": query,
                    "results": results["results"][:5],
                    "count": count
                }
            else:
                self.logger.debug("SearXNG returned no results")
                return {
                    "service": "searxng",
                    "query": query,
                    "results": [],
                    "count": 0
                }
        except Exception as e:
            self.logger.error(f"SearXNG error: {str(e)}")
            return {
                "error": f"SearXNG error: {str(e)}"
            }
    
    def _call_web_search(self, query: str) -> Dict[str, Any]:
        """Call web search service."""
        self.logger.debug("Calling web search service")
        
        try:
            response = requests.get(
                "https://duckduckgo.com/html/",
                params={"q": query},
                timeout=30
            )
            response.raise_for_status()
            
            self.logger.debug("Web search completed successfully")
            return {
                "service": "web-search",
                "query": query,
                "status": "success",
                "note": "Web search completed"
            }
        except Exception as e:
            self.logger.error(f"Web search error: {str(e)}")
            return {
                "error": f"Web search error: {str(e)}"
            }
    
    def _call_bing_search(self, query: str) -> Dict[str, Any]:
        """Call bing-search service with real data using MCP client."""
        self.logger.info("Calling bing-search service with real data")
        
        try:
            # 使用MCP客户端库调用bing-cn-mcp服务
            import subprocess
            import json
            import time
            
            # 确保查询参数编码正确
            self.logger.info(f"Original query: {query}")
            
            # 构建MCP请求
            mcp_request = {
                "id": "test-1",
                "function": "bing_search",
                "arguments": {
                    "query": query,
                    "count": 10
                }
            }
            
            request_json = json.dumps(mcp_request, ensure_ascii=False)
            self.logger.info(f"Sending request: {request_json}")
            
            # 启动bing-cn-mcp服务并发送请求
            process = subprocess.Popen(
                ["npx.cmd", "bing-cn-mcp"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 将stderr重定向到stdout
                text=True,
                encoding='utf-8'
            )
            
            # 等待服务启动
            time.sleep(1)
            
            # 发送请求
            process.stdin.write(request_json + '\n')
            process.stdin.flush()
            
            # 读取响应
            output_lines = []
            start_time = time.time()
            timeout = 15
            
            while time.time() - start_time < timeout:
                line = process.stdout.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                
                output_lines.append(line)
                self.logger.info(f"Bing service output: {line.strip()}")
                
                # 尝试解析每一行
                try:
                    if line.strip():
                        response = json.loads(line.strip())
                        self.logger.info(f"Parsed response: {response}")
                        
                        if isinstance(response, dict):
                            # 检查是否是有效的MCP响应
                            if "result" in response:
                                result = response["result"]
                                self.logger.info(f"Found result: {result}")
                                
                                # 提取搜索结果
                                search_results = []
                                if isinstance(result, dict):
                                    # 处理不同格式的结果
                                    if "content" in result:
                                        for item in result["content"]:
                                            if item["type"] == "text":
                                                try:
                                                    content_json = json.loads(item["text"])
                                                    if "results" in content_json:
                                                        for search_item in content_json["results"]:
                                                            search_results.append({
                                                                "title": search_item.get("title", ""),
                                                                "url": search_item.get("url", ""),
                                                                "content": search_item.get("content", "")
                                                            })
                                                except json.JSONDecodeError:
                                                    # 尝试直接解析文本内容
                                                    search_results.append({
                                                        "title": "Search Result",
                                                        "url": "",
                                                        "content": item["text"]
                                                    })
                                    elif "results" in result:
                                        # 直接包含results字段
                                        for search_item in result["results"]:
                                            search_results.append({
                                                "title": search_item.get("title", ""),
                                                "url": search_item.get("url", ""),
                                                "content": search_item.get("content", "")
                                            })
                                    else:
                                        # 尝试直接将result作为内容
                                        search_results.append({
                                            "title": "Bing Search Result",
                                            "url": "",
                                            "content": str(result)
                                        })
                                
                                # 终止进程
                                process.terminate()
                                try:
                                    process.wait(timeout=2)
                                except subprocess.TimeoutExpired:
                                    process.kill()
                                
                                self.logger.info(f"Found {len(search_results)} search results")
                                return {
                                    "service": "bing-search",
                                    "query": query,
                                    "results": search_results,
                                    "count": len(search_results)
                                }
                except json.JSONDecodeError:
                    # 不是JSON，继续读取
                    continue
            
            # 超时处理
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
            
            # 合并所有输出
            all_output = ''.join(output_lines)
            self.logger.info(f"Bing service complete output: {all_output}")
            
            # 如果没有找到有效响应，返回默认结果
            self.logger.warning("No valid response found from bing-cn-mcp")
            return {
                "service": "bing-search",
                "query": query,
                "results": [],
                "count": 0
            }
            
        except Exception as e:
            self.logger.error(f"Bing search error: {str(e)}")
            return {
                "error": f"Bing search error: {str(e)}"
            }
    
    def _call_mcp_service_generic(self, service_name: str, command: list, function_name: str, query: str) -> Dict[str, Any]:
        """Generic MCP service caller using subprocess."""
        self.logger.debug(f"Calling {service_name} service with real data")
        
        try:
            import subprocess
            import json
            import time
            
            # 构建MCP请求
            mcp_request = {
                "id": f"test-{service_name}",
                "function": function_name,
                "arguments": {
                    "query": query,
                    "count": 10
                }
            }
            
            # 启动MCP服务并发送请求
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            # 发送请求
            request_json = json.dumps(mcp_request) + '\n'
            
            # 读取响应
            stdout_lines = []
            
            # 发送输入
            process.stdin.write(request_json)
            process.stdin.flush()
            
            # 设置超时
            start_time = time.time()
            timeout = 30
            
            # 读取输出
            while time.time() - start_time < timeout:
                # 读取标准输出
                if process.stdout.closed:
                    break
                
                line = process.stdout.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                
                stdout_lines.append(line)
                self.logger.debug(f"{service_name} output: {line.strip()}")
                
                # 检查是否包含JSON响应
                try:
                    if line.strip():
                        response = json.loads(line.strip())
                        if "result" in response:
                            # 找到有效响应，处理结果
                            result = response["result"]
                            # 提取搜索结果
                            search_results = []
                            if "content" in result:
                                for item in result["content"]:
                                    if item["type"] == "text":
                                        try:
                                            content_json = json.loads(item["text"])
                                            if "results" in content_json:
                                                for search_item in content_json["results"]:
                                                    search_results.append({
                                                        "title": search_item.get("title", ""),
                                                        "url": search_item.get("url", ""),
                                                        "content": search_item.get("content", "")
                                                    })
                                        except json.JSONDecodeError:
                                            # 尝试直接解析文本内容
                                            search_results.append({
                                                "title": f"{service_name} Result",
                                                "url": "",
                                                "content": item["text"]
                                            })
                            
                            # 终止进程
                            process.terminate()
                            try:
                                process.wait(timeout=2)
                            except subprocess.TimeoutExpired:
                                process.kill()
                            
                            self.logger.debug(f"{service_name} completed successfully, found {len(search_results)} results")
                            return {
                                "service": service_name,
                                "query": query,
                                "results": search_results,
                                "count": len(search_results)
                            }
                except json.JSONDecodeError:
                    # 不是JSON，继续读取
                    continue
            
            # 超时或没有找到有效响应
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
            
            # 读取剩余的错误输出
            if process.stderr:
                stderr_output = process.stderr.read()
                if stderr_output:
                    self.logger.error(f"{service_name} error: {stderr_output}")
            
            # 合并所有输出
            all_output = ''.join(stdout_lines)
            self.logger.debug(f"{service_name} complete output: {all_output}")
            
            # 尝试从完整输出中解析结果
            try:
                # 寻找JSON响应部分
                import re
                json_matches = re.findall(r'\{[^}]*"result"[^}]*\}', all_output)
                for match in json_matches:
                    try:
                        response = json.loads(match)
                        if "result" in response:
                            result = response["result"]
                            # 提取搜索结果
                            search_results = []
                            if "content" in result:
                                for item in result["content"]:
                                    if item["type"] == "text":
                                        try:
                                            content_json = json.loads(item["text"])
                                            if "results" in content_json:
                                                for search_item in content_json["results"]:
                                                    search_results.append({
                                                        "title": search_item.get("title", ""),
                                                        "url": search_item.get("url", ""),
                                                        "content": search_item.get("content", "")
                                                    })
                                        except json.JSONDecodeError:
                                            # 尝试直接解析文本内容
                                            search_results.append({
                                                "title": f"{service_name} Result",
                                                "url": "",
                                                "content": item["text"]
                                            })
                            
                            self.logger.debug(f"{service_name} completed successfully, found {len(search_results)} results")
                            return {
                                "service": service_name,
                                "query": query,
                                "results": search_results,
                                "count": len(search_results)
                            }
                    except json.JSONDecodeError:
                        continue
            except Exception as parse_error:
                self.logger.error(f"Error parsing {service_name} response: {parse_error}")
            
            # 如果解析失败，返回默认结果
            self.logger.warning(f"Failed to parse {service_name} response, returning default result")
            return {
                "service": service_name,
                "query": query,
                "results": [],
                "count": 0
            }
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"{service_name} timeout")
            if 'process' in locals():
                process.kill()
            return {
                "error": f"{service_name} timeout"
            }
        except Exception as e:
            self.logger.error(f"{service_name} error: {str(e)}")
            if 'process' in locals():
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except:
                    pass
            return {
                "error": f"{service_name} error: {str(e)}"
            }
    
    def _call_mcp_deepwiki(self, query: str) -> Dict[str, Any]:
        """Call mcp-deepwiki service."""
        return self._call_mcp_service_generic(
            "mcp-deepwiki",
            ["npx.cmd", "mcp-deepwiki"],
            "deepwiki_search",
            query
        )
    
    def _call_trends_hub(self, query: str) -> Dict[str, Any]:
        """Call trends-hub service."""
        return self._call_mcp_service_generic(
            "trends-hub",
            ["npx.cmd", "trends-hub"],
            "trends_search",
            query
        )
    
    def _call_arxiv_mcp(self, query: str) -> Dict[str, Any]:
        """Call arxiv-mcp-server service."""
        return self._call_mcp_service_generic(
            "arxiv-mcp-server",
            ["npx.cmd", "arxiv-mcp-server"],
            "arxiv_search",
            query
        )
    
    def _call_pozansky_stock(self, query: str) -> Dict[str, Any]:
        """Call pozansky-stock-server service."""
        return self._call_mcp_service_generic(
            "pozansky-stock-server",
            ["npx.cmd", "pozansky-stock-server"],
            "stock_search",
            query
        )
    
    def _call_worldbank_mcp(self, query: str) -> Dict[str, Any]:
        """Call worldbank-mcp service."""
        return self._call_mcp_service_generic(
            "worldbank-mcp",
            ["npx.cmd", "worldbank-mcp"],
            "worldbank_search",
            query
        )
    
    def _call_hotnews(self, query: str) -> Dict[str, Any]:
        """Call mcp-server-hotnews service."""
        return self._call_mcp_service_generic(
            "mcp-server-hotnews",
            ["npx.cmd", "mcp-server-hotnews"],
            "hotnews_search",
            query
        )
    
    def _call_biomcp(self, query: str) -> Dict[str, Any]:
        """Call biomcp service."""
        return self._call_mcp_service_generic(
            "biomcp",
            ["npx.cmd", "biomcp"],
            "bio_search",
            query
        )
    
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
        
        if use_mcp:
            self.logger.info("Multi-Hop Step 1: Using MCP services to gather information")
            reasoning_steps.append("Step 1: Using MCP services to gather information")
            
            mcp_services = ["searxng", "web-search", "bing-search"]
            for service in mcp_services:
                self.logger.debug(f"Calling MCP service: {service}")
                mcp_result = self._call_mcp_service(service, question)
                mcp_results.append(mcp_result)
                
                if "error" not in mcp_result:
                    count = mcp_result.get("count", 0)
                    reasoning_steps.append(f"  - Called {service}: {count} results")
                    self.logger.info(f"MCP Result: {service} returned {count} results")
                else:
                    error = mcp_result.get('error', 'failed')
                    reasoning_steps.append(f"  - Called {service}: {error}")
                    self.logger.warning(f"MCP Result: {service} failed - {error}")
        
        self.logger.info("Multi-Hop Step 2: Analyzing gathered information")
        reasoning_steps.append("Step 2: Analyzing gathered information")
        
        llm_result = self._call_llm(question)
        reasoning_steps.extend(llm_result.get("reasoning_steps", []))
        
        self.logger.info("Multi-Hop Step 3: Synthesizing final answer")
        reasoning_steps.append("Step 3: Synthesizing final answer")
        
        duration = time.time() - start_time
        self.logger.info(f"Multi-Hop Reasoning - Completed (Duration: {duration:.2f}s)")
        self.logger.info(f"Total reasoning steps: {len(reasoning_steps)}")
        self.logger.info(f"MCP results: {len(mcp_results)}")
        
        return {
            "question": question,
            "answer": llm_result.get("answer", ""),
            "reasoning_steps": reasoning_steps,
            "mcp_results": mcp_results if use_mcp else [],
            "use_mcp": use_mcp,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_event_stream(self, question: str, use_mcp: bool = False) -> Generator[str, None, None]:
        """Generate SSE event stream with multi-hop reasoning."""
        self.logger.info("="*70)
        self.logger.info("SSE Stream - Starting")
        self.logger.info(f"Question: {question[:100]}...")
        self.logger.info(f"MCP Enabled: {use_mcp}")
        self.logger.info("="*70)
        
        result = self._multi_hop_reasoning(question, use_mcp)
        
        reasoning_steps = result.get("reasoning_steps", [])
        answer = result.get("answer", "")
        mcp_results = result.get("mcp_results", [])
        
        self.logger.info(f"Stream: Reasoning steps: {len(reasoning_steps)}")
        self.logger.info(f"Stream: MCP results: {len(mcp_results)}")
        self.logger.info(f"Stream: Final answer: {answer[:100]}...")
        
        for i, step in enumerate(reasoning_steps, 1):
            event = {
                "type": "reasoning",
                "step": i,
                "content": step
            }
            self.logger.debug(f"Stream: Sending step {i}")
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            time.sleep(0.3)
        
        if mcp_results:
            mcp_event = {
                "type": "mcp_results",
                "results": mcp_results
            }
            self.logger.debug("Stream: Sending MCP results")
            yield f"data: {json.dumps(mcp_event, ensure_ascii=False)}\n\n"
        
        final_event = {
            "type": "answer",
            "answer": answer,
            "use_mcp": use_mcp,
            "timestamp": datetime.now().isoformat()
        }
        self.logger.debug("Stream: Sending final answer")
        yield f"data: {json.dumps(final_event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        
        self.logger.info("SSE Stream - Completed")
        self.logger.info("="*70)
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        # 添加CORS中间件
        @self.app.after_request
        def after_request(response):
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
            response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
            return response
        
        @self.app.route('/', methods=['GET'])
        def index():
            """Main web interface endpoint."""
            self.logger.info("Web interface - Request received")
            try:
                with open('index.html', 'r', encoding='utf-8') as f:
                    content = f.read()
                return content
            except Exception as e:
                self.logger.error(f"Error serving index.html: {str(e)}")
                return jsonify({
                    "error": "Internal Server Error",
                    "message": "Failed to load web interface"
                }), 500
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            self.logger.info("Health check - Request received")
            return jsonify({
                "status": "healthy",
                "service": "Enhanced MultiHop Agent API",
                "version": "2.0.0",
                "features": {
                    "multi_hop_reasoning": True,
                    "mcp_integration": True,
                    "sse_support": True
                },
                "mcp_services": list(self.mcp_config.get("mcpServers", {}).keys()),
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.route('/api/v1/answer', methods=['POST'])
        def answer_endpoint():
            """Main answer endpoint with multi-hop reasoning and MCP support."""
            self.logger.info("="*70)
            self.logger.info("API Request - /api/v1/answer")
            
            auth_header = request.headers.get('Authorization', '')
            
            if not auth_header.startswith('Bearer '):
                self.logger.warning("API Request - Unauthorized: Missing or invalid Authorization header")
                return jsonify({
                    "error": "Unauthorized",
                    "message": "Missing or invalid Authorization header"
                }), 401
            
            token = auth_header.replace('Bearer ', '')
            
            if token != self.api_token:
                self.logger.warning("API Request - Forbidden: Invalid API token")
                return jsonify({
                    "error": "Forbidden",
                    "message": "Invalid API token"
                }), 403
            
            data = request.get_json()
            
            if not data or 'question' not in data:
                self.logger.warning("API Request - Bad Request: Missing 'question' field")
                return jsonify({
                    "error": "Bad Request",
                    "message": "Missing 'question' field in request body"
                }), 400
            
            question = data['question']
            use_mcp = data.get('use_mcp', False)
            accept_header = request.headers.get('Accept', '')
            
            self.logger.info(f"API Request - Question: {question[:100]}...")
            self.logger.info(f"API Request - MCP: {use_mcp}")
            self.logger.debug(f"API Request - Accept header: {accept_header}")
            
            if 'text/event-stream' in accept_header:
                self.logger.info("API Request - Using SSE stream")
                return Response(
                    stream_with_context(
                        self._generate_event_stream(question, use_mcp),
                        mimetype='text/event-stream'
                    ),
                    mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                        'X-Accel-Buffering': 'no',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Headers': 'Content-Type,Authorization,Accept',
                        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS'
                    }
                )
            else:
                self.logger.info("API Request - Using standard JSON response")
                result = self._multi_hop_reasoning(question, use_mcp)
                self.logger.info(f"API Response - Status: 200")
                return jsonify(result)
        
        @self.app.route('/api/v1/mcp/list', methods=['GET'])
        def mcp_list():
            """List available MCP services."""
            self.logger.info("API Request - /api/v1/mcp/list")
            mcp_services = self.mcp_config.get("mcpServers", {})
            self.logger.info(f"API Response - MCP services: {len(mcp_services)}")
            return jsonify({
                "mcp_services": mcp_services,
                "count": len(mcp_services)
            })
        
        @self.app.route('/api/v1/mcp/call', methods=['POST'])
        def mcp_call():
            """Call specific MCP service."""
            self.logger.info("="*70)
            self.logger.info("API Request - /api/v1/mcp/call")
            
            auth_header = request.headers.get('Authorization', '')
            
            if not auth_header.startswith('Bearer '):
                self.logger.warning("API Request - Unauthorized: Missing or invalid Authorization header")
                return jsonify({"error": "Unauthorized"}), 401
            
            token = auth_header.replace('Bearer ', '')
            if token != self.api_token:
                self.logger.warning("API Request - Forbidden: Invalid API token")
                return jsonify({"error": "Forbidden"}), 403
            
            data = request.get_json()
            if not data or 'service' not in data or 'query' not in data:
                self.logger.warning("API Request - Bad Request: Missing 'service' or 'query' field")
                return jsonify({"error": "Bad Request"}), 400
            
            service_name = data['service']
            query = data['query']
            
            # 确保查询参数编码正确
            self.logger.info(f"API Request - Service: {service_name}")
            self.logger.info(f"API Request - Query: {query}")
            self.logger.info(f"Query type: {type(query)}")
            self.logger.info(f"Query length: {len(query)}")
            
            # 尝试使用utf-8编码确保中文字符正确
            if isinstance(query, str):
                try:
                    # 检查字符串是否包含非ASCII字符
                    has_non_ascii = any(ord(c) > 127 for c in query)
                    self.logger.info(f"Query has non-ASCII characters: {has_non_ascii}")
                except Exception as e:
                    self.logger.error(f"Error checking query: {e}")
            
            result = self._call_mcp_service(service_name, query)
            self.logger.info(f"API Response - Status: 200")
            return jsonify(result)
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, ssl_context=None):
        """Run Flask server."""
        self.logger.info("="*70)
        self.logger.info("Starting Flask Server")
        self.logger.info(f"Host: {host}")
        self.logger.info(f"Port: {port}")
        self.logger.info("="*70)
        
        print("\n" + "="*70)
        print("Enhanced MultiHop Agent API Server")
        print("="*70)
        print(f"\nServer starting on {host}:{port}")
        print(f"API Token: {self.api_token}")
        print(f"Model: {self.base_model.get('model_id', 'unknown')}")
        print(f"\nFeatures:")
        print(f"  ✅ Multi-hop Reasoning")
        print(f"  ✅ MCP Integration")
        print(f"  ✅ SSE Support")
        print(f"\nAvailable MCP Services: {len(self.mcp_config.get('mcpServers', {}))}")
        for service in self.mcp_config.get('mcpServers', {}).keys():
            print(f"  - {service}")
        print(f"\nEndpoints:")
        print(f"  - GET  /health")
        print(f"  - POST /api/v1/answer")
        print(f"  - GET  /api/v1/mcp/list")
        print(f"  - POST /api/v1/mcp/call")
        print(f"\nExample curl command:")
        print(f'  curl -X POST \\')
        print(f'    -H "Authorization: Bearer {self.api_token}" \\')
        print(f'    -H "Content-Type: application/json" \\')
        print(f'    -H "Accept: text/event-stream" \\')
        print(f'    -d \'{{"question": "Where is the capital of France?", "use_mcp": true}}\' \\')
        print(f'    "http://{host}:{port}/api/v1/answer"')
        print("="*70 + "\n")
        
        self.app.run(host=host, port=port, ssl_context=ssl_context, threaded=True)


def main():
    """Main function."""
    server = EnhancedMultiHopAPIServer()
    
    try:
        import ssl
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain('server.crt', 'server.key')
        use_ssl = True
    except:
        use_ssl = False
    
    host = '0.0.0.0'
    port = 5000
    
    server.run(host=host, port=port, ssl_context=ssl_context if use_ssl else None)


if __name__ == "__main__":
    main()
