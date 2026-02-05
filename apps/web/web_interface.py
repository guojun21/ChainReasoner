#!/usr/bin/env python3
"""
Enhanced MultiHop Agent Web Interface
Provides a web-based interface with multi-hop reasoning and MCP integration.
"""



import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml
from flask import Flask, jsonify, render_template_string, request

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
DEFAULT_MCP_CONFIG_PATH = BASE_DIR / "configs" / "mcp_config.json"
sys.path.insert(0, str(BASE_DIR))

from src.utils.logger_config import get_logger, MultiHopLogger


class EnhancedMultiHopWebInterface:
    """Enhanced Web Interface with Multi-hop Reasoning and MCP Integration."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = get_logger("web", "web.log")
        self.logger.info("="*70)
        self.logger.info("Enhanced MultiHop Web Interface - Starting")
        self.logger.info("="*70)
        
        self.config = self._load_config(config_path)
        self.base_model = self.config.get("base_model", {})
        self.api_token = self.config.get("api_token", "default_token_123456")
        self.mcp_config = self._load_mcp_config()
        self.app = Flask(__name__)
        self._setup_routes()
        self._setup_template()
        
        self.logger.info(f"Model: {self.base_model.get('model_id', 'unknown')}")
        self.logger.info(f"MCP Services: {len(self.mcp_config.get('mcpServers', {}))} available")
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if not config_file.is_absolute():
            config_file = BASE_DIR / config_file
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_mcp_config(self) -> Dict[str, Any]:
        """Load MCP configuration."""
        if DEFAULT_MCP_CONFIG_PATH.exists():
            with open(DEFAULT_MCP_CONFIG_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"mcpServers": {}}
    
    def _call_llm(self, question: str) -> Dict[str, Any]:
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
        mcp_servers = self.mcp_config.get("mcpServers", {})
        
        if service_name not in mcp_servers:
            return {
                "error": f"MCP service '{service_name}' not found",
                "available_services": list(mcp_servers.keys())
            }
        
        print(f"\n[Web MCP] Calling service: {service_name}")
        print(f"[Web MCP] Query: {query}")
        
        try:
            if service_name == "searxng":
                return self._call_searxng(query)
            elif service_name == "web-search":
                return self._call_web_search(query)
            else:
                return {
                    "error": f"MCP service '{service_name}' not yet implemented",
                    "note": "This service is configured but not yet integrated"
                }
        except Exception as e:
            return {
                "error": f"MCP service error: {str(e)}"
            }
    
    def _call_searxng(self, query: str) -> Dict[str, Any]:
        """Call SearXNG search service."""
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
                return {
                    "service": "searxng",
                    "query": query,
                    "results": results["results"][:5],
                    "count": len(results.get("results", []))
                }
            else:
                return {
                    "service": "searxng",
                    "query": query,
                    "results": [],
                    "count": 0
                }
        except Exception as e:
            return {
                "error": f"SearXNG error: {str(e)}"
            }
    
    def _call_web_search(self, query: str) -> Dict[str, Any]:
        """Call web search service."""
        try:
            response = requests.get(
                "https://duckduckgo.com/html/",
                params={"q": query},
                timeout=30
            )
            response.raise_for_status()
            
            return {
                "service": "web-search",
                "query": query,
                "status": "success",
                "note": "Web search completed"
            }
        except Exception as e:
            return {
                "error": f"Web search error: {str(e)}"
            }
    
    def _multi_hop_reasoning(self, question: str, use_mcp: bool = False) -> Dict[str, Any]:
        """Perform multi-hop reasoning with optional MCP integration."""
        print(f"\n[Web MultiHop] Starting multi-hop reasoning for: {question}")
        print(f"[Web MultiHop] MCP enabled: {use_mcp}")
        
        reasoning_steps = []
        mcp_results = []
        
        if use_mcp:
            reasoning_steps.append("Step 1: 使用MCP服务收集信息")
            
            mcp_services = ["searxng", "web-search"]
            for service in mcp_services:
                mcp_result = self._call_mcp_service(service, question)
                mcp_results.append(mcp_result)
                
                if "error" not in mcp_result:
                    count = mcp_result.get("count", 0)
                    reasoning_steps.append(f"  - 调用 {service}: 获取 {count} 条结果")
                else:
                    reasoning_steps.append(f"  - 调用 {service}: {mcp_result.get('error', 'failed')}")
        
        reasoning_steps.append("Step 2: 分析收集的信息")
        
        llm_result = self._call_llm(question)
        reasoning_steps.extend(llm_result.get("reasoning_steps", []))
        
        reasoning_steps.append("Step 3: 综合最终答案")
        
        return {
            "question": question,
            "answer": llm_result.get("answer", ""),
            "reasoning_steps": reasoning_steps,
            "mcp_results": mcp_results if use_mcp else [],
            "use_mcp": use_mcp,
            "timestamp": datetime.now().isoformat()
        }
    
    def _setup_template(self):
        """Setup HTML template."""
        self.html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MultiHop Agent - 增强版智能问答系统</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .input-section h2, .output-section h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.5em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        textarea {
            width: 100%;
            min-height: 120px;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .mcp-toggle {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .mcp-toggle label {
            display: flex;
            align-items: center;
            gap: 5px;
            cursor: pointer;
            font-weight: 600;
            color: #333;
        }
        
        .mcp-toggle input[type="checkbox"] {
            width: 18px;
            height: 18px;
            cursor: pointer;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 1.1em;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            width: 100%;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .reasoning-steps {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .reasoning-steps h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        
        .step {
            padding: 10px 15px;
            margin-bottom: 10px;
            background: white;
            border-left: 4px solid #667eea;
            border-radius: 4px;
            font-size: 0.95em;
            line-height: 1.6;
        }
        
        .question-box {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .question-box h3 {
            margin-bottom: 10px;
            font-size: 1.2em;
        }
        
        .question-content {
            font-size: 1.1em;
            line-height: 1.6;
            font-weight: 500;
        }
        
        .answer-box {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin-top: 20px;
        }
        
        .answer-box h3 {
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .answer-content {
            font-size: 1.1em;
            line-height: 1.7;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .mcp-results {
            background: #fff3cd;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .mcp-results h3 {
            color: #764ba2;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        
        .mcp-result {
            padding: 10px 15px;
            margin-bottom: 10px;
            background: white;
            border-left: 3px solid #764ba2;
            border-radius: 4px;
            font-size: 0.9em;
        }
        
        .mcp-result .service-name {
            font-weight: 600;
            color: #764ba2;
        }
        
        .mcp-result .result-count {
            color: #666;
            font-size: 0.85em;
        }
        
        .empty-state {
            text-align: center;
            color: #999;
            padding: 40px;
            font-style: italic;
        }
        
        .history-section {
            margin-top: 30px;
        }
        
        .history-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 3px solid #764ba2;
        }
        
        .history-question {
            font-weight: 600;
            color: #333;
            margin-bottom: 5px;
        }
        
        .history-answer {
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 MultiHop Agent - 增强版</h1>
            <p>支持多跳推理和MCP服务集成的智能问答系统</p>
        </div>
        
        <div class="main-content">
            <div class="card input-section">
                <h2>📝 提问</h2>
                <form method="POST" action="/ask">
                    <div class="form-group">
                        <label for="question">请输入您的问题：</label>
                        <textarea 
                            id="question" 
                            name="question" 
                            placeholder="例如：中国的首都是哪里？"
                            required
                        ></textarea>
                    </div>
                    
                    <div class="mcp-toggle">
                        <label>
                            <input type="checkbox" name="use_mcp" value="true">
                            <span>启用MCP服务增强搜索</span>
                        </label>
                        <span style="color: #666; font-size: 0.9em;">
                            （使用SearXNG和Web搜索服务）
                        </span>
                    </div>
                    
                    <button type="submit" id="submitBtn">
                        提交问题
                    </button>
                </form>
            </div>
            
            <div class="card output-section">
                <h2>💡 答案</h2>
                
                {% if question %}
                <div class="question-box">
                    <h3>❓ 问题</h3>
                    <div class="question-content">{{ question }}</div>
                </div>
                {% endif %}
                
                {% if reasoning_steps %}
                <div class="reasoning-steps">
                    <h3>🧠 推理过程</h3>
                    {% for step in reasoning_steps %}
                    <div class="step">{{ step }}</div>
                    {% endfor %}
                </div>
                {% endif %}
                
                {% if mcp_results %}
                <div class="mcp-results">
                    <h3>🔍 MCP搜索结果</h3>
                    {% for result in mcp_results %}
                    <div class="mcp-result">
                        <span class="service-name">{{ result.get('service', 'Unknown') }}:</span>
                        {% if 'error' in result %}
                            <span style="color: #dc3545;">{{ result['error'] }}</span>
                        {% else %}
                            <span class="result-count">获取 {{ result.get('count', 0) }} 条结果</span>
                        {% endif %}
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
                
                {% if answer %}
                <div class="answer-box">
                    <h3>✅ 最终答案</h3>
                    <div class="answer-content">{{ answer }}</div>
                </div>
                {% endif %}
                
                {% if not answer and not reasoning_steps %}
                <div class="empty-state">
                    请在左侧输入问题并提交...
                </div>
                {% endif %}
            </div>
        </div>
        
        {% if history %}
        <div class="card history-section">
            <h2>📚 历史记录</h2>
            {% for item in history %}
            <div class="history-item">
                <div class="history-question">Q: {{ item.question }}</div>
                <div class="history-answer">A: {{ item.answer[:100] }}...</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
    </div>
</body>
</html>
        """
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/', methods=['GET'])
        def index():
            """Home page."""
            return render_template_string(self.html_template)
        
        @self.app.route('/ask', methods=['GET', 'POST'])
        def ask():
            """Process question."""
            if request.method == 'GET':
                self.logger.info("Web Request - GET /ask")
                return render_template_string(self.html_template)
            
            self.logger.info("="*70)
            self.logger.info("Web Request - POST /ask")
            
            question = request.form.get('question', '').strip()
            use_mcp = request.form.get('use_mcp') == 'true'
            
            if not question:
                self.logger.warning("Web Request - Bad Request: Missing question")
                return render_template_string(self.html_template, error="请输入问题")
            
            self.logger.info(f"Web Request - Question: {question[:100]}...")
            self.logger.info(f"Web Request - MCP: {use_mcp}")
            
            result = self._multi_hop_reasoning(question, use_mcp)
            
            reasoning_steps = result.get("reasoning_steps", [])
            answer = result.get("answer", "")
            mcp_results = result.get("mcp_results", [])
            
            self.logger.info(f"Web Response - Reasoning steps: {len(reasoning_steps)}")
            self.logger.info(f"Web Response - MCP results: {len(mcp_results)}")
            self.logger.info(f"Web Response - Answer: {answer[:100]}...")
            
            return render_template_string(
                self.html_template,
                question=question,
                reasoning_steps=reasoning_steps,
                answer=answer,
                mcp_results=mcp_results if mcp_results else None,
                history=self._get_history()
            )
        
        @self.app.route('/api/ask', methods=['POST'])
        def api_ask():
            """API endpoint for programmatic access."""
            self.logger.info("="*70)
            self.logger.info("Web Request - POST /api/ask")
            
            data = request.get_json()
            
            if not data or 'question' not in data:
                self.logger.warning("Web Request - Bad Request: Missing 'question' field")
                return jsonify({
                    "error": "Bad Request",
                    "message": "Missing 'question' field"
                }), 400
            
            question = data['question']
            use_mcp = data.get('use_mcp', False)
            
            self.logger.info(f"Web Request - Question: {question[:100]}...")
            self.logger.info(f"Web Request - MCP: {use_mcp}")
            
            result = self._multi_hop_reasoning(question, use_mcp)
            
            self.logger.info(f"Web Response - Status: 200")
            return jsonify(result)
        
        @self.app.route('/mcp/list', methods=['GET'])
        def mcp_list():
            """List available MCP services."""
            return jsonify({
                "mcp_services": self.mcp_config.get("mcpServers", {}),
                "count": len(self.mcp_config.get("mcpServers", {}))
            })
    
    def _get_history(self) -> List[Dict[str, str]]:
        """Get question history from file."""
        history_file = Path("web_history.json")
        
        if not history_file.exists():
            return []
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    
    def _save_to_history(self, question: str, answer: str, use_mcp: bool):
        """Save question to history."""
        history = self._get_history()
        
        history.insert(0, {
            "question": question,
            "answer": answer,
            "use_mcp": use_mcp,
            "timestamp": datetime.now().isoformat()
        })
        
        history = history[:20]
        
        with open("web_history.json", 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    def run(self, host: str = '0.0.0.0', port: int = 8080):
        """Run Flask web server."""
        print("\n" + "="*70)
        print("Enhanced MultiHop Agent Web Interface")
        print("="*70)
        print(f"\nServer starting on http://{host}:{port}")
        print(f"Model: {self.base_model.get('model_id', 'unknown')}")
        print(f"\nFeatures:")
        print(f"  ✅ Multi-hop Reasoning")
        print(f"  ✅ MCP Integration")
        print(f"  ✅ Enhanced UI")
        print(f"\nAvailable MCP Services: {len(self.mcp_config.get('mcpServers', {}))}")
        for service in self.mcp_config.get('mcpServers', {}).keys():
            print(f"  - {service}")
        print(f"\nOpen your browser and visit: http://localhost:{port}")
        print("="*70 + "\n")
        
        self.app.run(host=host, port=port, debug=False)


def main():
    """Main function."""
    web = EnhancedMultiHopWebInterface()
    web.run()


if __name__ == "__main__":
    main()
