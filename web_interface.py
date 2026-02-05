#!/usr/bin/env python3
"""
MultiHop Agent Web Interface
Provides a web-based interface for question answering.
"""



import yaml
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from flask import Flask, render_template_string, request, jsonify


class MultiHopWebInterface:
    """Web Interface for MultiHop Agent."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.base_model = self.config.get("base_model", {})
        self.api_token = self.config.get("api_token", "default_token_123456")
        self.app = Flask(__name__)
        self._setup_routes()
        self._setup_template()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
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
    
    def _setup_template(self):
        """Setup HTML template."""
        self.html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MultiHop Agent - 智能问答系统</title>
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
            max-width: 1200px;
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
            <h1>🤖 MultiHop Agent</h1>
            <p>基于大语言模型的智能问答系统</p>
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
                            placeholder="例如：爱因斯坦在哪里工作？"
                            required
                        ></textarea>
                    </div>
                    <button type="submit" id="submitBtn">
                        提交问题
                    </button>
                </form>
            </div>
            
            <div class="card output-section">
                <h2>💡 答案</h2>
                {% if reasoning_steps %}
                <div class="reasoning-steps">
                    <h3>🧠 推理过程</h3>
                    {% for step in reasoning_steps %}
                    <div class="step">{{ step }}</div>
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
                return render_template_string(self.html_template)
            
            question = request.form.get('question', '').strip()
            
            if not question:
                return render_template_string(self.html_template, 
                    error="请输入问题")
            
            print(f"\n[Web] Processing question: {question}")
            
            result = self._call_llm(question)
            
            reasoning_steps = result.get("reasoning_steps", [])
            answer = result.get("answer", "")
            
            print(f"[Web] Reasoning steps: {len(reasoning_steps)}")
            print(f"[Web] Answer: {answer[:100]}...")
            
            return render_template_string(
                self.html_template,
                question=question,
                reasoning_steps=reasoning_steps,
                answer=answer,
                history=self._get_history()
            )
        
        @self.app.route('/api/ask', methods=['POST'])
        def api_ask():
            """API endpoint for programmatic access."""
            data = request.get_json()
            
            if not data or 'question' not in data:
                return jsonify({
                    "error": "Bad Request",
                    "message": "Missing 'question' field"
                }), 400
            
            question = data['question']
            result = self._call_llm(question)
            
            return jsonify({
                "answer": result.get("answer", ""),
                "reasoning_steps": result.get("reasoning_steps", []),
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.route('/history', methods=['GET'])
        def history():
            """Get question history."""
            return jsonify(self._get_history())
    
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
    
    def _save_to_history(self, question: str, answer: str):
        """Save question to history."""
        history = self._get_history()
        
        history.insert(0, {
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        })
        
        history = history[:20]
        
        with open("web_history.json", 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    def run(self, host: str = '0.0.0.0', port: int = 8080):
        """Run Flask web server."""
        print("\n" + "="*70)
        print("MultiHop Agent Web Interface")
        print("="*70)
        print(f"\nServer starting on http://{host}:{port}")
        print(f"Model: {self.base_model.get('model_id', 'unknown')}")
        print(f"\nOpen your browser and visit: http://localhost:{port}")
        print("="*70 + "\n")
        
        self.app.run(host=host, port=port, debug=False)


def main():
    """Main function."""
    web = MultiHopWebInterface()
    web.run()


if __name__ == "__main__":
    main()