#!/usr/bin/env python3
"""
MultiHop Agent API Server
Provides HTTP/HTTPS endpoints for question answering with event-stream support.
"""



import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import requests
import yaml
from flask import Flask, Response, jsonify, request, stream_with_context

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"


class MultiHopAPIServer:
    """API Server for MultiHop Agent."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.base_model = self.config.get("base_model", {})
        self.api_token = self.config.get("api_token", "default_token_123456")
        self.app = Flask(__name__)
        self._setup_routes()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if not config_file.is_absolute():
            config_file = BASE_DIR / config_file
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _call_llm(self, question: str) -> Dict[str, Any]:
        """Call LLM API with reasoning process."""
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
    
    def _generate_event_stream(self, question: str) -> Generator[str, None, None]:
        """Generate SSE event stream."""
        print(f"\n[API] Processing question: {question}")
        print(f"[API] Starting reasoning process...")
        
        result = self._call_llm(question)
        
        reasoning_steps = result.get("reasoning_steps", [])
        answer = result.get("answer", "")
        
        print(f"[API] Reasoning steps: {len(reasoning_steps)}")
        print(f"[API] Final answer: {answer[:100]}...")
        
        for i, step in enumerate(reasoning_steps, 1):
            event = {
                "type": "reasoning",
                "step": i,
                "content": step
            }
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            time.sleep(0.3)
        
        final_event = {
            "type": "answer",
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
        yield f"data: {json.dumps(final_event, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                "status": "healthy",
                "service": "MultiHop Agent API",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat()
            })
        
        @self.app.route('/api/v1/answer', methods=['POST'])
        def answer_endpoint():
            """Main answer endpoint with event-stream support."""
            auth_header = request.headers.get('Authorization', '')
            
            if not auth_header.startswith('Bearer '):
                return jsonify({
                    "error": "Unauthorized",
                    "message": "Missing or invalid Authorization header"
                }), 401
            
            token = auth_header.replace('Bearer ', '')
            
            if token != self.api_token:
                return jsonify({
                    "error": "Forbidden",
                    "message": "Invalid API token"
                }), 403
            
            data = request.get_json()
            
            if not data or 'question' not in data:
                return jsonify({
                    "error": "Bad Request",
                    "message": "Missing 'question' field in request body"
                }), 400
            
            question = data['question']
            accept_header = request.headers.get('Accept', '')
            
            if 'text/event-stream' in accept_header:
                return Response(
                    stream_with_context(
                        self._generate_event_stream(question),
                        mimetype='text/event-stream'
                    ),
                    mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                        'X-Accel-Buffering': 'no'
                    }
                )
            else:
                result = self._call_llm(question)
                return jsonify({
                    "answer": result.get("answer", ""),
                    "reasoning_steps": result.get("reasoning_steps", []),
                    "timestamp": datetime.now().isoformat()
                })
        
        @self.app.route('/api/v1/batch', methods=['POST'])
        def batch_endpoint():
            """Batch processing endpoint."""
            auth_header = request.headers.get('Authorization', '')
            
            if not auth_header.startswith('Bearer '):
                return jsonify({
                    "error": "Unauthorized",
                    "message": "Missing or invalid Authorization header"
                }), 401
            
            token = auth_header.replace('Bearer ', '')
            
            if token != self.api_token:
                return jsonify({
                    "error": "Forbidden",
                    "message": "Invalid API token"
                }), 403
            
            data = request.get_json()
            
            if not data or 'questions' not in data:
                return jsonify({
                    "error": "Bad Request",
                    "message": "Missing 'questions' field in request body"
                }), 400
            
            questions = data['questions']
            results = []
            
            for i, question in enumerate(questions, 1):
                print(f"\n[API] Processing batch question {i}/{len(questions)}")
                result = self._call_llm(question)
                results.append({
                    "question": question,
                    "answer": result.get("answer", ""),
                    "reasoning_steps": result.get("reasoning_steps", [])
                })
            
            return jsonify({
                "results": results,
                "total": len(results),
                "timestamp": datetime.now().isoformat()
            })
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, ssl_context=None):
        """Run the Flask server."""
        print("\n" + "="*70)
        print("MultiHop Agent API Server")
        print("="*70)
        print(f"\nServer starting on {host}:{port}")
        print(f"API Token: {self.api_token}")
        print(f"Model: {self.base_model.get('model_id', 'unknown')}")
        print(f"\nEndpoints:")
        print(f"  - GET  /health")
        print(f"  - POST /api/v1/answer")
        print(f"  - POST /api/v1/batch")
        print(f"\nExample curl command:")
        print(f'  curl -X POST \\')
        print(f'    -H "Authorization: Bearer {self.api_token}" \\')
        print(f'    -H "Content-Type: application/json" \\')
        print(f'    -H "Accept: text/event-stream" \\')
        print(f'    -d \'{{"question": "Where is the capital of France?"}}\' \\')
        print(f'    "http://{host}:{port}/api/v1/answer"')
        print("="*70 + "\n")
        
        self.app.run(host=host, port=port, ssl_context=ssl_context, threaded=True)


def main():
    """Main function."""
    server = MultiHopAPIServer()
    
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