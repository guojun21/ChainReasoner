#!/usr/bin/env python3
"""Enhanced web interface with multi-hop reasoning and MCP integration.

Why: Provides a browser-based UI for interactive question answering
with visual reasoning steps and MCP search results.
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
from apps.web.web_templates import ENHANCED_TEMPLATE
from apps.web.web_interface_mcp_service_handler import dispatch_web_mcp_service_call


class EnhancedMultiHopReasoningWebInterface:
    """Flask web app: form-based Q&A with multi-hop reasoning."""

    def __init__(self, config_path: Optional[str] = None):
        self.logger = get_logger("web", "web.log")
        self.logger.info("Enhanced MultiHop Web Interface - Starting")
        self.config = self._load_config(config_path)
        self.base_model = self.config.get("base_model", {})
        self.api_token = self.config.get("api_token", "default_token_123456")
        self.mcp_config = self._load_mcp_config()
        self.html_template = ENHANCED_TEMPLATE
        self.app = Flask(__name__)
        self._setup_routes()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        config_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if not config_file.is_absolute():
            config_file = BASE_DIR / config_file
        with open(config_file, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def _load_mcp_config(self) -> Dict[str, Any]:
        if DEFAULT_MCP_CONFIG_PATH.exists():
            with open(DEFAULT_MCP_CONFIG_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {"mcpServers": {}}

    def _call_llm(self, question: str) -> Dict[str, Any]:
        """Send question to LLM and parse structured response."""
        start_time = time.time()
        self.logger.info("LLM API Call - Starting")
        api_url = self.base_model.get("api_url")
        api_key = self.base_model.get("api_key")
        model_id = self.base_model.get("model_id")

        system_prompt = (
            "You are a helpful AI assistant that answers questions accurately and concisely.\n"
            "Format: REASONING PROCESS:\n[Step 1: ...]\n...\nFINAL ANSWER:\n[answer]"
        )
        payload = {
            "model": model_id,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": question}],
            "temperature": self.base_model.get("temperature", 0.7),
            "max_tokens": self.base_model.get("max_tokens", 2048),
        }
        try:
            response = requests.post(api_url, headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"
            }, json=payload, timeout=60)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]

            reasoning_steps, final_answer = [], content
            if "REASONING PROCESS:" in content:
                parts = content.split("REASONING PROCESS:")[1].split("FINAL ANSWER:")
                reasoning_steps = [l.strip() for l in parts[0].strip().split("\n") if l.strip()]
                final_answer = parts[1].strip() if len(parts) > 1 else ""
            duration = time.time() - start_time
            self.logger.info("LLM Success (%.2fs)", duration)
            return {"reasoning_steps": reasoning_steps, "answer": final_answer}
        except Exception as exc:
            self.logger.error("LLM Failed: %s", exc)
            return {"reasoning_steps": [f"Error: {exc}"], "answer": f"Error: {exc}"}

    def _multi_hop_reasoning(self, question: str, use_mcp: bool = False) -> Dict[str, Any]:
        """Run multi-hop pipeline: optional MCP search + LLM reasoning."""
        reasoning_steps, mcp_results = [], []
        if use_mcp:
            reasoning_steps.append("Step 1: Collecting info via MCP services")
            for service in ["searxng", "web-search"]:
                result = dispatch_web_mcp_service_call(service, question, self.mcp_config)
                mcp_results.append(result)
                count = result.get("count", 0) if "error" not in result else result.get("error", "failed")
                reasoning_steps.append(f"  - {service}: {count}")
        reasoning_steps.append("Step 2: Analysing information")
        llm_result = self._call_llm(question)
        reasoning_steps.extend(llm_result.get("reasoning_steps", []))
        reasoning_steps.append("Step 3: Synthesizing answer")
        return {
            "question": question, "answer": llm_result.get("answer", ""),
            "reasoning_steps": reasoning_steps,
            "mcp_results": mcp_results if use_mcp else [],
            "use_mcp": use_mcp, "timestamp": datetime.now().isoformat(),
        }

    def _setup_routes(self) -> None:
        @self.app.route("/", methods=["GET"])
        def index():
            return render_template_string(self.html_template)

        @self.app.route("/ask", methods=["GET", "POST"])
        def ask():
            if request.method == "GET":
                return render_template_string(self.html_template)
            question = request.form.get("question", "").strip()
            use_mcp = request.form.get("use_mcp") == "true"
            if not question:
                return render_template_string(self.html_template, error="Please enter a question")
            self.logger.info("Question: %s...", question[:100])
            result = self._multi_hop_reasoning(question, use_mcp)
            return render_template_string(
                self.html_template, question=question,
                reasoning_steps=result.get("reasoning_steps", []),
                answer=result.get("answer", ""),
                mcp_results=result.get("mcp_results") or None,
                history=self._get_history(),
            )

        @self.app.route("/api/ask", methods=["POST"])
        def api_ask():
            data = request.get_json()
            if not data or "question" not in data:
                return jsonify({"error": "Missing 'question'"}), 400
            return jsonify(self._multi_hop_reasoning(data["question"], data.get("use_mcp", False)))

        @self.app.route("/mcp/list", methods=["GET"])
        def mcp_list():
            services = self.mcp_config.get("mcpServers", {})
            return jsonify({"mcp_services": services, "count": len(services)})

    def _get_history(self) -> List[Dict[str, str]]:
        history_file = Path("web_history.json")
        if not history_file.exists():
            return []
        try:
            with open(history_file, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return []

    def run(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        print(f"\nEnhanced MultiHop Web Interface on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=False)


def main():
    EnhancedMultiHopReasoningWebInterface().run()


if __name__ == "__main__":
    main()
