#!/usr/bin/env python3
"""Interactive console interface with multi-hop reasoning and MCP.

Why: Provides a REPL-style command-line Q&A experience for quick
testing and debugging — no web server required.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
DEFAULT_MCP_CONFIG_PATH = BASE_DIR / "configs" / "mcp_config.json"
sys.path.insert(0, str(BASE_DIR))

from src.utils.logger_config import get_logger, MultiHopLogger
from apps.console.console_interface_mcp_service_handler import dispatch_console_mcp_service_call


class EnhancedMultiHopConsoleInterface:
    """REPL console for interactive question answering."""

    def __init__(self, config_path: Optional[str] = None):
        self.logger = get_logger("console", "console.log")
        self.logger.info("MultiHop Console Interface - Starting")
        self.config = self._load_config(config_path)
        self.base_model = self.config.get("base_model", {})
        self.mcp_config = self._load_mcp_config()
        self.history: List[Dict] = []
        self._load_history()

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
        return {}

    def _load_history(self) -> None:
        history_file = BASE_DIR / "console_history_enhanced.json"
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as fh:
                    self.history = json.load(fh)
            except Exception:
                self.history = []

    def _save_history(self) -> None:
        with open(BASE_DIR / "console_history_enhanced.json", "w", encoding="utf-8") as fh:
            json.dump(self.history[-50:], fh, indent=2, ensure_ascii=False)

    def _call_llm(self, question: str, context: str = "") -> Dict[str, Any]:
        """Send question (+ optional search context) to LLM."""
        start_time = time.time()
        self.logger.info("LLM call starting")
        api_url = self.base_model.get("api_url")
        api_key = self.base_model.get("api_key")
        model_id = self.base_model.get("model_id")

        system_prompt = (
            "You are a helpful AI assistant that answers questions accurately.\n"
            "Format: REASONING PROCESS:\n[Step 1: ...]\n...\nFINAL ANSWER:\n[answer]"
        )
        user_content = f"Context:\n{context}\n\nQuestion: {question}" if context else question
        payload = {
            "model": model_id,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_content}],
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
            self.logger.info("LLM success (%.2fs)", time.time() - start_time)
            return {"reasoning_steps": reasoning_steps, "answer": final_answer}
        except Exception as exc:
            self.logger.error("LLM failed: %s", exc)
            return {"reasoning_steps": [f"Error: {exc}"], "answer": f"Error: {exc}"}

    def _multi_hop_reasoning(self, question: str, use_mcp: bool = False) -> Dict[str, Any]:
        """Run MCP search + LLM reasoning pipeline."""
        reasoning_steps, mcp_results, context = [], [], ""
        if use_mcp:
            reasoning_steps.append("Step 1: Gathering info via MCP")
            for service in ["searxng", "web-search"]:
                result = dispatch_console_mcp_service_call(service, question, self.mcp_config, self.logger)
                mcp_results.append(result)
                if "error" not in result:
                    reasoning_steps.append(f"  - {service}: {result.get('count', 0)} results")
                    for idx, item in enumerate(result.get("results", [])[:3], 1):
                        context += f"\n[{idx}] {item.get('title', '')}\n{item.get('snippet', '')}\n"
                else:
                    reasoning_steps.append(f"  - {service}: {result.get('error')}")

        reasoning_steps.append("Step 2: Analysing information")
        llm_result = self._call_llm(question, context)
        reasoning_steps.extend(llm_result.get("reasoning_steps", []))
        reasoning_steps.append("Step 3: Synthesizing answer")
        return {
            "question": question, "answer": llm_result.get("answer", ""),
            "reasoning_steps": reasoning_steps,
            "mcp_results": mcp_results if use_mcp else [],
            "use_mcp": use_mcp, "timestamp": datetime.now().isoformat(),
        }

    def process_question(self, question: str, use_mcp: bool = False) -> None:
        """Process a question and display results."""
        print(f"\n{'=' * 70}\nQuestion: {question}\n{'=' * 70}")
        print(f"\nReasoning (MCP: {'on' if use_mcp else 'off'})...")
        result = self._multi_hop_reasoning(question, use_mcp)

        for idx, step in enumerate(result.get("reasoning_steps", []), 1):
            print(f"  Step {idx}: {step}")
        if result.get("mcp_results"):
            print("\nMCP Results:")
            for item in result["mcp_results"]:
                status = f"{item.get('count', 0)} results" if "error" not in item else f"Error: {item['error']}"
                print(f"  - {item.get('service', '?')}: {status}")
        print(f"\nFinal Answer:\n  {result.get('answer', '')}")

        self.history.insert(0, {
            "question": question, "answer": result.get("answer", ""),
            "use_mcp": use_mcp, "timestamp": datetime.now().isoformat(),
        })
        self._save_history()

    def show_history(self, limit: int = 5) -> None:
        print(f"\n{'=' * 70}\nRecent {limit} entries\n{'=' * 70}")
        for idx, item in enumerate(self.history[:limit], 1):
            print(f"\n[{idx}] Q: {item['question']}")
            print(f"    A: {item['answer'][:100]}...")

    def show_help(self) -> None:
        print(f"\n{'=' * 70}\nCommands:\n"
              f"  <question>     — ask directly\n"
              f"  /mcp <question>— ask with MCP search\n"
              f"  /history [n]   — show last n entries\n"
              f"  /clear         — clear screen\n"
              f"  /help          — this help\n"
              f"  /quit          — exit\n"
              f"\nModel: {self.base_model.get('model_id', 'unknown')}\n"
              f"MCP services: {len(self.mcp_config.get('mcpServers', {}))}\n{'=' * 70}")

    def run(self) -> None:
        """Main REPL loop."""
        print(f"\nMultiHop Console | Model: {self.base_model.get('model_id', 'unknown')}")
        self.show_help()

        if not sys.stdin.isatty():
            for line in sys.stdin:
                if line.strip():
                    self.process_question(line.strip())
            return

        while True:
            try:
                user_input = input("\n> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("/quit", "/exit", "q"):
                    print("\nGoodbye!")
                    break
                if user_input.lower() in ("/clear", "/cls"):
                    os.system("cls" if os.name == "nt" else "clear")
                    continue
                if user_input.lower() in ("/help", "/h"):
                    self.show_help()
                    continue
                if user_input.lower().startswith("/history"):
                    parts = user_input.split()
                    self.show_history(int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5)
                    continue
                if user_input.lower().startswith("/mcp"):
                    question = user_input[4:].strip()
                    if question:
                        self.process_question(question, use_mcp=True)
                    else:
                        print("Please provide a question after /mcp")
                    continue
                self.process_question(user_input)
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
            except Exception as exc:
                print(f"\nError: {exc}")


def main():
    EnhancedMultiHopConsoleInterface().run()


if __name__ == "__main__":
    main()
