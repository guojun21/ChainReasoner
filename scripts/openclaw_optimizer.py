#!/usr/bin/env python3
"""
OpenClaw-driven optimization loop for ChainReasoner.
Uses MCP HTTP interface to run progressive eval, read logs, suggest and apply tuning.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
AUTORUN_DIR = BASE_DIR.parent / "autorunAtnight"
MCP_URL = "http://localhost:3001/mcp"
EVAL_COMMAND = f'cd "{BASE_DIR}" && .venv/bin/python scripts/run_progressive_eval.py'
SUMMARY_FILE = AUTORUN_DIR / "outputs" / "last_progressive_summary.json"
PROGRESS_FILE = AUTORUN_DIR / "logs" / "openclaw_progress.json"
CONFIG_FILE = BASE_DIR / "configs" / "config.yaml"
API_LOG = AUTORUN_DIR / "logs" / "api.log"
TRACE_LOG = AUTORUN_DIR / "logs" / "search_trace.jsonl"
OPEN_DEEP_RESEARCH_DIR = BASE_DIR / "references" / "forRef" / "open_deep_research"
ODR_CLAUDE = OPEN_DEEP_RESEARCH_DIR / "CLAUDE.md"
ODR_README = OPEN_DEEP_RESEARCH_DIR / "README.md"
ODR_PROMPTS = OPEN_DEEP_RESEARCH_DIR / "src" / "open_deep_research" / "prompts.py"
ODR_CONFIG = OPEN_DEEP_RESEARCH_DIR / "src" / "open_deep_research" / "configuration.py"


class OpenClawClient:
    def __init__(self, mcp_url: str = MCP_URL):
        self.mcp_url = mcp_url

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "call_tool",
            "params": {"name": name, "arguments": arguments}
        }
        response = requests.post(self.mcp_url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("result", {})

    def read_file(self, path: Path) -> str:
        result = self.call_tool("read_file", {"path": str(path)})
        return result.get("content", "")

    def write_file(self, path: Path, content: str):
        self.call_tool("write_file", {"path": str(path), "content": content})

    def execute_command(self, command: str) -> str:
        result = self.call_tool("execute_command", {"command": command})
        return result.get("stdout", "")

    def llm_chat(self, prompt: str, model: str = "qwen3-max") -> str:
        result = self.call_tool("llm_chat", {"messages": [{"role": "user", "content": prompt}], "model": model})
        return result.get("content", "")


def parse_yaml(content: str) -> Dict[str, Any]:
    return yaml.safe_load(content) or {}


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_tail(content: str, max_chars: int = 4000) -> str:
    if not content:
        return ""
    return content[-max_chars:]


def heuristic_tuning(score: int, current: Dict[str, Any]) -> Dict[str, Any]:
    tuning = dict(current)
    if score < 2:
        tuning["max_queries"] = min(6, current.get("max_queries", 4) + 1)
        tuning["max_results_per_query"] = min(12, current.get("max_results_per_query", 8) + 2)
        tuning["max_evidence"] = min(10, current.get("max_evidence", 6) + 2)
        tuning["adaptive_threshold_n"] = max(0.3, current.get("adaptive_threshold_n", 0.5) - 0.1)
    elif score < 8:
        tuning["max_queries"] = min(6, current.get("max_queries", 4) + 1)
        tuning["adaptive_threshold_n"] = max(0.4, current.get("adaptive_threshold_n", 0.5) - 0.05)
    else:
        tuning["adaptive_threshold_n"] = min(0.6, current.get("adaptive_threshold_n", 0.5) + 0.05)
    return tuning


def build_prompt(odr_claude: str, odr_readme: str, odr_prompts: str, odr_config: str, summary: Dict[str, Any], api_log: str, trace_log: str) -> str:
    return f"""You are OpenClaw optimizer. Use open_deep_research as the primary reference.

## Open Deep Research CLAUDE
{odr_claude}

## Open Deep Research README (excerpt)
{odr_readme[:2000]}

## Open Deep Research Prompts (excerpt)
{odr_prompts[:2000]}

## Open Deep Research Config (excerpt)
{odr_config[:2000]}

## Current Summary
{json.dumps(summary, ensure_ascii=False, indent=2)}

## API Log (tail)
{api_log}

## Search Trace (tail)
{trace_log}

请输出JSON，仅包含字段 tuning，用于更新 config.yaml 中的 search_agent 参数。
示例：
{{"tuning": {{"max_queries": 5, "adaptive_threshold_n": 0.45, "enable_llm_answer": true}}}}
"""


def apply_tuning(config: Dict[str, Any], tuning: Dict[str, Any]) -> Dict[str, Any]:
    search_cfg = config.get("search_agent", {})
    for key, value in tuning.items():
        if key in [
            "max_queries",
            "per_query_delay",
            "max_results_per_query",
            "max_evidence",
            "adaptive_threshold_n",
            "brave_min_interval_seconds",
            "enable_rewrite",
            "enable_llm_answer",
            "enable_llm_verify"
        ]:
            search_cfg[key] = value
    config["search_agent"] = search_cfg
    return config


def run_round(client: OpenClawClient, round_id: int, round_minutes: int = 30) -> Optional[Dict[str, Any]]:
    start_time = time.time()
    print(f"Round {round_id} starting...")

    stdout = client.execute_command(EVAL_COMMAND)
    print(stdout)

    summary = load_json(SUMMARY_FILE)
    if not summary:
        print("No summary found. Stopping.")
        return None

    if summary.get("threshold") is not None and not summary.get("passed", False):
        print("Stage failed. Stopping.")
        return summary

    odr_claude = client.read_file(ODR_CLAUDE) if ODR_CLAUDE.exists() else ""
    odr_readme = client.read_file(ODR_README) if ODR_README.exists() else ""
    odr_prompts = client.read_file(ODR_PROMPTS) if ODR_PROMPTS.exists() else ""
    odr_config = client.read_file(ODR_CONFIG) if ODR_CONFIG.exists() else ""
    api_log_content = client.read_file(API_LOG) if API_LOG.exists() else ""
    trace_log_content = client.read_file(TRACE_LOG) if TRACE_LOG.exists() else ""
    api_log_tail = read_tail(api_log_content, 4000)
    trace_log_tail = read_tail(trace_log_content, 4000)

    prompt = build_prompt(odr_claude, odr_readme, odr_prompts, odr_config, summary, api_log_tail, trace_log_tail)
    llm_out = client.llm_chat(prompt)

    tuning = {}
    try:
        tuning = json.loads(llm_out).get("tuning", {})
    except Exception:
        tuning = {}

    config_content = client.read_file(CONFIG_FILE)
    config = parse_yaml(config_content)
    search_cfg = config.get("search_agent", {})
    if not tuning:
        tuning = heuristic_tuning(summary.get("score", 0), search_cfg)

    updated = apply_tuning(config, tuning)
    client.write_file(CONFIG_FILE, yaml.safe_dump(updated, allow_unicode=True, sort_keys=False))

    progress = load_json(PROGRESS_FILE)
    rounds = progress.get("rounds", [])
    rounds.append({
        "round": round_id,
        "started_at": datetime.now().isoformat(),
        "summary": summary,
        "tuning": tuning
    })
    progress["rounds"] = rounds
    progress["last_round"] = round_id
    progress["last_output"] = summary.get("output_file")
    save_json(PROGRESS_FILE, progress)

    elapsed = time.time() - start_time
    sleep_time = max(0, round_minutes * 60 - elapsed)
    if sleep_time:
        time.sleep(sleep_time)

    return summary


def main():
    client = OpenClawClient()
    max_rounds = 3
    for i in range(1, max_rounds + 1):
        summary = run_round(client, i)
        if not summary:
            break
        if summary.get("threshold") is not None and not summary.get("passed", False):
            break


if __name__ == "__main__":
    main()
