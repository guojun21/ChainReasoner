#!/usr/bin/env python3
"""Single-question test to verify the multi-perspective logging system.

Why: After adding per-run faithful API call trace logging, we need a fast
way to confirm that all 4 log files (eval_progress, llm_trace, search_trace,
mcp_trace) are correctly generated with complete request/response data.

This script runs ONLY question[0] and then validates every log file.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "logs"
sys.path.insert(0, str(BASE_DIR))

from apps.api.enhanced_multi_hop_api_server import EnhancedMultiHopReasoningApiServer
from src.utils.per_run_faithful_api_call_trace_logger import create_per_run_trace_logger_and_directory

QUESTIONS_FILE = BASE_DIR / "data" / "qa" / "question.json"


def load_first_question() -> dict:
    """Load a question from the dataset by index (default: 0).

    Set QUESTION_INDEX env var to change which question to test.
    """
    target_idx = int(os.environ.get("QUESTION_INDEX", "0"))
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as fh:
        idx = 0
        for line in fh:
            line = line.strip()
            if line:
                if idx == target_idx:
                    return json.loads(line)
                idx += 1
    raise RuntimeError(f"Question index {target_idx} not found in question.json")


def verify_log_file(file_path: Path, expected_format: str) -> dict:
    """Verify a log file exists, is non-empty, and has valid content.

    Returns a summary dict with line_count, file_size, sample_line.
    """
    result = {"path": str(file_path), "exists": file_path.exists(), "valid": False}
    if not file_path.exists():
        result["error"] = "file does not exist"
        return result

    file_size = file_path.stat().st_size
    result["file_size_bytes"] = file_size
    if file_size == 0:
        result["error"] = "file is empty"
        return result

    with open(file_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    result["line_count"] = len(lines)

    if expected_format == "jsonl":
        # Verify each line is valid JSON
        valid_lines = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                valid_lines += 1
            except json.JSONDecodeError as exc:
                result["error"] = f"invalid JSON on line {valid_lines + 1}: {exc}"
                return result
        result["valid_record_count"] = valid_lines
        # Show first record as sample
        if lines:
            first_record = json.loads(lines[0].strip())
            result["sample_keys"] = list(first_record.keys())
            result["sample_type"] = first_record.get("type", first_record.get("purpose", "?"))
    elif expected_format == "text":
        result["valid_record_count"] = len([l for l in lines if l.strip()])
        if lines:
            result["first_line"] = lines[0].strip()[:100]

    result["valid"] = True
    return result


def main():
    print("\n" + "=" * 60)
    print("  Multi-Perspective Logging Verification Test")
    print("=" * 60 + "\n")

    # Step 1: Create per-run trace logger
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_logger = create_per_run_trace_logger_and_directory(
        LOG_DIR, run_timestamp)
    run_dir = trace_logger.run_directory
    print(f"[1] Per-run log directory: {run_dir}\n")

    # Step 1.5: Create eval_progress.txt via DualOutputStreamWriter-style tee
    eval_progress_path = run_dir / "eval_progress.txt"
    eval_progress_file = open(eval_progress_path, "w", encoding="utf-8")

    class _TeeWriter:
        """Minimal tee that mirrors stdout to eval_progress.txt."""
        def __init__(self, terminal, log_file):
            self._terminal = terminal
            self._log_file = log_file
        def write(self, msg):
            self._terminal.write(msg)
            self._log_file.write(msg)
            self._log_file.flush()
        def flush(self):
            self._terminal.flush()
            self._log_file.flush()

    original_stdout = sys.stdout
    sys.stdout = _TeeWriter(original_stdout, eval_progress_file)

    # Step 2: Initialize server with trace logger
    print("[2] Initializing server with trace logger...")
    server = EnhancedMultiHopReasoningApiServer(trace_logger=trace_logger)
    print(f"    Server ready. Model: {server.base_model.get('model_id', '?')}\n")

    # Step 3: Load and run question[0]
    question_data = load_first_question()
    question_text = question_data.get("question", "")
    question_id = question_data.get("id", 0)
    print(f"[3] Running question {question_id}: {question_text[:80]}...")
    start_time = time.time()
    result = server._multi_hop_reasoning(question_text, use_mcp=True)
    elapsed = time.time() - start_time
    print(f"    Answer: {result.get('answer', 'N/A')}")
    print(f"    Elapsed: {elapsed:.1f}s")
    print(f"    Reasoning steps: {len(result.get('reasoning_steps', []))}")
    print(f"    Search traces: {len(result.get('mcp_results', []))}\n")

    # Step 4: Close trace logger and eval progress file to flush everything
    trace_logger.close()
    eval_progress_file.flush()
    eval_progress_file.close()
    sys.stdout = original_stdout
    print("[4] Trace logger closed. Verifying log files...\n")

    # Step 5: Verify all 4 log files
    log_files = {
        "eval_progress.txt": ("text", "视角1: 评测进度"),
        "llm_api_call_trace.jsonl": ("jsonl", "视角2: LLM API 调用"),
        "search_api_call_trace.jsonl": ("jsonl", "视角3: 搜索引擎调用"),
        "mcp_transport_trace.jsonl": ("jsonl", "视角4: MCP 传输层调用"),
    }

    all_passed = True
    print("=" * 60)
    print("  Log File Verification Results")
    print("=" * 60)

    for filename, (fmt, description) in log_files.items():
        file_path = run_dir / filename
        info = verify_log_file(file_path, fmt)
        status = "PASS" if info["valid"] else "FAIL"
        if not info["valid"]:
            all_passed = False

        print(f"\n  [{status}] {description}")
        print(f"       File: {filename}")
        if info.get("file_size_bytes") is not None:
            print(f"       Size: {info['file_size_bytes']:,} bytes")
        if info.get("line_count") is not None:
            print(f"       Lines: {info['line_count']}")
        if info.get("valid_record_count") is not None:
            print(f"       Records: {info['valid_record_count']}")
        if info.get("sample_keys"):
            print(f"       Sample keys: {info['sample_keys']}")
        if info.get("sample_type"):
            print(f"       Sample type: {info['sample_type']}")
        if info.get("first_line"):
            print(f"       First line: {info['first_line']}")
        if info.get("error"):
            print(f"       Error: {info['error']}")

    print("\n" + "=" * 60)
    if all_passed:
        print("  ALL LOG FILES VERIFIED SUCCESSFULLY")
    else:
        print("  SOME LOG FILES FAILED VERIFICATION")
        sys.exit(1)
    print("=" * 60)
    print(f"\n  Run directory: {run_dir}\n")


if __name__ == "__main__":
    main()
