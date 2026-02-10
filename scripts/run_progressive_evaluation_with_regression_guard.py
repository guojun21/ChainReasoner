#!/usr/bin/env python3
"""Progressive evaluation with early stopping and regression detection.

Why: Evaluates the answer pipeline in stages (5 -> 10 -> 20 -> ... -> 100)
with accuracy gates at each stage.  The "only-forward" mechanism prevents
code changes from regressing previously correct answers.

Outputs JSONL — the competition-required format.

== PERF-OPTIMIZED ==
Uses ThreadPoolExecutor for concurrent question processing (default 3 workers).

== DO NOT MODIFY SCORING ==
_normalize_for_compare() and score_results() are the scoring core.
Do not add abbreviation normalisation, fuzzy matching, or relaxed scoring.
"""

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
AUTORUN_DIR = BASE_DIR.parent / "autorunAtnight"
sys.path.insert(0, str(BASE_DIR))

from apps.api.enhanced_multi_hop_api_server import EnhancedMultiHopReasoningApiServer
from src.utils.per_run_faithful_api_call_trace_logger import create_per_run_trace_logger_and_directory
from scripts.evaluation_baseline_loader_and_regression_checker import (
    scan_output_directory_for_highest_scoring_baseline,
    detect_answer_regressions_against_baseline,
)

QUESTIONS_FILE = BASE_DIR / "data" / "qa" / "question.json"
STANDARD_FILE = BASE_DIR / "data" / "qa" / "the_standard_answers.json"
OUTPUT_DIR = AUTORUN_DIR / "outputs"
SUMMARY_FILE = OUTPUT_DIR / "last_progressive_summary.json"
LOG_DIR = BASE_DIR / "logs"

MAX_WORKERS = 3  # Brave API rate limit constrains this
_print_lock = threading.Lock()


# ── Tee: capture stdout to file + terminal ──────────────────────────────

class DualOutputStreamWriter:
    """Duplicate stdout to both terminal and a log file."""

    def __init__(self, log_path: Path):
        self._terminal = sys.stdout
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self._terminal.write(message)
        self._log_file.write(message)
        self._log_file.flush()

    def flush(self):
        self._terminal.flush()
        self._log_file.flush()

    def close(self):
        self._log_file.close()

    @property
    def log_path(self) -> Path:
        return Path(self._log_file.name)


# ── Scoring (DO NOT MODIFY) ─────────────────────────────────────────────

def _normalize_answer(value):
    """Whitespace-normalise an answer string."""
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _normalize_for_compare(value):
    """Lowercase + strip trailing period — DO NOT relax this."""
    if value is None:
        return ""
    text = " ".join(str(value).strip().split()).lower()
    return text.rstrip(".")


def score_results(results, standard_map):
    """Count exact-match correct answers — DO NOT MODIFY."""
    return sum(
        1 for item in results
        if _normalize_for_compare(item.get("answer")) == _normalize_for_compare(standard_map.get(str(item.get("id")), ""))
    )


# ── Data loading ────────────────────────────────────────────────────────

def load_questions(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_standard_map(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {str(item.get("id")): _normalize_answer(item.get("answer"))
            for item in data.get("answers", [])}


# ── Concurrent question processing ─────────────────────────────────────

def _answer_one_question(server, question_data, idx):
    """Process a single question. Returns (idx, {id, answer}, elapsed_s)."""
    question_id = question_data.get("id", idx)
    question_text = question_data.get("question", "")
    start_time = time.time()
    try:
        result = server._multi_hop_reasoning(question_text, use_mcp=True)
        answer = result.get("answer", "Unknown")
    except Exception as exc:
        answer = "Unknown"
        print(f"  [ERROR] Q{question_id}: {exc}", flush=True)
    return idx, {"id": question_id, "answer": answer}, time.time() - start_time


def _run_stage_concurrent(server, questions, start_idx, end_idx):
    """Run questions[start_idx:end_idx] concurrently; return results in order."""
    batch = [(questions[i], i) for i in range(start_idx, end_idx)]
    if not batch:
        return []
    results_dict = {}
    done_count = 0
    total = len(batch)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_answer_one_question, server, q, idx): idx
                   for q, idx in batch}
        for future in as_completed(futures):
            idx, result, elapsed = future.result()
            results_dict[idx] = result
            done_count += 1
            progress_pct = int(done_count / total * 100)
            bar = "\u2588" * (progress_pct // 5) + "\u2591" * (20 - progress_pct // 5)
            with _print_lock:
                print(f"  [{bar}] {progress_pct:3d}% | Q{result['id']:>3} ({done_count}/{total}) "
                      f"{elapsed:.0f}s => {result['answer'][:50]}", flush=True)

    return [results_dict[i] for i in range(start_idx, end_idx)]


# ── IO helpers ──────────────────────────────────────────────────────────

def save_results(results, score):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"{score}\u5206_{timestamp}_answers.jsonl"
    with open(output_file, "w", encoding="utf-8") as fh:
        for item in results:
            fh.write(json.dumps({"id": item["id"], "answer": item["answer"]}, ensure_ascii=False) + "\n")
    return output_file


def write_summary(summary):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    return SUMMARY_FILE


def print_search_backend_preflight_summary(summary: dict) -> None:
    """Print startup preflight status for each search backend."""
    print("\n[Preflight] Search backend availability check", flush=True)
    engines = summary.get("engines", {}) if isinstance(summary, dict) else {}
    for engine_name in ("google", "brave", "iqs", "duckduckgo"):
        info = engines.get(engine_name, {})
        configured = info.get("configured", False)
        enabled = info.get("enabled", False)
        status = info.get("status", "unknown")
        reason = info.get("reason", "") or "-"
        print(
            f"[Preflight] {engine_name:<10} configured={configured} enabled={enabled} "
            f"status={status} reason={reason[:180]}",
            flush=True,
        )


# ── Main ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Progressive evaluation for ChainReasoner")
    parser.add_argument("--no-stage", "--full", action="store_true", dest="no_stage",
                        help="Skip staged gates, run all 100 questions (regression detection still active)")
    parser.add_argument("--max-stage", type=int, default=None, metavar="N",
                        help="Only run up to stage N (e.g. --max-stage 1 = first 5 questions only). "
                             "Stages: 1=5q, 2=10q, 3=20q, 4=40q, 5=60q, 6=80q, 7=100q")
    parser.add_argument("--question-limit", type=int, default=None, metavar="N",
                        help="Only process the first N questions (e.g. --question-limit 1 for Q0 only). "
                             "Automatically skips staged gates.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Create per-run log directory with all trace files co-located
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_logger = create_per_run_trace_logger_and_directory(
        LOG_DIR, run_timestamp)
    run_dir = trace_logger.run_directory

    tee = DualOutputStreamWriter(run_dir / "eval_progress.txt")
    sys.stdout = tee
    print(f"[LOG] {tee.log_path}", flush=True)
    print(f"[TRACE] {run_dir}", flush=True)

    if not QUESTIONS_FILE.exists() or not STANDARD_FILE.exists():
        print("Missing questions or standard answers file")
        return

    questions = load_questions(QUESTIONS_FILE)
    standard_map = load_standard_map(STANDARD_FILE)

    # Apply question limit if specified
    if args.question_limit is not None:
        questions = questions[:args.question_limit]
        args.no_stage = True  # Skip staged gates when limiting questions
        print(f"[LIMIT] Question limit: {args.question_limit} (staged gates disabled)")

    server = EnhancedMultiHopReasoningApiServer(trace_logger=trace_logger)
    preflight_summary = server.run_search_backend_preflight_checks()
    print_search_backend_preflight_summary(preflight_summary)

    baseline_ids, baseline_file, baseline_score = scan_output_directory_for_highest_scoring_baseline(
        OUTPUT_DIR, standard_map, _normalize_for_compare)
    if baseline_ids:
        print(f"\n[Baseline] Best: {baseline_score} pts ({baseline_file})")
        print(f"[Baseline] Correct IDs ({len(baseline_ids)}): {sorted(baseline_ids, key=lambda x: int(x))}")
        print(f"[Baseline] Regressions on these will halt evaluation\n", flush=True)
    else:
        print("\n[Baseline] No history — regression detection skipped\n", flush=True)

    all_stages = [
        (5, 1), (10, 3), (20, 7), (40, 16), (60, 27), (80, 40), (100, None),
    ]
    if args.no_stage:
        stages = [(len(questions), None)]
    elif args.max_stage is not None:
        stages = all_stages[:max(1, args.max_stage)]
    else:
        stages = all_stages

    results, current_count, last_output = [], 0, None

    for stage_total, threshold in stages:
        target = min(stage_total, len(questions))
        if current_count < target:
            stage_start = time.time()
            new_count = target - current_count
            label = "FULL RUN" if args.no_stage else f"STAGE {stage_total}"
            print(f"\n{'=' * 60}\n  {label} | Q{current_count}~Q{target - 1} | {new_count} questions | workers={MAX_WORKERS}\n{'=' * 60}", flush=True)
            results.extend(_run_stage_concurrent(server, questions, current_count, target))
            print(f"{'=' * 60}\n  {label} done | {time.time() - stage_start:.0f}s\n{'=' * 60}", flush=True)

        current_count = target
        score = score_results(results, standard_map)

        # Regression check
        if baseline_ids:
            regressions = detect_answer_regressions_against_baseline(
                results, standard_map, baseline_ids, _normalize_for_compare)
            if regressions:
                last_output = save_results(results, score)
                print(f"\nRegression detected! Score {score}, {len(regressions)} regressions vs baseline {baseline_score}", flush=True)
                for reg in regressions:
                    print(f"  X Q{reg['qid']}: expected \"{reg['standard_answer']}\" got \"{reg['current_answer']}\"", flush=True)
                write_summary({"stage_total": stage_total, "ran": current_count, "score": score,
                               "threshold": threshold, "passed": False, "output_file": str(last_output),
                               "regression_detected": True, "regression_count": len(regressions),
                               "baseline_score": baseline_score})
                if not args.no_stage:
                    return
                print("[--no-stage] Regressions logged, continuing\n", flush=True)

        last_output = save_results(results, score)
        write_summary({"stage_total": stage_total, "ran": current_count, "score": score,
                        "threshold": threshold, "passed": True if threshold is None else score >= threshold,
                        "output_file": str(last_output)})
        print(f"\n{('FULL' if args.no_stage else f'Stage {stage_total}')} Score: {score}/{current_count}")
        print(f"Output: {last_output}\nSummary: {SUMMARY_FILE}")

        if not args.no_stage and threshold is not None and score < threshold:
            print("Stage failed, stopping early.")
            break

    if args.no_stage and results:
        submit_file = OUTPUT_DIR / f"{score}\u5206_{datetime.now().strftime('%Y%m%d_%H%M%S')}_submit.jsonl"
        with open(submit_file, "w", encoding="utf-8") as fh:
            for question_data in questions:
                question_id = question_data.get("id")
                matched = next((r for r in results if r["id"] == question_id), None)
                fh.write(json.dumps({"id": question_id, "answer": matched["answer"] if matched else "Unknown"}, ensure_ascii=False) + "\n")
        print(f"\nSubmit file: {submit_file}")


if __name__ == "__main__":
    main()
