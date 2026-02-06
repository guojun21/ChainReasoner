#!/usr/bin/env python3
"""
Progressive evaluation with early stopping.
Stages: 5 -> 20 -> 100 with thresholds 2 and 8.
Outputs JSON array with id + answer only.

[PERF-OPTIMIZED 2026-02-06] 使用 ThreadPoolExecutor 并发处理题目，大幅提升速度。
  - 每阶段内的题目并发执行（默认 3 个 worker），受限于 Brave API rate limit
  - 每题完成立即打印耗时，便于观察
  - 注意：如果你（Cursor agent）正在修改 api_server.py 或 constrained_search.py，
    本脚本的并发逻辑不受影响，因为每个线程共享同一个 server 实例。
"""

import json
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
AUTORUN_DIR = BASE_DIR.parent / "autorunAtnight"
sys.path.insert(0, str(BASE_DIR))

from apps.api.api_server import EnhancedMultiHopAPIServer

QUESTIONS_FILE = BASE_DIR / "data" / "qa" / "question.json"
STANDARD_FILE = BASE_DIR / "data" / "qa" / "the_standard_answers.json"
OUTPUT_DIR = AUTORUN_DIR / "outputs"
SUMMARY_FILE = OUTPUT_DIR / "last_progressive_summary.json"

# 并发 worker 数（受 Brave API rate limit 约束，不宜过大）
MAX_WORKERS = 3


def _normalize_answer(value):
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _normalize_for_compare(value):
    """Normalize answer for comparison: lowercase and strip punctuation."""
    if value is None:
        return ""
    text = " ".join(str(value).strip().split()).lower()
    text = text.rstrip(".")
    return text


def load_questions(file_path: Path):
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def load_standard_map(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        standard_data = json.load(f)
    standard_answers = standard_data.get("answers", [])
    return {str(item.get("id")): _normalize_answer(item.get("answer")) for item in standard_answers}


def score_results(results, standard_map):
    matches = 0
    for item in results:
        qid = str(item.get("id"))
        if qid not in standard_map:
            continue
        our = _normalize_for_compare(item.get("answer"))
        ref = _normalize_for_compare(standard_map[qid])
        if our == ref:
            matches += 1
    return matches


def save_results(results, score):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"{score}分_{timestamp}_answers.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return output_file


def write_summary(summary):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return SUMMARY_FILE


def _answer_one_question(server, q, idx):
    """处理单个题目，返回 (idx, {"id": ..., "answer": ...}, elapsed_seconds)。"""
    qid = q.get("id", idx)
    text = q.get("question", "")
    t0 = time.time()
    try:
        result = server._multi_hop_reasoning(text, use_mcp=True)
        answer = result.get("answer", "Unknown")
    except Exception as e:
        answer = "Unknown"
        print(f"  [ERROR] Q{qid}: {e}", flush=True)
    elapsed = time.time() - t0
    return idx, {"id": qid, "answer": answer}, elapsed


# 用于打印的锁，避免多线程输出混乱
_print_lock = threading.Lock()


def _run_stage_concurrent(server, questions, start_idx, end_idx):
    """并发处理 [start_idx, end_idx) 范围内的题目。返回按原始顺序排列的结果列表。"""
    batch = [(questions[i], i) for i in range(start_idx, end_idx)]
    if not batch:
        return []

    results_dict = {}
    total = len(batch)
    done_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_answer_one_question, server, q, idx): idx
            for q, idx in batch
        }
        for future in as_completed(futures):
            idx, result, elapsed = future.result()
            results_dict[idx] = result
            done_count += 1
            qid = result["id"]
            ans_preview = result["answer"][:40]
            with _print_lock:
                print(
                    f"  Q{qid} done ({done_count}/{total}) "
                    f"{elapsed:.1f}s => {ans_preview}",
                    flush=True,
                )

    # 按原始索引顺序返回
    return [results_dict[i] for i in range(start_idx, end_idx)]


def main():
    if not QUESTIONS_FILE.exists():
        print(f"Questions file not found: {QUESTIONS_FILE}")
        return
    if not STANDARD_FILE.exists():
        print(f"Standard answers not found: {STANDARD_FILE}")
        return

    questions = load_questions(QUESTIONS_FILE)
    standard_map = load_standard_map(STANDARD_FILE)
    server = EnhancedMultiHopAPIServer()

    stages = [(5, 2), (20, 8), (100, None)]
    results = []
    current_count = 0
    last_output = None

    for stage_total, threshold in stages:
        target = min(stage_total, len(questions))
        if current_count < target:
            stage_t0 = time.time()
            print(f"\n=== Stage {stage_total} ({current_count}→{target}) workers={MAX_WORKERS} ===", flush=True)

            stage_results = _run_stage_concurrent(server, questions, current_count, target)
            results.extend(stage_results)

            stage_elapsed = time.time() - stage_t0
            print(f"=== Stage {stage_total} done in {stage_elapsed:.1f}s ===", flush=True)

        current_count = target
        score = score_results(results, standard_map)
        last_output = save_results(results, score)

        summary = {
            "stage_total": stage_total,
            "ran": current_count,
            "score": score,
            "threshold": threshold,
            "passed": True if threshold is None else score >= threshold,
            "output_file": str(last_output)
        }
        write_summary(summary)
        print(f"Stage {stage_total} Score: {score}")
        print(f"Output: {last_output}")
        print(f"Summary: {SUMMARY_FILE}")

        if threshold is not None and score < threshold:
            print("Stage failed, stopping early.")
            break


if __name__ == "__main__":
    main()
