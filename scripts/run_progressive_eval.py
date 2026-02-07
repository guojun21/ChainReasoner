#!/usr/bin/env python3
"""
Progressive evaluation with early stopping and regression detection.
Stages: 5 -> 10 -> 20 -> 40 -> 60 -> 80 -> 100 with ~40% threshold at each gate.
Outputs JSONL (one JSON object per line) — competition-required format.

== 只进不退机制 ==
评测前自动加载历史最佳答案（outputs/ 里分数最高的文件）作为 baseline。
每完成一道题，如果该题在 baseline 中答对了但本轮答错了，立即终止评测并报告退步原因。
这确保代码修改不会导致以前答对的题突然答错（退步）。

[PERF-OPTIMIZED 2026-02-06] 使用 ThreadPoolExecutor 并发处理题目，大幅提升速度。
  - 每阶段内的题目并发执行（默认 3 个 worker），受限于 Brave API rate limit
  - 每题完成立即打印耗时，便于观察
  - 注意：如果你（Cursor agent）正在修改 api_server.py 或 constrained_search.py，
    本脚本的并发逻辑不受影响，因为每个线程共享同一个 server 实例。

== 禁止修改评分逻辑 ==
_normalize_for_compare() 和 score_results() 是评分核心函数，
禁止添加缩写归一化、包含关系匹配、模糊匹配等放松评分标准的逻辑。
最终比赛用比赛方的评分系统，不是本地的。
"""

import argparse
import io
import json
import re
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
LOG_DIR = BASE_DIR / "logs"

# 并发 worker 数（受 Brave API rate limit 约束，不宜过大）
MAX_WORKERS = 3


# --------------- Tee: 同时输出到终端和日志文件 ---------------

class TeeWriter:
    """将 stdout 同时写到终端和日志 txt 文件。"""

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


def _normalize_answer(value):
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _normalize_for_compare(value):
    """Normalize answer for comparison: lowercase and strip punctuation.
    禁止修改此函数！不得添加缩写归一化、包含匹配等放松评分的逻辑。
    """
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
    """禁止修改此函数！评分必须保持精确文本匹配。"""
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


# --------------- 只进不退：历史最佳 baseline ---------------

def _load_results_file(file_path: Path) -> list:
    """Load answer results from either JSON array (.json) or JSONL (.jsonl) file."""
    with open(file_path, "r", encoding="utf-8") as fh:
        content = fh.read().strip()
    if not content:
        return []
    # Try JSON array first
    if content.startswith("["):
        return json.loads(content)
    # Otherwise treat as JSONL (one JSON object per line)
    results = []
    for line in content.split("\n"):
        line = line.strip()
        if line:
            results.append(json.loads(line))
    return results


def _load_best_baseline(standard_map):
    """扫描 outputs/ 目录，找到分数最高的答案文件，返回其中答对的题目集合。
    支持 .json (旧格式) 和 .jsonl (新格式) 两种文件。
    返回: (baseline_correct_ids: set, baseline_file: str, baseline_score: int)
    """
    if not OUTPUT_DIR.exists():
        return set(), "", 0

    best_score = -1
    best_file = None

    for f in OUTPUT_DIR.iterdir():
        # Match both old (.json) and new (.jsonl) naming
        m = re.match(r"(\d+)分_\d+_answers\.jsonl?$", f.name)
        if m:
            file_score = int(m.group(1))
            if file_score > best_score:
                best_score = file_score
                best_file = f

    if not best_file or best_score <= 0:
        return set(), "", 0

    try:
        baseline_results = _load_results_file(best_file)
    except Exception:
        return set(), "", 0

    correct_ids = set()
    for item in baseline_results:
        qid = str(item.get("id"))
        if qid not in standard_map:
            continue
        our = _normalize_for_compare(item.get("answer"))
        ref = _normalize_for_compare(standard_map[qid])
        if our == ref:
            correct_ids.add(qid)

    return correct_ids, str(best_file), best_score


def _check_regression(results, standard_map, baseline_correct_ids):
    """检查本轮结果是否有退步（baseline 答对但本轮答错的题）。
    返回: list of {qid, baseline_answer(correct), current_answer, standard_answer}
    """
    regressions = []
    for item in results:
        qid = str(item.get("id"))
        if qid not in baseline_correct_ids:
            continue  # baseline 也没答对，不算退步
        if qid not in standard_map:
            continue
        our = _normalize_for_compare(item.get("answer"))
        ref = _normalize_for_compare(standard_map[qid])
        if our != ref:
            regressions.append({
                "qid": qid,
                "current_answer": item.get("answer"),
                "standard_answer": standard_map[qid],
                "note": f"Q{qid} 退步！baseline 答对了，本轮答错"
            })
    return regressions


# --------------- /只进不退 ---------------


def save_results(results, score):
    """Save results as JSONL (one JSON object per line) — the format required by the competition."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"{score}分_{timestamp}_answers.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps({"id": item["id"], "answer": item["answer"]}, ensure_ascii=False) + "\n")
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
            ans_preview = result["answer"][:50]
            pct = int(done_count / total * 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            with _print_lock:
                print(
                    f"  [{bar}] {pct:3d}% | Q{qid:>3} ({done_count}/{total}) "
                    f"{elapsed:.0f}s => {ans_preview}",
                    flush=True,
                )

    # 按原始索引顺序返回
    return [results_dict[i] for i in range(start_idx, end_idx)]


def parse_args():
    parser = argparse.ArgumentParser(description="Progressive evaluation for ChainReasoner")
    parser.add_argument(
        "--no-stage", "--full",
        action="store_true",
        dest="no_stage",
        help="跳过分阶段门槛检查，直接跑完全部 100 题（退步检测仍生效）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 设置 Tee: 终端输出同时写到 logs/eval_{timestamp}.txt
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"eval_{timestamp_str}.txt"
    tee = TeeWriter(log_file)
    sys.stdout = tee
    print(f"[LOG] 日志文件: {log_file}", flush=True)

    if not QUESTIONS_FILE.exists():
        print(f"Questions file not found: {QUESTIONS_FILE}")
        return
    if not STANDARD_FILE.exists():
        print(f"Standard answers not found: {STANDARD_FILE}")
        return

    questions = load_questions(QUESTIONS_FILE)
    standard_map = load_standard_map(STANDARD_FILE)
    server = EnhancedMultiHopAPIServer()

    # 加载历史最佳 baseline
    baseline_correct_ids, baseline_file, baseline_score = _load_best_baseline(standard_map)
    if baseline_correct_ids:
        print(f"\n[Baseline] 历史最佳: {baseline_score}分 ({baseline_file})", flush=True)
        print(f"[Baseline] 答对的题目 ({len(baseline_correct_ids)}题): {sorted(baseline_correct_ids, key=lambda x: int(x))}", flush=True)
        print(f"[Baseline] 本轮如果这些题答错，将立即终止评测\n", flush=True)
    else:
        print("\n[Baseline] 无历史最佳记录，跳过退步检测\n", flush=True)

    # --no-stage / --full: 跳过分阶段，直接跑全部 100 题（一个大 stage）
    if args.no_stage:
        stages = [
            (len(questions), None),  # 全部跑完，无门槛
        ]
        print("[MODE] --no-stage: 不分阶段，直接跑完全部 %d 题\n" % len(questions), flush=True)
    else:
        stages = [
            (5,   1),     # Stage 5:   至少 1/5  = 20%  — 快速冒烟测试
            (10,  3),     # Stage 10:  至少 3/10 = 30%  — 基本能力
            (20,  7),     # Stage 20:  至少 7/20 = 35%  — 稳定性验证
            (40,  16),    # Stage 40:  至少 16/40 = 40% — 中等要求
            (60,  27),    # Stage 60:  至少 27/60 = 45% — 高于平均
            (80,  40),    # Stage 80:  至少 40/80 = 50% — 半数正确
            (100, None),  # Stage 100: 跑完所有（退步检测仍生效）
        ]

    results = []
    current_count = 0
    last_output = None

    for stage_total, threshold in stages:
        target = min(stage_total, len(questions))
        if current_count < target:
            stage_t0 = time.time()
            new_count = target - current_count
            print(f"\n{'='*60}", flush=True)
            if args.no_stage:
                print(f"  FULL RUN | 题目 Q{current_count}~Q{target-1} | 共 {new_count} 题 | workers={MAX_WORKERS}", flush=True)
            else:
                print(f"  STAGE {stage_total} | 题目 Q{current_count}~Q{target-1} | 新增 {new_count} 题 | workers={MAX_WORKERS}", flush=True)
            print(f"{'='*60}", flush=True)

            stage_results = _run_stage_concurrent(server, questions, current_count, target)
            results.extend(stage_results)

            stage_elapsed = time.time() - stage_t0
            print(f"{'='*60}", flush=True)
            if args.no_stage:
                print(f"  FULL RUN 完成 | 耗时 {stage_elapsed:.0f}s ({stage_elapsed/60:.1f}min)", flush=True)
            else:
                print(f"  STAGE {stage_total} 完成 | 耗时 {stage_elapsed:.0f}s ({stage_elapsed/60:.1f}min)", flush=True)
            print(f"{'='*60}", flush=True)

        current_count = target
        score = score_results(results, standard_map)

        # ---- 只进不退：检查退步（--no-stage 模式下也生效）----
        if baseline_correct_ids:
            regressions = _check_regression(results, standard_map, baseline_correct_ids)
            if regressions:
                # 还是先保存结果，方便分析
                last_output = save_results(results, score)

                print(f"\n{'='*60}", flush=True)
                print(f"退步检测失败！本轮得 {score} 分，但有 {len(regressions)} 道题退步了", flush=True)
                print(f"   Baseline: {baseline_score}分 ({baseline_file})", flush=True)
                print(f"{'='*60}", flush=True)
                for reg in regressions:
                    print(f"  X Q{reg['qid']}: baseline 答对 \"{reg['standard_answer']}\"", flush=True)
                    print(f"     本轮答了 \"{reg['current_answer']}\" — 退步！", flush=True)
                print(f"{'='*60}", flush=True)
                print(f"\n结论：代码修改导致退步，必须回滚或修复以上退步的题目。", flush=True)
                print(f"本轮新答对但 baseline 没对的题不能以牺牲已对的题为代价。\n", flush=True)

                # 写 summary 标记失败和退步原因
                regression_detail = [
                    f"Q{r['qid']}: baseline正确=\"{r['standard_answer']}\" 本轮错误=\"{r['current_answer']}\""
                    for r in regressions
                ]
                summary = {
                    "stage_total": stage_total,
                    "ran": current_count,
                    "score": score,
                    "threshold": threshold,
                    "passed": False,
                    "output_file": str(last_output),
                    "regression_detected": True,
                    "regression_count": len(regressions),
                    "regression_detail": regression_detail,
                    "baseline_score": baseline_score,
                    "baseline_file": baseline_file,
                    "failure_reason": f"退步{len(regressions)}题: " + "; ".join(
                        f"Q{r['qid']}(\"{r['standard_answer']}\"→\"{r['current_answer']}\")"
                        for r in regressions
                    )
                }
                write_summary(summary)
                print(f"Output: {last_output}")
                print(f"Summary: {SUMMARY_FILE}")
                # --no-stage 模式下退步也不中断，只记录警告，继续跑完
                if not args.no_stage:
                    return  # 分阶段模式下立即终止
                else:
                    print("[--no-stage] 退步已记录，继续跑完剩余题目\n", flush=True)
        # ---- /只进不退 ----

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

        if args.no_stage:
            print(f"\nFinal Score: {score}/{current_count}")
        else:
            print(f"Stage {stage_total} Score: {score}")
        print(f"Output: {last_output}")
        print(f"Summary: {SUMMARY_FILE}")

        if not args.no_stage and threshold is not None and score < threshold:
            print("Stage failed, stopping early.")
            break

    # --no-stage: 全部跑完后也生成 submit 文件
    if args.no_stage and results:
        submit_file = OUTPUT_DIR / f"{score}分_{datetime.now().strftime('%Y%m%d_%H%M%S')}_submit.jsonl"
        with open(submit_file, "w", encoding="utf-8") as f:
            for q in questions:
                qid = q.get("id")
                matched = next((r for r in results if r["id"] == qid), None)
                ans = matched["answer"] if matched else "Unknown"
                f.write(json.dumps({"id": qid, "answer": ans}, ensure_ascii=False) + "\n")
        print(f"\nSubmit file: {submit_file}")


if __name__ == "__main__":
    main()
