#!/usr/bin/env python3
"""
Progressive evaluation with early stopping.
Stages: 5 -> 20 -> 100 with thresholds 2 and 8.
Outputs JSON array with id + answer only.
"""

import json
import sys
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


def _normalize_answer(value):
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def _normalize_for_compare(value):
    """Normalize answer for comparison: lowercase and strip punctuation."""
    if value is None:
        return ""
    text = " ".join(str(value).strip().split()).lower()
    # Strip trailing periods and common punctuation
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
        for idx in range(current_count, target):
            q = questions[idx]
            qid = q.get("id", idx)
            text = q.get("question", "")
            result = server._multi_hop_reasoning(text, use_mcp=True)
            answer = result.get("answer", "Unknown")
            results.append({"id": qid, "answer": answer})
            if (idx + 1) % 5 == 0:
                print(f"Progress: {idx + 1}/{target}")

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
