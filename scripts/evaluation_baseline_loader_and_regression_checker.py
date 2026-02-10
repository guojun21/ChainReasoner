"""Baseline loading and regression detection for progressive evaluation.

Why: The "only-forward" mechanism prevents code changes from breaking
previously correct answers.  Before each evaluation run, we load the
best historical score file, identify which questions were answered
correctly, and terminate early if any of those regress.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_evaluation_results_from_json_or_jsonl_file(file_path: Path) -> list:
    """Load answers from .json (array) or .jsonl (one-per-line) format."""
    content = file_path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.split("\n") if line.strip()]


def scan_output_directory_for_highest_scoring_baseline(output_dir: Path, standard_map: Dict[str, str],
                       normalize_fn=None) -> Tuple[Set[str], str, int]:
    """Scan outputs/ for the highest-scoring file and return its correct IDs.

    Returns (correct_question_ids, filename, score).
    """
    if not output_dir.exists():
        return set(), "", 0

    best_score, best_file = -1, None
    for file_path in output_dir.iterdir():
        match = re.match(r"(\d+)分_\d+_answers\.jsonl?$", file_path.name)
        if match:
            score = int(match.group(1))
            if score > best_score:
                best_score = score
                best_file = file_path

    if not best_file or best_score <= 0:
        return set(), "", 0

    try:
        baseline_results = load_evaluation_results_from_json_or_jsonl_file(best_file)
    except Exception:
        return set(), "", 0

    correct_ids = set()
    for item in baseline_results:
        question_id = str(item.get("id"))
        if question_id not in standard_map:
            continue
        our_answer = normalize_fn(item.get("answer")) if normalize_fn else str(item.get("answer", "")).lower()
        reference = normalize_fn(standard_map[question_id]) if normalize_fn else standard_map[question_id].lower()
        if our_answer == reference:
            correct_ids.add(question_id)

    return correct_ids, str(best_file), best_score


def detect_answer_regressions_against_baseline(results: list, standard_map: Dict[str, str],
                     baseline_correct_ids: Set[str],
                     normalize_fn=None) -> List[dict]:
    """Find questions that baseline got right but this run got wrong.

    Returns list of {qid, current_answer, standard_answer, note}.
    """
    regressions = []
    for item in results:
        question_id = str(item.get("id"))
        if question_id not in baseline_correct_ids or question_id not in standard_map:
            continue
        our_answer = normalize_fn(item.get("answer")) if normalize_fn else str(item.get("answer", "")).lower()
        reference = normalize_fn(standard_map[question_id]) if normalize_fn else standard_map[question_id].lower()
        if our_answer != reference:
            regressions.append({
                "qid": question_id,
                "current_answer": item.get("answer"),
                "standard_answer": standard_map[question_id],
                "note": f"Q{question_id} regressed — baseline correct, this run wrong",
            })
    return regressions
