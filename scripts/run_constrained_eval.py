#!/usr/bin/env python3
"""
Run constrained search evaluation and export answers.
Outputs JSON with only id and answer, filename prefixed by score.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from apps.api.api_server import EnhancedMultiHopAPIServer
QUESTIONS_FILE = BASE_DIR / "data" / "qa" / "question.json"
STANDARD_FILE = BASE_DIR / "data" / "qa" / "the_standard_answers.json"
OUTPUT_DIR = BASE_DIR / "data" / "qa" / "outputs"


def _normalize_answer(value):
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def load_questions(file_path: Path):
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def compare_with_standard(results, standard_file: Path) -> int:
    if not standard_file.exists():
        return 0
    with open(standard_file, "r", encoding="utf-8") as f:
        standard_data = json.load(f)
    standard_answers = standard_data.get("answers", [])
    standard_map = {str(item.get("id")): _normalize_answer(item.get("answer")) for item in standard_answers}
    result_map = {str(item.get("id")): _normalize_answer(item.get("answer")) for item in results}

    matches = 0
    for qid, expected in standard_map.items():
        if result_map.get(qid) == expected:
            matches += 1
    return matches


def main():
    if not QUESTIONS_FILE.exists():
        print(f"Questions file not found: {QUESTIONS_FILE}")
        return

    questions = load_questions(QUESTIONS_FILE)
    if not questions:
        print("No questions loaded.")
        return

    server = EnhancedMultiHopAPIServer()

    results = []
    for i, q in enumerate(questions, 1):
        qid = q.get("id", i - 1)
        text = q.get("question", "")
        result = server._multi_hop_reasoning(text, use_mcp=True)
        answer = result.get("answer", "Unknown")
        results.append({"id": qid, "answer": answer})
        if i % 10 == 0:
            print(f"Progress: {i}/{len(questions)}")

    score = compare_with_standard(results, STANDARD_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"{score}分_{timestamp}_answers.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Score: {score}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
