#!/usr/bin/env python3
"""
Script to process test questions and generate answers using the MultiHop Agent API.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parents[1]

# Configuration
API_URL = "http://localhost:5000/api/v1/answer"
API_TOKEN = "multihop_agent_token_2024"
INPUT_FILE = BASE_DIR / "data" / "qa" / "question.json"
OUTPUT_DIR = BASE_DIR / "data" / "qa" / "outputs"
STANDARD_ANSWERS_FILE = BASE_DIR / "data" / "qa" / "the_standard_answers.json"

def load_questions(file_path):
    """Load questions from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    questions = []
    for line in data.strip().split('\n'):
        if line.strip():
            questions.append(json.loads(line))
    
    return questions

def call_api(question, use_mcp=True):
    """Call the MultiHop Agent API to answer a question."""
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    data = {
        "question": question,
        "use_mcp": use_mcp
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("answer", "")
    except Exception as e:
        print(f"Error calling API: {e}")
        return f"Error: {str(e)}"

def process_questions(questions):
    """Process all questions and generate answers."""
    results = []
    
    print(f"Processing {len(questions)} questions...")
    print("="*70)
    
    for i, q in enumerate(questions, 1):
        question_id = q.get("id", i - 1)
        question_text = q.get("question", "")
        
        print(f"\n[{i}/{len(questions)}] Processing question {question_id}")
        print(f"Question: {question_text[:100]}...")
        
        answer = call_api(question_text, use_mcp=True)
        
        result = {
            "id": question_id,
            "answer": answer
        }
        results.append(result)
        
        print(f"Answer: {answer[:100]}...")
        
        if i < len(questions):
            time.sleep(2)
    
    print("\n" + "="*70)
    print(f"Completed processing {len(questions)} questions")
    
    return results

def save_results(results, output_file):
    """Save results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_file}")

def _normalize_answer(value):
    """Normalize answer for comparison."""
    if value is None:
        return ""
    return " ".join(str(value).strip().split())

def compare_with_standard(results, standard_file):
    """Compare generated answers with standard answers."""
    if not Path(standard_file).exists():
        print(f"Standard answers not found: {standard_file}")
        return 0, 0, 0
    
    with open(standard_file, 'r', encoding='utf-8') as f:
        standard_data = json.load(f)
    
    standard_answers = standard_data.get("answers", [])
    standard_map = {str(item.get("id")): _normalize_answer(item.get("answer")) for item in standard_answers}
    result_map = {str(item.get("id")): _normalize_answer(item.get("answer")) for item in results}
    
    matches = 0
    mismatches = []
    missing = []
    
    for qid, expected in standard_map.items():
        actual = result_map.get(qid)
        if actual is None:
            missing.append(qid)
            continue
        if actual == expected:
            matches += 1
        else:
            mismatches.append((qid, expected, actual))
    
    print("\n" + "="*70)
    print("Comparison with Standard Answers")
    print("="*70)
    print(f"Total Standard Answers: {len(standard_map)}")
    print(f"Matches: {matches}")
    print(f"Mismatches: {len(mismatches)}")
    print(f"Missing: {len(missing)}")
    
    if mismatches:
        print("\nMismatched Questions (id: expected -> actual):")
        for qid, expected, actual in mismatches[:20]:
            print(f"  - {qid}: {expected} -> {actual}")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")
    
    if missing:
        print("\nMissing Question IDs:")
        print(", ".join(missing[:20]))
        if len(missing) > 20:
            print(f"... and {len(missing) - 20} more")
    
    return matches, len(mismatches), len(missing)

def main():
    """Main function."""
    print("="*70)
    print("MultiHop Agent - Test Question Processor")
    print("="*70)
    
    if not Path(INPUT_FILE).exists():
        print(f"Error: Input file {INPUT_FILE} not found")
        return
    
    questions = load_questions(INPUT_FILE)
    print(f"Loaded {len(questions)} questions from {INPUT_FILE}")
    
    results = process_questions(questions)
    matches, mismatches, missing = compare_with_standard(results, STANDARD_ANSWERS_FILE)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"{matches}分_{timestamp}_answers.json"
    save_results(results, output_file)
    
    print("\n" + "="*70)
    print("Processing complete!")
    print("="*70)

if __name__ == "__main__":
    main()
