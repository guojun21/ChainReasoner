#!/usr/bin/env python3
"""
Test Script for MultiHop Agent API
Processes 100 questions from question.json and generates answers file in JSONL format.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS_PATH = BASE_DIR / "data" / "qa" / "question.json"
DEFAULT_OUTPUT_PATH = BASE_DIR / "answer02.json"
STANDARD_ANSWERS_PATH = BASE_DIR / "data" / "qa" / "the_standard_answers.json"


class TestScript:
    """Test script for MultiHop Agent API."""
    
    def __init__(self, api_url: str = "http://127.0.0.1:5000/api/v1/answer", api_token: str = "multihop_agent_token_2024"):
        self.api_url = api_url
        self.api_token = api_token
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
    def load_questions(self, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load questions from JSON file (JSON Lines format)."""
        questions = []
        questions_file = Path(file_path) if file_path else DEFAULT_QUESTIONS_PATH
        if not questions_file.is_absolute():
            questions_file = BASE_DIR / questions_file
        with open(questions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        question = json.loads(line)
                        questions.append(question)
                    except json.JSONDecodeError:
                        print(f"Warning: Failed to parse line: {line[:50]}...")
        return questions
    
    def call_api(self, question: str) -> str:
        """Call API to get answer."""
        payload = {"question": question}
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            answer = result.get("answer", "")
            
            return answer
        except Exception as e:
            print(f"  Error: {e}")
            return ""
    
    def process_batch(self, questions: List[Dict[str, Any]], output_file: Optional[str] = None):
        """Process batch of questions."""
        output_path = Path(output_file) if output_file else DEFAULT_OUTPUT_PATH
        if not output_path.is_absolute():
            output_path = BASE_DIR / output_path
        print(f"\n{'='*70}")
        print(f"Test Script - MultiHop Agent API")
        print(f"{'='*70}")
        print(f"\nTotal questions: {len(questions)}")
        print(f"API URL: {self.api_url}")
        print(f"Output file: {output_path}")
        
        results = []
        
        for i, question_data in enumerate(questions, 1):
            question_id = question_data.get("id", i - 1)
            question_text = question_data.get("question", "")
            
            print(f"\n[{i}/{len(questions)}] ID: {question_id}")
            print(f"Question: {question_text[:100]}...")
            
            start_time = time.time()
            answer = self.call_api(question_text)
            elapsed = time.time() - start_time
            
            print(f"Answer: {answer[:100]}...")
            print(f"Time: {elapsed:.2f}s")
            
            results.append({
                "id": question_id,
                "answer": answer
            })
            
            if i % 10 == 0:
                print(f"\nProgress: {i}/{len(questions)} questions completed")
        
        self.save_results(results, output_path)
        self.compare_with_standard(results)
        
        print(f"\n{'='*70}")
        print(f"Completed! Processed {len(results)} questions")
        print(f"Results saved to {output_path}")
        print(f"{'='*70}")
    
    def save_results(self, results: List[Dict[str, str]], output_file: Path):
        """Save results to JSONL file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                json_line = json.dumps(result, ensure_ascii=False)
                f.write(json_line + '\n')
        
        print(f"\nFormat: JSONL - Each line is a JSON object with 'id' and 'answer' fields")

    def _normalize_answer(self, value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).strip().split())

    def compare_with_standard(self, results: List[Dict[str, str]]):
        """Compare results with standard answers."""
        if not STANDARD_ANSWERS_PATH.exists():
            print(f"\nStandard answers not found: {STANDARD_ANSWERS_PATH}")
            return
        
        with open(STANDARD_ANSWERS_PATH, 'r', encoding='utf-8') as f:
            standard_data = json.load(f)
        
        standard_answers = standard_data.get("answers", [])
        standard_map = {str(item.get("id")): self._normalize_answer(item.get("answer")) for item in standard_answers}
        result_map = {str(item.get("id")): self._normalize_answer(item.get("answer")) for item in results}
        
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


def main():
    """Main function."""
    print("\n" + "="*70)
    print("MultiHop Agent Test Script")
    print("="*70)
    
    tester = TestScript()
    
    try:
        questions = tester.load_questions()
        print(f"\nLoaded {len(questions)} questions from data/qa/question.json")
        
        if len(questions) == 0:
            print("No questions found in data/qa/question.json")
            return
        
        tester.process_batch(questions, "answer02.json")
        
    except FileNotFoundError:
        print("\nError: data/qa/question.json file not found!")
        print("Please ensure data/qa/question.json exists in the project root.")
    except json.JSONDecodeError:
        print("\nError: Invalid JSON format in data/qa/question.json")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()
