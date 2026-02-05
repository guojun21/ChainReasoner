#!/usr/bin/env python3
"""
Batch Test Script for MultiHop Agent API
Processes 100 questions from question.json and generates answers file.
"""



import json
import requests
import time
from pathlib import Path
from typing import List, Dict, Any


class BatchTester:
    """Batch tester for MultiHop Agent API."""
    
    def __init__(self, api_url: str = "http://127.0.0.1:5000/api/v1/answer", api_token: str = "multihop_agent_token_2024"):
        self.api_url = api_url
        self.api_token = api_token
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
    def load_questions(self, file_path: str = "question.json") -> List[Dict[str, Any]]:
        """Load questions from JSON file (JSON Lines format)."""
        questions = []
        with open(file_path, 'r', encoding='utf-8') as f:
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
            return result.get("answer", "")
        except Exception as e:
            print(f"  Error: {e}")
            return f"Error: {str(e)}"
    
    def process_batch(self, questions: List[Dict[str, Any]], output_file: str = "answers.json"):
        """Process batch of questions."""
        print(f"\n{'='*70}")
        print(f"Batch Test - MultiHop Agent API")
        print(f"{'='*70}")
        print(f"\nTotal questions: {len(questions)}")
        print(f"API URL: {self.api_url}")
        print(f"Output file: {output_file}")
        
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
        
        self.save_results(results, output_file)
        
        print(f"\n{'='*70}")
        print(f"Completed! Processed {len(results)} questions")
        print(f"Results saved to {output_file}")
        print(f"{'='*70}")
    
    def save_results(self, results: List[Dict[str, str]], output_file: str):
        """Save results to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                json_line = json.dumps(result, ensure_ascii=False)
                f.write(json_line + '\n')
        
        print(f"\nFormat: Each line is a JSON object with 'id' and 'answer' fields")


def main():
    """Main function."""
    print("\n" + "="*70)
    print("MultiHop Agent Batch Test Script")
    print("="*70)
    
    tester = BatchTester()
    
    try:
        questions = tester.load_questions("question.json")
        print(f"\nLoaded {len(questions)} questions from question.json")
        
        if len(questions) == 0:
            print("No questions found in question.json")
            return
        
        tester.process_batch(questions, "answers.json")
        
    except FileNotFoundError:
        print("\nError: question.json file not found!")
        print("Please ensure question.json exists in the current directory.")
    except json.JSONDecodeError:
        print("\nError: Invalid JSON format in question.json")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()