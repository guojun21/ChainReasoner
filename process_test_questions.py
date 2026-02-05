#!/usr/bin/env python3
"""
Script to process test questions and generate answers using the MultiHop Agent API.
"""

import json
import requests
import time
from pathlib import Path

# Configuration
API_URL = "http://localhost:5000/api/v1/answer"
API_TOKEN = "multihop_agent_token_2024"
INPUT_FILE = "question.json"
OUTPUT_FILE = "submission.json"

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
    save_results(results, OUTPUT_FILE)
    
    print("\n" + "="*70)
    print("Processing complete!")
    print("="*70)

if __name__ == "__main__":
    main()
