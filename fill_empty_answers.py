#!/usr/bin/env python3
"""
Script to fill empty answers in answer02.json by calling the API service.
"""

import json
import requests
import time
from pathlib import Path

# Configuration
API_URL = "http://localhost:5000/api/v1/answer"
API_TOKEN = "multihop_agent_token_2024"
ANSWER_FILE = "answer02.json"
QUESTION_FILE = "question.json"

def load_answers(file_path):
    """Load answers from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    # Handle both JSON array and line-separated JSON
    if data.strip().startswith('['):
        return json.loads(data)
    else:
        answers = []
        for line in data.strip().split('\n'):
            if line.strip():
                answers.append(json.loads(line))
        return answers

def load_questions(file_path):
    """Load questions from JSON file."""
    questions = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                questions[q.get('id')] = q.get('question', '')
    return questions

def call_api(question, use_mcp=True, max_retries=3):
    """Call the MultiHop Agent API to answer a question with retry mechanism."""
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    data = {
        "question": question,
        "use_mcp": use_mcp
    }
    
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}...")
            response = requests.post(API_URL, headers=headers, json=data, timeout=180)
            response.raise_for_status()
            result = response.json()
            return result.get("answer", "")
        except Exception as e:
            print(f"  Error calling API: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in 5 seconds...")
                time.sleep(5)
            else:
                return f"Error: {str(e)}"

def fill_empty_answers():
    """Fill empty answers in answer02.json."""
    print("=" * 70)
    print("Filling Empty Answers in answer02.json")
    print("=" * 70)
    
    # Check if files exist
    if not Path(ANSWER_FILE).exists():
        print(f"Error: Answer file {ANSWER_FILE} not found")
        return
    
    if not Path(QUESTION_FILE).exists():
        print(f"Error: Question file {QUESTION_FILE} not found")
        return
    
    # Load data
    answers = load_answers(ANSWER_FILE)
    questions = load_questions(QUESTION_FILE)
    
    print(f"Loaded {len(answers)} answers from {ANSWER_FILE}")
    print(f"Loaded {len(questions)} questions from {QUESTION_FILE}")
    
    # Find empty answers
    empty_answers = []
    for ans in answers:
        if not ans.get('answer', '').strip() or 'Error:' in ans.get('answer', ''):
            empty_answers.append(ans)
    
    print(f"Found {len(empty_answers)} empty or error answers to fill")
    print("=" * 70)
    
    if not empty_answers:
        print("No empty answers found. Exiting.")
        return
    
    # Fill empty answers
    for i, ans in enumerate(empty_answers, 1):
        answer_id = ans.get('id')
        question = questions.get(answer_id, '')
        
        if not question:
            print(f"[{i}/{len(empty_answers)}] Skipping ID {answer_id}: No question found")
            continue
        
        print(f"\n[{i}/{len(empty_answers)}] Processing ID {answer_id}")
        print(f"Question: {question[:100]}...")
        
        # Call API to get answer
        answer = call_api(question, use_mcp=True)
        
        # Update the answer
        for a in answers:
            if a.get('id') == answer_id:
                a['answer'] = answer
                break
        
        print(f"Answer: {answer[:100]}...")
        
        # Save progress after each answer
        with open(ANSWER_FILE, 'w', encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False, indent=2)
        
        print(f"Saved progress to {ANSWER_FILE}")
        
        # Add delay to avoid API rate limiting
        if i < len(empty_answers):
            print("Waiting 2 seconds before next request...")
            time.sleep(2)
    
    print("\n" + "=" * 70)
    print(f"Completed filling {len(empty_answers)} empty answers")
    print(f"Results saved to {ANSWER_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    fill_empty_answers()
