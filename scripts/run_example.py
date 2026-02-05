#!/usr/bin/env python3
"""
Example script to run the MultiHop Agent on sample questions.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]


def load_sample_questions():
    """Load sample questions for testing."""
    # In real usage, this would load from data/qa/question.json
    return [
        {"id": "Q1", "question": "Where did Albert Einstein work?"},
        {"id": "Q2", "question": "What did Marie Curie discover?"},
        {"id": "Q3", "question": "Who developed the Theory of Relativity?"}
    ]


def main():
    # This is a placeholder - the actual implementation would import from src
    print("This is an example script template.")
    print("To run the actual agent, ensure the project structure is correct and Python path is set.")
    
    # Sample documents for demonstration
    sample_documents = [
        "Albert Einstein was born in Ulm, Germany and later worked at Princeton University.",
        "Marie Curie was born in Warsaw, Poland and discovered Radium while working at Sorbonne University.",
        "The Theory of Relativity was developed by Albert Einstein in the early 20th century."
    ]
    
    print(f"Sample documents: {len(sample_documents)}")
    questions = load_sample_questions()
    print(f"Sample questions: {len(questions)}")
    
    # Results would be generated here in actual implementation
    results = []
    for q in questions:
        results.append({
            "question_id": q["id"],
            "answer": "Sample answer for demonstration",
            "confidence": 0.85,
            "processing_time": 1.2
        })
    
    # Save results
    output_file = BASE_DIR / "results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
