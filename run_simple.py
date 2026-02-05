#!/usr/bin/env python3
"""
Simplified MultiHop Agent runner using LLM API.
This version works without Neo4j and provides basic question answering.
"""



import yaml
import requests
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


class SimpleLLMAgent:
    """Simple agent using LLM API for question answering."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.base_model = self.config.get("base_model", {})
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM API."""
        api_url = self.base_model.get("api_url")
        api_key = self.base_model.get("api_key")
        model_id = self.base_model.get("model_id")
        temperature = self.base_model.get("temperature", 0.7)
        max_tokens = self.base_model.get("max_tokens", 2048)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {str(e)}"
    
    def process_question(self, question: str, question_id: str = "unknown") -> Dict[str, Any]:
        """Process a single question using LLM."""
        print(f"\n{'='*60}")
        print(f"Processing Question: {question}")
        print(f"{'='*60}")
        
        system_prompt = """You are a helpful AI assistant that answers questions accurately and concisely.
Provide well-reasoned answers with clear explanations when needed."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        start_time = datetime.now()
        answer = self._call_llm(messages)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\nAnswer: {answer}")
        print(f"Processing Time: {processing_time:.2f}s")
        
        return {
            "question_id": question_id,
            "question": question,
            "answer": answer,
            "answer_type": "general",
            "confidence": 0.85,
            "evidence": [],
            "reasoning_steps": [],
            "processing_time": round(processing_time, 2),
            "generated_at": datetime.now().isoformat()
        }
    
    def process_questions_batch(self, questions: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Process a batch of questions."""
        results = []
        for i, question_data in enumerate(questions, 1):
            print(f"\n{'#'*60}")
            print(f"Question {i}/{len(questions)}")
            print(f"{'#'*60}")
            
            question = question_data.get("question", "")
            question_id = question_data.get("id", f"Q{i}")
            
            result = self.process_question(question, question_id)
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str = "results.json"):
        """Save results to JSON file."""
        output_data = {
            "submission_time": datetime.now().isoformat(),
            "total_questions": len(results),
            "answers": results,
            "metadata": {
                "agent_version": "SimpleLLMAgent v1.0",
                "model": self.base_model.get("model_id", "unknown")
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"Results saved to {output_file}")
        print(f"{'='*60}")


def load_sample_questions():
    """Load sample questions for testing."""
    return [
        {"id": "Q1", "question": "Where did Albert Einstein work?"},
        {"id": "Q2", "question": "What did Marie Curie discover?"},
        {"id": "Q3", "question": "Who developed the Theory of Relativity?"},
        {"id": "Q4", "question": "What is the capital of France?"},
        {"id": "Q5", "question": "Explain the concept of photosynthesis."}
    ]


def load_questions_from_file(file_path: str = "questions.json"):
    """Load questions from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("questions", [])
    except FileNotFoundError:
        print(f"File {file_path} not found. Using sample questions.")
        return load_sample_questions()
    except Exception as e:
        print(f"Error loading questions: {e}. Using sample questions.")
        return load_sample_questions()


def main():
    """Main function to run the agent."""
    print("\n" + "="*60)
    print("Simple LLM Agent - MultiHop Question Answering")
    print("="*60)
    
    agent = SimpleLLMAgent()
    
    print(f"\nModel: {agent.base_model.get('model_id', 'unknown')}")
    print(f"API URL: {agent.base_model.get('api_url', 'unknown')}")
    
    questions = load_questions_from_file()
    print(f"\nLoaded {len(questions)} questions")
    
    results = agent.process_questions_batch(questions)
    agent.save_results(results)
    
    print("\n" + "="*60)
    print("All questions processed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()