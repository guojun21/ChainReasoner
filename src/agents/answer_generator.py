
"""
Answer Generator for producing final structured answers.
Formats results according to competition requirements.
"""

import json
from datetime import datetime
from typing import Any, Dict, List


class AnswerGenerator:
    """Generates final answers in the required structured format."""
    
    def __init__(self, output_format: str = "json"):
        self.output_format = output_format
        self.answer_templates = self._load_answer_templates()
    
    def _load_answer_templates(self) -> Dict[str, str]:
        """Load answer templates for different question types."""
        return {
            "entity_identification": {
                "question_id": "{question_id}",
                "answer": "{answer}",
                "answer_type": "entity",
                "confidence": "{confidence}",
                "evidence": "{evidence}",
                "reasoning_steps": "{reasoning_steps}"
            },
            "fact_retrieval": {
                "question_id": "{question_id}",
                "answer": "{answer}",
                "answer_type": "fact",
                "confidence": "{confidence}",
                "evidence": "{evidence}",
                "reasoning_steps": "{reasoning_steps}"
            },
            "causal_reasoning": {
                "question_id": "{question_id}",
                "answer": "{answer}",
                "answer_type": "causal",
                "confidence": "{confidence}",
                "evidence": "{evidence}",
                "reasoning_steps": "{reasoning_steps}"
            },
            "general": {
                "question_id": "{question_id}",
                "answer": "{answer}",
                "answer_type": "general",
                "confidence": "{confidence}",
                "evidence": "{evidence}",
                "reasoning_steps": "{reasoning_steps}"
            }
        }
    
    def generate_answer(self, 
                       question_id: str,
                       answer: str, 
                       question_type: str = "general",
                       confidence: float = 0.8,
                       evidence: List[str] = None,
                       reasoning_steps: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a structured answer based on the provided information."""
        evidence = evidence or []
        reasoning_steps = reasoning_steps or []
        
        # Select appropriate template
        template = self.answer_templates.get(question_type, self.answer_templates["general"])
        
        # Format the answer
        formatted_answer = {
            "question_id": question_id,
            "answer": answer,
            "answer_type": question_type,
            "confidence": round(confidence, 3),
            "evidence": evidence,
            "reasoning_steps": reasoning_steps,
            "generated_at": datetime.now().isoformat(),
            "validation_status": "validated" if confidence > 0.7 else "needs_review"
        }
        
        return formatted_answer
    
    def batch_generate_answers(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate answers for a batch of results."""
        answers = []
        for result in results:
            answer = self.generate_answer(
                question_id=result.get("question_id", "unknown"),
                answer=result.get("answer", ""),
                question_type=result.get("question_type", "general"),
                confidence=result.get("confidence", 0.8),
                evidence=result.get("evidence", []),
                reasoning_steps=result.get("reasoning_steps", [])
            )
            answers.append(answer)
        
        return answers
    
    def save_answers_to_file(self, answers: List[Dict[str, Any]], filepath: str):
        """Save answers to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(answers, f, indent=2, ensure_ascii=False)
        
        print(f"Answers saved to {filepath}")
    
    def format_for_competition(self, answers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format answers according to competition requirements."""
        # Tianchi competition format
        competition_format = {
            "submission_time": datetime.now().isoformat(),
            "total_questions": len(answers),
            "answers": answers,
            "metadata": {
                "agent_version": "MultiHopAgent v1.0",
                "compliance": "No AI search tools used",
                "retrieval_methods": ["BM25", "Contriever", "Bing API"]
            }
        }
        
        return competition_format
