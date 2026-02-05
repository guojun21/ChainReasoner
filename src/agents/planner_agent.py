
"""
Planner Agent for multi-hop reasoning task decomposition.
Responsible for parsing complex questions and generating sub-task sequences.
"""

from typing import Any, Dict, List


class PlannerAgent:
    """Planner Agent that decomposes complex questions into sub-tasks."""
    
    def __init__(self):
        self.task_history = []
    
    def parse_question(self, question: str) -> Dict[str, Any]:
        """
        Parse the input question to identify key entities and relationships.
        
        Args:
            question (str): The input multi-hop reasoning question
            
        Returns:
            Dict[str, Any]: Parsed question structure with entities and relations
        """
        # Simple entity extraction (in real implementation, this would be more sophisticated)
        # For demo purposes, we'll use basic keyword matching
        parsed = {
            "original_question": question,
            "entities": self._extract_entities(question),
            "relations": self._extract_relations(question),
            "question_type": self._classify_question_type(question)
        }
        return parsed
    
    def _extract_entities(self, question: str) -> List[str]:
        """Extract key entities from the question."""
        # In a real implementation, this would use NER or other techniques
        # For now, return empty list as placeholder
        return []
    
    def _extract_relations(self, question: str) -> List[str]:
        """Extract relations from the question."""
        # Placeholder implementation
        return []
    
    def _classify_question_type(self, question: str) -> str:
        """Classify the type of multi-hop question."""
        # Placeholder implementation
        if "who" in question.lower():
            return "entity_identification"
        elif "what" in question.lower():
            return "fact_retrieval"
        elif "how" in question.lower() or "why" in question.lower():
            return "causal_reasoning"
        else:
            return "general"
    
    def decompose_task(self, parsed_question: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Decompose the parsed question into a sequence of sub-tasks.
        
        Args:
            parsed_question (Dict[str, Any]): Parsed question structure
            
        Returns:
            List[Dict[str, Any]]: List of sub-tasks to execute
        """
        original_question = parsed_question["original_question"]
        question_type = parsed_question["question_type"]
        
        # Generate sub-tasks based on question type
        if question_type == "entity_identification":
            sub_tasks = [
                {
                    "task_id": 1,
                    "task_type": "entity_search",
                    "query": original_question,
                    "description": "Find the main entity mentioned in the question"
                },
                {
                    "task_id": 2, 
                    "task_type": "relation_search",
                    "query": f"What are the key facts about the entity found in task 1?",
                    "description": "Find relationships and facts about the identified entity",
                    "depends_on": [1]
                }
            ]
        elif question_type == "fact_retrieval":
            sub_tasks = [
                {
                    "task_id": 1,
                    "task_type": "initial_search", 
                    "query": original_question,
                    "description": "Initial search for the main fact"
                },
                {
                    "task_id": 2,
                    "task_type": "verification_search",
                    "query": f"Verify the fact found in task 1 from multiple sources",
                    "description": "Cross-verify the retrieved fact",
                    "depends_on": [1]
                }
            ]
        else:
            # General case - single search task
            sub_tasks = [
                {
                    "task_id": 1,
                    "task_type": "comprehensive_search",
                    "query": original_question,
                    "description": "Comprehensive search for the question"
                }
            ]
        
        self.task_history.append({
            "question": original_question,
            "sub_tasks": sub_tasks
        })
        
        return sub_tasks
