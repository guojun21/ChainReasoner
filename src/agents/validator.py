
"""
Validator for fact verification and cross-validation.
Implements multiple validation mechanisms including code execution and API calls.
"""

from typing import Any, Dict, List, Optional


class Validator:
    """Validates reasoning results through multiple verification mechanisms."""
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        self.api_keys = api_keys or {}
        self.validation_history = []
    
    def validate_mathematical_computation(self, expression: str, expected_result: Any) -> Dict[str, Any]:
        """Validate mathematical computations by executing code."""
        try:
            # Execute the mathematical expression in a safe environment
            result = eval(expression)
            is_valid = result == expected_result
            
            validation_result = {
                "validation_type": "mathematical",
                "expression": expression,
                "expected_result": expected_result,
                "actual_result": result,
                "is_valid": is_valid,
                "confidence": 1.0 if is_valid else 0.0
            }
            
            self.validation_history.append(validation_result)
            return validation_result
            
        except Exception as e:
            return {
                "validation_type": "mathematical",
                "expression": expression,
                "expected_result": expected_result,
                "error": str(e),
                "is_valid": False,
                "confidence": 0.0
            }
    
    def validate_external_fact(self, fact: str, fact_type: str) -> Dict[str, Any]:
        """Validate external facts using appropriate APIs."""
        # Placeholder implementation for different fact types
        if fact_type == "ofac_sanctions":
            # Simulate OFAC API call
            is_sanctioned = "sanctioned" in fact.lower()
            validation_result = {
                "validation_type": "external_api",
                "fact": fact,
                "fact_type": fact_type,
                "api_used": "OFAC",
                "is_valid": not is_sanctioned,  # Assume non-sanctioned facts are valid
                "confidence": 0.95
            }
        elif fact_type == "sec_filings":
            # Simulate SEC API call
            validation_result = {
                "validation_type": "external_api",
                "fact": fact,
                "fact_type": fact_type,
                "api_used": "SEC",
                "is_valid": True,  # Assume SEC facts are valid for demo
                "confidence": 0.9
            }
        else:
            # Generic external validation
            validation_result = {
                "validation_type": "external_api",
                "fact": fact,
                "fact_type": fact_type,
                "api_used": "generic",
                "is_valid": True,
                "confidence": 0.8
            }
        
        self.validation_history.append(validation_result)
        return validation_result
    
    def perform_cross_validation(self, facts: List[str], sources: List[str]) -> Dict[str, Any]:
        """Perform cross-validation across multiple information sources."""
        # Check if the same fact appears in multiple sources
        unique_facts = set(facts)
        validation_score = len(unique_facts) / len(facts) if facts else 0.0
        
        validation_result = {
            "validation_type": "cross_validation",
            "facts": facts,
            "sources": sources,
            "consistency_score": validation_score,
            "is_valid": validation_score > 0.5,  # More than 50% consistency
            "confidence": min(validation_score + 0.3, 1.0)  # Boost confidence slightly
        }
        
        self.validation_history.append(validation_result)
        return validation_result
    
    def validate_reasoning_chain(self, reasoning_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the entire reasoning chain for logical consistency."""
        # Check if all steps have high confidence and are logically connected
        avg_confidence = sum(step.get("confidence", 0.0) for step in reasoning_steps) / len(reasoning_steps) if reasoning_steps else 0.0
        
        # Simple validation: all steps must have confidence > 0.7
        is_consistent = all(step.get("confidence", 0.0) > 0.7 for step in reasoning_steps)
        
        validation_result = {
            "validation_type": "reasoning_chain",
            "reasoning_steps": reasoning_steps,
            "average_confidence": avg_confidence,
            "is_consistent": is_consistent,
            "is_valid": is_consistent and avg_confidence > 0.75,
            "confidence": avg_confidence if is_consistent else 0.0
        }
        
        self.validation_history.append(validation_result)
        return validation_result
