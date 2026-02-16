"""Answer submission tool — terminal tool that ends the MiniAgent loop.

Why: The submit_answer tool is special — it terminates the MiniAgent loop and
returns the final answer.  Separating it into its own file makes the terminal
semantics explicit and keeps the tool registry clean.
"""

from typing import Any

from src.mini_agent.tool_base_and_registry import MiniAgentToolBase


class SubmitFinalAnswerTool(MiniAgentToolBase):
    """Submit the final answer and terminate the MiniAgent loop."""

    name = "submit_answer"
    description = "Submit your final answer and STOP."
    when_to_use = "You have enough evidence to answer confidently. THIS IS MANDATORY."
    when_not_to_use = "You still have unanswered questions about the evidence."
    is_terminal = True
    parameters_schema = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The precise, short factual answer",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score 0.0-1.0",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why this answer is correct",
            },
        },
        "required": ["answer", "confidence", "reasoning"],
    }

    def execute(self, **kwargs: Any) -> str:
        """Return confirmation — actual handling is done by the main loop."""
        return "(answer submitted)"
