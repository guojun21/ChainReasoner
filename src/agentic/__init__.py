"""Backward-compatibility shim — module has been renamed to src/mini_agent/.

This shim ensures any code that still imports from ``src.agentic`` continues
to work without modification.
"""

from src.mini_agent import run_mini_agent_loop

# Backward-compatible alias
run_agentic_evidence_decision_loop = run_mini_agent_loop

__all__ = ["run_agentic_evidence_decision_loop", "run_mini_agent_loop"]
