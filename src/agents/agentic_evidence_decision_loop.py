"""Backward-compatibility shim — moved to src/mini_agent/.

Why: The agentic evidence decision loop has been refactored and renamed to
MiniAgent.  The new modular package lives at src/mini_agent/ with separate
files for tool registry, tool implementations, protocol parser, offload
middleware, prompt builder, and the main loop.

This shim preserves the old import path so existing code that imports
``run_agentic_evidence_decision_loop`` from here continues to work.

New location: src/mini_agent/mini_agent_loop.py
"""

from src.mini_agent.mini_agent_loop import run_mini_agent_loop

# Backward-compatible alias
run_agentic_evidence_decision_loop = run_mini_agent_loop

__all__ = ["run_agentic_evidence_decision_loop", "run_mini_agent_loop"]
