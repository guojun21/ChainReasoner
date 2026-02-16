"""MiniAgent — modular tool-use loop for final answer decision.

Why: The multi-hop pipeline collects evidence but makes its final decision via
a rigid voting mechanism.  This module provides a nanobot-style MiniAgent loop
where the LLM can freely browse evidence using registered tools and submit a
final answer only when confident.

Key components:
  - MiniAgentToolBase / MiniAgentToolRegistry — tool abstraction and registry
  - run_mini_agent_loop — main ReAct loop entry point
  - prompt_builder — dynamic system prompt generation
  - tool_call_protocol_parser — triple-fallback tool call parser
  - large_result_offload — prevent context explosion from big tool outputs
"""

from src.mini_agent.mini_agent_loop import (
    run_mini_agent_loop,
)

__all__ = ["run_mini_agent_loop"]
