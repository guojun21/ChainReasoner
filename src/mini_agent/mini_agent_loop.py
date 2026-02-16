"""MiniAgent loop — nanobot-style ReAct loop for final answer decision.

Why: The multi-hop pipeline collects evidence into a scratchpad but makes its
final decision via a rigid voting mechanism.  This module provides a modular
MiniAgent loop (inspired by nanobot loop.py) where the LLM can freely browse
evidence using registered tools and submit a final answer only when confident.

Key improvements over the old monolithic implementation:
  1. Iteration limit raised to 12 (from 6) with adaptive early termination
  2. Large tool results offloaded to scratchpad files (not hard-truncated)
  3. Triple-fallback tool-call parser (XML + JSON block + free-text)
  4. Modular tool registry (each tool is a self-contained class)
  5. Enhanced structured logging for every iteration

References:
  - nanobot (15k stars): agent/loop.py — while has_tool_calls ReAct loop
  - deepagents (9.2k stars): filesystem middleware — large result offload
"""

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional

from src.mini_agent.tool_base_and_registry import MiniAgentToolRegistry
from src.mini_agent.tool_call_protocol_parser import parse_tool_call_from_llm_response
from src.mini_agent.large_result_offload import (
    offload_large_tool_result_to_scratchpad_file_with_metadata,
    TOOLS_EXCLUDED_FROM_OFFLOAD,
)
from src.mini_agent.prompt_builder import (
    build_mini_agent_system_prompt,
)
from src.mini_agent.local_evidence_tools import (
    ListEvidenceAndPageFilesTool,
    GetEvidenceIndexOverviewTool,
    ReadEvidenceFileTool,
    GrepEvidenceKeywordSearchTool,
    SearchLocalBm25EvidenceTool,
)
from src.mini_agent.web_search_tools import (
    WebSearchTool,
    FetchPageByUrlTool,
    DeepWikiSearchTool,
)
from src.mini_agent.answer_submission_tool import SubmitFinalAnswerTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_ITERATIONS = 12  # Raised from 6; nanobot uses 20
_CONFIDENCE_EARLY_STOP_THRESHOLD = 0.7
_MIN_LOCAL_EVIDENCE_CHARS_FOR_EARLY_STOP = 2000
_LOCAL_TOOL_NAMES = frozenset({
    "grep_evidence", "read_file", "search_local", "list_files", "get_index",
})


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------

def _build_tool_registry(
    scratchpad: Any,
    search_fn: Optional[Callable] = None,
    mcp_config: Optional[Dict] = None,
    mcp_http_clients: Optional[Dict] = None,
    language_priority: str = "bilingual_equal",
    trace_logger: Optional[Any] = None,
) -> MiniAgentToolRegistry:
    """Create and populate a tool registry with all MiniAgent tools.

    Why: Centralised registry creation ensures all tools are consistently
    initialised with the correct dependencies (scratchpad, search_fn, etc.).
    """
    registry = MiniAgentToolRegistry()

    # Local evidence tools
    registry.register_tool(ListEvidenceAndPageFilesTool(scratchpad))
    registry.register_tool(GetEvidenceIndexOverviewTool(scratchpad))
    registry.register_tool(ReadEvidenceFileTool(scratchpad))
    registry.register_tool(GrepEvidenceKeywordSearchTool(scratchpad))
    registry.register_tool(SearchLocalBm25EvidenceTool(scratchpad))

    # Network tools
    registry.register_tool(WebSearchTool(search_fn, language_priority, trace_logger=trace_logger))
    registry.register_tool(FetchPageByUrlTool(mcp_config, mcp_http_clients, trace_logger=trace_logger))
    registry.register_tool(DeepWikiSearchTool(mcp_config, mcp_http_clients))

    # Terminal tool
    registry.register_tool(SubmitFinalAnswerTool())

    return registry


# ---------------------------------------------------------------------------
# Adaptive countdown message builder
# ---------------------------------------------------------------------------

def _detect_all_candidates_rejected_pattern(tool_calls_log: List[Dict]) -> bool:
    """Return True if recent grep_evidence calls suggest all candidates are rejected.

    Why (from Q99 analysis): The agent did 4 grep_evidence calls, found that both
    Peter Ndlovu (born 1973) and Quinton Fortune (born 1977) fail the 1988-1995
    constraint, then submitted "None of the provided candidates satisfy all
    constraints" with 0.95 confidence.  It should have used web_search instead.

    Detection: If the last 3+ tool calls are all local (grep/read/search_local)
    and no web_search has been used yet, the agent is stuck in local-only mode.
    """
    if len(tool_calls_log) < 3:
        return False
    recent = tool_calls_log[-3:]
    has_web_search = any(tc.get("tool") == "web_search" for tc in tool_calls_log)
    all_local_recent = all(
        tc.get("tool") in {"grep_evidence", "read_file", "search_local", "list_files", "get_index"}
        for tc in recent
    )
    return all_local_recent and not has_web_search


def _build_adaptive_countdown_message(
    iteration: int,
    remaining: int,
    cumulative_local_chars: int,
    tool_calls_log: List[Dict],
) -> tuple[str, str]:
    """Build context-aware guidance message for the next iteration.

    Why: Adaptive prompting (inspired by the old Phase 5.4-5.6 logic) guides
    the LLM to use web_search when local evidence is thin, and to submit
    when evidence is sufficient — preventing both premature and endless loops.

    Enhancement (from Q99 analysis): Detects when the agent is stuck in
    local-only grep mode with all candidates rejected, and forces it to
    use web_search to find alternative candidates before giving up.
    """
    if remaining <= 1:
        return (
            f"[Tool calls remaining: {remaining}] "
            f"You MUST call submit_answer NOW. This is your last chance.",
            "force_submit_last_chance",
        )

    if remaining <= 2:
        if cumulative_local_chars < 500:
            return (
                f"[Tool calls remaining: {remaining}] "
                f"Local evidence is very thin ({cumulative_local_chars} chars). "
                f"Consider web_search for more info, then submit_answer next round.",
                "thin_local_evidence_near_end",
            )
        return (
            f"[Tool calls remaining: {remaining}] "
            f"You should call submit_answer soon.",
            "submit_soon",
        )

    # Detect all-candidates-rejected pattern: agent is stuck grepping locally
    # without finding a valid candidate.  Force it to use web_search.
    if _detect_all_candidates_rejected_pattern(tool_calls_log) and remaining >= 3:
        return (
            f"[Tool calls remaining: {remaining}] "
            f"CANDIDATE REJECTION ALERT: You have used {len(tool_calls_log)} local tool calls "
            f"without finding a valid candidate. All pipeline candidates may be WRONG.\n"
            f"You MUST use web_search NOW to find ALTERNATIVE candidates.\n"
            f"Construct queries that INCLUDE the constraints failed candidates missed "
            f"and EXCLUDE rejected entity names (e.g. add -Ndlovu -Fortune).\n"
            f"Do NOT submit 'Unknown' or 'None of the candidates' — search for the real answer first.",
            "candidate_rejection_force_web_search",
        )

    if iteration >= 2:
        return (
            f"[Tool calls remaining: {remaining}] "
            f"REFLECTION CHECKPOINT: Before continuing, verify:\n"
            f"1. Does your current best answer satisfy ALL constraints in the question?\n"
            f"2. Have you checked alternative candidates?\n"
            f"3. Is there a different entity that better matches?\n"
            f"If you are confident, call submit_answer. "
            f"If evidence is insufficient ({cumulative_local_chars} chars from local tools), use web_search.",
            "reflection_checkpoint",
        )

    if iteration >= 1 and cumulative_local_chars < 300:
        return (
            f"[Tool calls remaining: {remaining}] "
            f"Local evidence is thin ({cumulative_local_chars} chars). "
            f"Consider using web_search to find more information.",
            "thin_local_evidence_early",
        )

    return (
        f"[Tool calls remaining: {remaining}] "
        f"Continue reviewing or call submit_answer when ready.",
        "continue_review",
    )


# ---------------------------------------------------------------------------
# Main agentic loop
# ---------------------------------------------------------------------------

def run_mini_agent_loop(
    question: str,
    pipeline_answer: str,
    pipeline_candidates: Dict[str, str],
    scratchpad: Any,
    llm_fn: Callable[[str, str], str],
    search_fn: Optional[Callable] = None,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    trace_logger: Optional[Any] = None,
    mcp_config: Optional[Dict] = None,
    mcp_http_clients: Optional[Dict] = None,
    hop_evaluations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the MiniAgent loop with modular tools and adaptive control.

    Why: The pipeline's voting mechanism is rigid — it cannot browse evidence
    or do targeted follow-up searches.  This loop lets the LLM freely review
    all collected evidence and make a more informed final decision.

    Key improvements:
      - 12 iterations (up from 6) with adaptive early termination
      - Triple-fallback tool-call parser (XML + JSON + free-text)
      - Large results offloaded to scratchpad (not hard-truncated)
      - Modular tool registry (each tool is a self-contained class)
      - Enhanced structured logging per iteration

    Args:
        question: The original question.
        pipeline_answer: The answer from the pipeline's voting phase.
        pipeline_candidates: Dict of candidate answers (source -> answer).
        scratchpad: PerQuestionEvidenceScratchpad with collected evidence.
        llm_fn: Callable(system_prompt, user_prompt) -> str.
        search_fn: Optional search function for web_search tool.
        max_iterations: Maximum loop iterations.
        trace_logger: Optional trace logger.
        mcp_config: MCP configuration dict for fetch_page / deepwiki_search.
        mcp_http_clients: Dict of MCPHttpClient instances.

    Returns:
        Dict with keys: answer, confidence, reasoning, tool_calls_count,
        iterations, tool_calls_log, parse_failures, offloaded_files, elapsed_ms.
    """
    start_time = time.time()

    # Determine language priority for search tools
    language_priority = "bilingual_equal"
    try:
        from src.mini_agent.language_router import (
            classify_question_geographic_language_priority,
        )
        lang_result = classify_question_geographic_language_priority(
            question, llm_fn=None,  # Use rules only — no extra LLM call
        )
        language_priority = lang_result.get("priority", "bilingual_equal")
        logger.info("MiniAgent language priority: %s", language_priority)
    except Exception as exc:
        logger.debug("Language routing failed, using bilingual_equal: %s", exc)

    # Build tool registry
    registry = _build_tool_registry(
        scratchpad=scratchpad,
        search_fn=search_fn,
        mcp_config=mcp_config,
        mcp_http_clients=mcp_http_clients,
        language_priority=language_priority,
        trace_logger=trace_logger,
    )

    # Build prompts
    index_content = scratchpad.get_index() if scratchpad else ""
    system_prompt, user_prompt = build_mini_agent_system_prompt(
        registry=registry,
        question=question,
        pipeline_answer=pipeline_answer,
        pipeline_candidates=pipeline_candidates,
        index_content=index_content,
        max_iterations=max_iterations,
        hop_evaluations=hop_evaluations,
    )

    # Conversation history (simulated multi-turn via accumulated context)
    conversation_parts = [user_prompt]
    tool_calls_log: List[Dict[str, Any]] = []
    parse_failures = 0
    offloaded_files: List[str] = []
    consecutive_no_tool_calls = 0

    result: Dict[str, Any] = {
        "answer": pipeline_answer,
        "confidence": 0.5,
        "reasoning": "Defaulted to pipeline answer",
        "tool_calls_count": 0,
        "iterations": 0,
        "tool_calls_log": tool_calls_log,
        "parse_failures": 0,
        "offloaded_files": offloaded_files,
    }

    def _record_iteration_trace(
        iteration_index: int,
        details: Dict[str, Any],
    ) -> None:
        """Record one iteration-level trace event if the logger supports it."""
        if trace_logger and hasattr(trace_logger, "record_mini_agent_iteration_trace"):
            trace_logger.record_mini_agent_iteration_trace(
                question=question,
                iteration=iteration_index + 1,
                details=details,
            )

    for iteration in range(max_iterations):
        result["iterations"] = iteration + 1
        remaining = max_iterations - iteration - 1

        # Build full user prompt from conversation history
        full_user_prompt = "\n\n---\n\n".join(conversation_parts)

        # Call LLM
        try:
            llm_response = llm_fn(system_prompt, full_user_prompt)
        except Exception as exc:
            logger.warning("MiniAgent LLM call failed at iter %d: %s", iteration + 1, exc)
            _record_iteration_trace(iteration, {
                "status": "llm_error",
                "remaining_iterations": remaining,
                "parse_protocol": "none",
                "parse_success": False,
                "llm_response_length": 0,
                "error": str(exc),
            })
            break

        if not llm_response:
            logger.warning("MiniAgent: empty LLM response at iter %d", iteration + 1)
            consecutive_no_tool_calls += 1
            _record_iteration_trace(iteration, {
                "status": "empty_llm_response",
                "remaining_iterations": remaining,
                "parse_protocol": "none",
                "parse_success": False,
                "llm_response_length": 0,
                "consecutive_no_tool_calls": consecutive_no_tool_calls,
            })
            if consecutive_no_tool_calls >= 2:
                break
            continue

        # Parse tool call (triple fallback: XML > JSON block > free-text)
        tool_call = parse_tool_call_from_llm_response(llm_response)
        protocol_used = tool_call.get("_protocol", "none") if tool_call else "none"

        if tool_call is None:
            parse_failures += 1
            consecutive_no_tool_calls += 1

            # Try to extract answer from free text
            answer_match = re.search(
                r'(?:final answer|answer is|my answer)[:\s]*["\']?([^"\'\n]+)',
                llm_response, re.IGNORECASE,
            )
            if answer_match:
                result["answer"] = answer_match.group(1).strip()
                result["confidence"] = 0.6
                result["reasoning"] = "Extracted from LLM text (no submit_answer call)"
                _record_iteration_trace(iteration, {
                    "status": "free_text_answer_extracted",
                    "remaining_iterations": remaining,
                    "parse_protocol": protocol_used,
                    "parse_success": False,
                    "llm_response_length": len(llm_response),
                    "llm_reasoning_text": llm_response[:2000],
                    "answer_preview": result["answer"][:120],
                    "consecutive_no_tool_calls": consecutive_no_tool_calls,
                })
                break

            # Adaptive: nanobot terminates after 2 consecutive no-tool-call responses
            if consecutive_no_tool_calls >= 2:
                # Try Chinese answer patterns too
                zh_match = re.search(
                    r'(?:答案[是为：:]\s*|最终答案[是为：:]\s*)([^\n，。,\.]+)',
                    llm_response,
                )
                if zh_match:
                    result["answer"] = zh_match.group(1).strip()
                    result["confidence"] = 0.55
                    result["reasoning"] = "Extracted from Chinese LLM text (forced by no-tool-call limit)"
                _record_iteration_trace(iteration, {
                    "status": "parse_failure_forced_stop",
                    "remaining_iterations": remaining,
                    "parse_protocol": protocol_used,
                    "parse_success": False,
                    "llm_response_length": len(llm_response),
                    "llm_reasoning_text": llm_response[:2000],
                    "consecutive_no_tool_calls": consecutive_no_tool_calls,
                    "answer_preview": result.get("answer", "")[:120],
                })
                break

            # Prompt retry
            if remaining > 0:
                conversation_parts.append(
                    f"ERROR: Your response did not contain a <tool_call> tag. "
                    f"Your ENTIRE response must be ONLY a <tool_call> tag, nothing else. "
                    f'Example: <tool_call>{{"name": "submit_answer", "args": {{"answer": "your answer", "confidence": 0.8, "reasoning": "brief reason"}}}}</tool_call>\n\n'
                    f"[Tool calls remaining: {remaining}]"
                )
                logger.info(
                    "MiniAgent iter %d: parse failure (protocol=%s), prompting retry",
                    iteration + 1, protocol_used,
                )
                _record_iteration_trace(iteration, {
                    "status": "parse_failure_retry",
                    "remaining_iterations": remaining,
                    "parse_protocol": protocol_used,
                    "parse_success": False,
                    "llm_response_length": len(llm_response),
                    "llm_reasoning_text": llm_response[:2000],
                    "consecutive_no_tool_calls": consecutive_no_tool_calls,
                })
                continue
            _record_iteration_trace(iteration, {
                "status": "parse_failure_no_remaining_iterations",
                "remaining_iterations": remaining,
                "parse_protocol": protocol_used,
                "parse_success": False,
                "llm_response_length": len(llm_response),
                "llm_reasoning_text": llm_response[:2000],
                "consecutive_no_tool_calls": consecutive_no_tool_calls,
            })
            break

        # Valid tool call found — reset consecutive counter
        consecutive_no_tool_calls = 0

        # Handle submit_answer — terminal tool
        if tool_call["name"] == "submit_answer":
            args = tool_call.get("args", {})
            result["answer"] = str(args.get("answer", pipeline_answer)).strip()
            result["confidence"] = float(args.get("confidence", 0.5))
            result["reasoning"] = str(args.get("reasoning", ""))
            result["tool_calls_count"] += 1
            tool_calls_log.append({
                "iteration": iteration + 1,
                "tool": "submit_answer",
                "args": args,
                "protocol": protocol_used,
            })
            _record_iteration_trace(iteration, {
                "status": "submit_answer",
                "remaining_iterations": remaining,
                "parse_protocol": protocol_used,
                "parse_success": True,
                "llm_response_length": len(llm_response),
                "llm_reasoning_text": llm_response[:2000],
                "tool_name": "submit_answer",
                "tool_args": args,
                "result_length": 0,
                "tool_exec_elapsed_ms": 0,
                "was_offloaded": False,
                "offload_file_path": "",
                "cumulative_local_chars": sum(
                    tc.get("result_length", 0) for tc in tool_calls_log
                    if tc.get("tool") in _LOCAL_TOOL_NAMES
                ),
                "countdown_reason": "submitted",
            })
            break

        # Execute the tool via registry
        tool_exec_start = time.time()
        tool_result = registry.execute_tool_by_name(
            tool_call["name"], tool_call.get("args", {}),
        )
        tool_exec_elapsed_ms = int((time.time() - tool_exec_start) * 1000)
        result["tool_calls_count"] += 1

        # Offload large results to scratchpad files
        was_offloaded = False
        offload_file_path = ""
        if (tool_call["name"] not in TOOLS_EXCLUDED_FROM_OFFLOAD
                and scratchpad is not None):
            offloaded_result, was_offloaded, offload_file_path = (
                offload_large_tool_result_to_scratchpad_file_with_metadata(
                tool_name=tool_call["name"],
                tool_result=tool_result,
                scratchpad=scratchpad,
                )
            )
            if was_offloaded:
                offloaded_files.append(offload_file_path)
                tool_result = offloaded_result

        tool_calls_log.append({
            "iteration": iteration + 1,
            "tool": tool_call["name"],
            "args": tool_call.get("args", {}),
            "result_length": len(tool_result),
            "was_offloaded": was_offloaded,
            "offload_file_path": offload_file_path,
            "tool_exec_elapsed_ms": tool_exec_elapsed_ms,
            "protocol": protocol_used,
        })

        logger.info(
            "MiniAgent iter %d: tool=%s result_len=%d exec_ms=%d offloaded=%s protocol=%s",
            iteration + 1, tool_call["name"], len(tool_result),
            tool_exec_elapsed_ms, was_offloaded, protocol_used,
        )

        # Track cumulative local tool result length
        cumulative_local_chars = sum(
            tc.get("result_length", 0) for tc in tool_calls_log
            if tc.get("tool") in _LOCAL_TOOL_NAMES
        )

        # Build adaptive countdown message
        countdown_msg, countdown_reason = _build_adaptive_countdown_message(
            iteration, remaining, cumulative_local_chars, tool_calls_log,
        )

        # Log the countdown decision and pattern detection
        if trace_logger and hasattr(trace_logger, "record_event"):
            if countdown_reason == "candidate_rejection_force_web_search":
                trace_logger.record_event("all_candidates_rejected_detected",
                    f"iter={iteration} — forcing web_search due to local-only rejection pattern")
            trace_logger.record_event("countdown_message_selected",
                f"iter={iteration} reason={countdown_reason} remaining={remaining}",
                details={"countdown_reason": countdown_reason, "remaining": remaining})

        # Append tool call + result to conversation for next iteration
        conversation_parts.append(
            f"You called: {tool_call['name']}({json.dumps(tool_call.get('args', {}), ensure_ascii=False)})\n"
            f"Result:\n{tool_result}\n\n"
            f"{countdown_msg}"
        )

        _record_iteration_trace(iteration, {
            "status": "tool_executed",
            "remaining_iterations": remaining,
            "parse_protocol": protocol_used,
            "parse_success": True,
            "llm_response_length": len(llm_response),
            "llm_reasoning_text": llm_response[:2000],
            "tool_name": tool_call["name"],
            "tool_args": tool_call.get("args", {}),
            "result_length": len(tool_result),
            "tool_exec_elapsed_ms": tool_exec_elapsed_ms,
            "was_offloaded": was_offloaded,
            "offload_file_path": offload_file_path,
            "cumulative_local_chars": cumulative_local_chars,
            "countdown_reason": countdown_reason,
        })

    # Finalise result
    elapsed_ms = int((time.time() - start_time) * 1000)
    result["elapsed_ms"] = elapsed_ms
    result["parse_failures"] = parse_failures
    result["offloaded_files"] = offloaded_files
    result["language_priority"] = language_priority

    # Build tool calls breakdown for logging
    tool_breakdown: Dict[str, int] = {}
    for tc in tool_calls_log:
        tool_name = tc.get("tool", "unknown")
        tool_breakdown[tool_name] = tool_breakdown.get(tool_name, 0) + 1

    logger.info(
        "MiniAgent loop complete: answer='%s' confidence=%.2f "
        "iterations=%d tool_calls=%d parse_failures=%d offloaded=%d "
        "lang_priority=%s elapsed_ms=%d breakdown=%s",
        result["answer"][:60], result["confidence"],
        result["iterations"], result["tool_calls_count"],
        parse_failures, len(offloaded_files),
        language_priority, elapsed_ms, tool_breakdown,
    )

    # Record trace
    if trace_logger and hasattr(trace_logger, "record_scratchpad_operation_trace"):
        trace_logger.record_scratchpad_operation_trace(
            operation="mini_agent_loop",
            details={
                "answer": result["answer"],
                "confidence": result["confidence"],
                "reasoning": result["reasoning"],
                "iterations": result["iterations"],
                "tool_calls_count": result["tool_calls_count"],
                "tool_calls_log": tool_calls_log,
                "parse_failures": parse_failures,
                "offloaded_files": offloaded_files,
                "language_priority": language_priority,
                "tool_breakdown": tool_breakdown,
                "elapsed_ms": elapsed_ms,
            },
        )

    return result
