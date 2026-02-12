"""Per-hop result validator and corrector — verifies each hop's output before proceeding.

Why: 56% of multi-hop errors came from wrong intermediate entities (e.g. Q97
found '福临' instead of '皇太极').  Without validation, these errors propagate
through the entire reasoning chain.

This module implements:
  1. evaluate_hop_result_and_decide_next_action() — unified 3-in-1 call that
     validates, reflects, and checks chain completion in a SINGLE LLM call
     (P0-d: merged validate_hop + reflection + completion_check)
  2. generate_corrected_hop_queries() — on failure, generate refined queries

References:
  - Research_Agent: validate_hop_result(), correct_hop_result()
  - Enhancing-Multi-Hop-QA: SelfVerifier with SUPPORTS/REFUTES/INSUFFICIENT
  - everything-claude-code: iterative retrieval pattern (DISPATCH→EVALUATE→REFINE→LOOP)
  - analysis_claude_code v3: sub-agent context isolation
"""

import json
import logging
import re
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum retrieval cycles per hop (from everything-claude-code iterative retrieval)
MAX_RETRIEVAL_CYCLES_PER_HOP = 2


# ---------------------------------------------------------------------------
# Unified hop evaluation (P0-d: merged 3 calls → 1)
# ---------------------------------------------------------------------------

def evaluate_hop_result_and_decide_next_action(
    question: str,
    hop_num: int,
    total_hops: int,
    hop_target: str,
    hop_stop_condition: str,
    hop_result: str,
    previous_hop_summaries: str,
    total_stop_condition: str,
    is_last_hop: bool,
    llm_fn: Optional[Callable[[str, str], str]] = None,
) -> Dict[str, Any]:
    """Unified evaluation: validate hop + reflect on progress + check chain completion.

    Why (P0-d, from everything-claude-code iterative retrieval + analysis_claude_code
    sub-agent pattern): The old flow made 3 separate LLM calls per hop that all
    asked essentially the same question ("is the evidence sufficient?").  Merging
    them into one call reduces per-hop LLM overhead from 5-7 calls to 2-4.

    Returns:
        Dict with keys:
            valid (bool): Does this hop's result satisfy its stop condition?
            confidence (float): 0-1 confidence in the evaluation
            extracted_entity (str): The key entity/answer found in this hop
            next_action (str): "continue" | "refine" | "sufficient"
            missing_info (str): What specific info is still needed
            is_chain_complete (bool): Can we already answer the original question?
            chain_answer (str): Direct answer if chain is complete
            reasoning (str): Brief explanation of the decision
    """
    fallback = {
        "valid": bool(hop_result),
        "confidence": 0.5 if hop_result else 0.0,
        "extracted_entity": "",
        "next_action": "continue",
        "missing_info": "",
        "is_chain_complete": False,
        "chain_answer": "",
        "reasoning": "No LLM available" if not llm_fn else "Empty hop result",
    }
    if not llm_fn or not hop_result:
        return fallback

    system_prompt = (
        "You are a multi-hop research evaluator. Given a hop's search result, do THREE things in ONE response:\n\n"
        "1. VALIDATE: Does this hop's result satisfy its target and stop condition?\n"
        "2. REFLECT: Given all findings so far, what is the overall progress?\n"
        "3. DECIDE: What should happen next?\n\n"
        "Output a single JSON object:\n"
        "{\n"
        '  "valid": true/false,\n'
        '  "confidence": 0.0-1.0,\n'
        '  "extracted_entity": "the key entity/name/date/number found in this hop",\n'
        '  "next_action": "continue" or "refine" or "sufficient",\n'
        '  "missing_info": "what specific information is still missing (or empty if sufficient)",\n'
        '  "is_chain_complete": true/false,\n'
        '  "chain_answer": "direct answer to original question if chain is complete, else empty",\n'
        '  "reasoning": "1-2 sentence explanation"\n'
        "}\n\n"
        "Rules:\n"
        '- "valid"=true ONLY if the result clearly contains info satisfying the stop condition\n'
        '- "extracted_entity": extract the EXACT entity/name/number — do NOT paraphrase\n'
        '- "next_action"="sufficient" if we have enough evidence to answer the original question\n'
        '- "next_action"="refine" if this hop\'s search missed the target — need better queries\n'
        '- "next_action"="continue" if this hop is done but we need more hops\n'
        '- "is_chain_complete"=true ONLY if accumulated evidence fully answers the original question\n'
        "- Output ONLY the JSON, nothing else"
    )
    user_prompt = (
        f"Original question: {question}\n"
        f"Hop {hop_num}/{total_hops} target: {hop_target}\n"
        f"Stop condition: {hop_stop_condition}\n"
        f"Total stop condition: {total_stop_condition}\n"
    )
    if previous_hop_summaries:
        user_prompt += f"\nPrevious findings:\n{previous_hop_summaries}\n"
    user_prompt += (
        f"\nThis hop's search result:\n{hop_result[:3000]}\n\n"
        f"Evaluate this hop and decide next action:"
    )

    try:
        raw = llm_fn(system_prompt, user_prompt)
        if not raw:
            return fallback

        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            # Ensure all keys with correct types
            result["valid"] = bool(result.get("valid", False))
            result["confidence"] = float(result.get("confidence", 0.5))
            result.setdefault("extracted_entity", "")
            result.setdefault("next_action", "continue")
            result.setdefault("missing_info", "")
            result["is_chain_complete"] = bool(result.get("is_chain_complete", False))
            result.setdefault("chain_answer", "")
            result.setdefault("reasoning", "")
            # Normalize next_action
            if result["next_action"] not in ("continue", "refine", "sufficient"):
                result["next_action"] = "continue"
            return result
    except Exception as exc:
        logger.debug("Unified hop evaluation parse failed: %s", exc)

    return fallback


# ---------------------------------------------------------------------------
# Hop query refinement (used in iterative retrieval loop)
# ---------------------------------------------------------------------------

def generate_refined_hop_queries_from_evaluation(
    question: str,
    hop_num: int,
    hop_target: str,
    current_result: str,
    missing_info: str,
    reasoning: str,
    llm_fn: Optional[Callable[[str, str], str]] = None,
) -> list:
    """Generate refined search queries based on evaluation feedback.

    Why (from everything-claude-code iterative retrieval REFINE phase):
    Instead of generic "retry", use the specific missing_info from evaluation
    to generate targeted follow-up queries that address exactly what was missed.

    Returns:
        List of 1-3 refined search queries.
    """
    if not llm_fn:
        return [hop_target]

    system_prompt = (
        "A search attempt did not find all needed information. "
        "Generate 1-3 BETTER search queries based on what is still missing.\n\n"
        "Rules:\n"
        "- Focus on the MISSING information specifically\n"
        "- Include disambiguating terms from the existing results\n"
        "- Be more specific than the original query\n"
        "- If a wrong entity was found, add terms to distinguish the correct one\n"
        "- Output ONLY search queries, one per line, no numbering"
    )
    user_prompt = (
        f"Original question: {question}\n"
        f"Search target: {hop_target}\n"
        f"What we found: {current_result[:400]}\n"
        f"What is MISSING: {missing_info}\n"
        f"Evaluation reasoning: {reasoning}\n\n"
        f"Generate refined search queries to find the missing info:"
    )

    try:
        raw = llm_fn(system_prompt, user_prompt)
        if not raw:
            return [hop_target]
        queries = []
        for line in raw.strip().split("\n"):
            line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
            line = re.sub(r"^[-•]\s*", "", line).strip().strip('"').strip("'")
            if line and len(line) >= 4:
                queries.append(line)
        return queries[:3] if queries else [hop_target]
    except Exception:
        return [hop_target]


# ---------------------------------------------------------------------------
# Legacy functions (kept for backward compatibility, used by old callers)
# ---------------------------------------------------------------------------

def validate_single_hop_result(
    question: str,
    hop_num: int,
    hop_target: str,
    hop_stop_condition: str,
    hop_result: str,
    llm_fn: Optional[Callable[[str, str], str]] = None,
) -> Dict[str, Any]:
    """Legacy: Validate a single hop result. Delegates to unified evaluator."""
    result = evaluate_hop_result_and_decide_next_action(
        question=question, hop_num=hop_num, total_hops=hop_num,
        hop_target=hop_target, hop_stop_condition=hop_stop_condition,
        hop_result=hop_result, previous_hop_summaries="",
        total_stop_condition=hop_stop_condition, is_last_hop=True,
        llm_fn=llm_fn,
    )
    return {
        "valid": result["valid"],
        "confidence": result["confidence"],
        "reasoning": result["reasoning"],
        "extracted_entity": result["extracted_entity"],
    }


def generate_corrected_hop_queries(
    question: str,
    hop_num: int,
    hop_target: str,
    failed_result: str,
    failure_reasoning: str,
    llm_fn: Optional[Callable[[str, str], str]] = None,
) -> list:
    """Legacy: Generate corrected queries. Delegates to refined version."""
    return generate_refined_hop_queries_from_evaluation(
        question=question, hop_num=hop_num, hop_target=hop_target,
        current_result=failed_result, missing_info="",
        reasoning=failure_reasoning, llm_fn=llm_fn,
    )


def check_total_reasoning_chain_completion(
    question: str,
    accumulated_evidence: str,
    total_stop_condition: str,
    llm_fn: Optional[Callable[[str, str], str]] = None,
) -> Tuple[bool, float, str]:
    """Legacy: Check chain completion. Delegates to unified evaluator."""
    result = evaluate_hop_result_and_decide_next_action(
        question=question, hop_num=1, total_hops=1,
        hop_target="完成回答", hop_stop_condition=total_stop_condition,
        hop_result=accumulated_evidence[:3000], previous_hop_summaries="",
        total_stop_condition=total_stop_condition, is_last_hop=True,
        llm_fn=llm_fn,
    )
    info = result["chain_answer"] if result["is_chain_complete"] else result["missing_info"]
    return (result["is_chain_complete"], result["confidence"], info)
