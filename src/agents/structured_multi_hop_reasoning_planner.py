"""Structured multi-hop reasoning planner — decomposes questions into hop plans.

Why: The old system used hardcoded 5-phase sequential execution with no explicit
hop targets or stop conditions.  56% of errors came from multi-hop chain breaks
where the correct domain was found but the wrong entity was selected.

This module provides structured hop planning (borrowed from Research_Agent's
parse_multi_hop_plan) where each hop has a clear target, tool, and stop condition.
It also includes a think-step (borrowed from open_deep_research's think_tool)
that forces reflection between hops.

References:
  - Research_Agent: parse_multi_hop_plan(), planning.yaml
  - open_deep_research: think_tool (strategic reflection between searches)
  - SQuAI: LLM-based question splitting
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

DEFAULT_HOP_PLAN = {
    "hop_count": 1,
    "hops": [
        {
            "hop_num": 1,
            "target": "直接搜索问题关键词找到答案",
            "stop_condition": "获取到能回答原问题的具体信息",
        }
    ],
    "total_stop_condition": "获取到完整答案",
}


# ---------------------------------------------------------------------------
# Hop plan generation
# ---------------------------------------------------------------------------

def generate_structured_multi_hop_plan(
    question: str,
    llm_fn: Optional[Callable[[str, str], str]] = None,
) -> Dict[str, Any]:
    """Generate a structured multi-hop reasoning plan from a question.

    Why: Without explicit hop targets, the agent blindly generates follow-up
    queries that often drift from the true reasoning chain.  A structured plan
    ensures each hop has a clear objective and completion criterion.

    Borrowed from Research_Agent parse_multi_hop_plan() with adaptations.

    Returns:
        A dict with keys: hop_count, hops (list of hop dicts), total_stop_condition.
        Each hop has: hop_num, target, stop_condition.
    """
    if not llm_fn:
        return dict(DEFAULT_HOP_PLAN)

    system_prompt = (
        "你是一个多跳推理规划专家。给定一个复杂问题，你需要分析它的推理链并输出结构化的多跳计划。\n\n"
        "## 规则\n"
        "1. 判断问题需要几跳推理（1跳、2跳、或3跳）\n"
        "2. 每跳必须有明确的搜索目标(target)和终止条件(stop_condition)\n"
        "3. 后跳的target应依赖前跳的结果\n"
        "4. 对于反推类问题（由结果推原因），按逆序拆解\n"
        "5. 中文问题用中文描述target，英文问题用英文描述target\n\n"
        "## 输出格式（严格JSON，无任何额外文本）\n"
        '{"hop_count": 数字, "hops": [{"hop_num": 1, "target": "搜索目标", "stop_condition": "获取到X信息即停止"}, ...], "total_stop_condition": "获取到完整答案"}\n\n'
        "## 示例\n"
        '问题: "一位欧洲学者创立的开源硬件商业实体的英文名称"\n'
        '输出: {"hop_count": 3, "hops": [{"hop_num": 1, "target": "找到该开源硬件项目名称和创始学者", "stop_condition": "获取到项目名和学者名"}, {"hop_num": 2, "target": "找到该学者创立的商业实体", "stop_condition": "获取到公司名"}, {"hop_num": 3, "target": "确认该公司在欧洲停止交易但亚洲业务延续的细节", "stop_condition": "获取到公司英文全称"}], "total_stop_condition": "获取到商业实体的完整英文名称"}\n\n'
        '问题: "Who is the author of article X in journal Y?"\n'
        '输出: {"hop_count": 2, "hops": [{"hop_num": 1, "target": "Find article X in journal Y", "stop_condition": "Found the specific article"}, {"hop_num": 2, "target": "Identify the author of that article", "stop_condition": "Found author name"}], "total_stop_condition": "Get the full author name"}'
    )
    user_prompt = f"问题: {question}\n\n输出JSON多跳计划:"

    try:
        raw = llm_fn(system_prompt, user_prompt)
        if not raw:
            return dict(DEFAULT_HOP_PLAN)
        return _parse_hop_plan_json(raw, question)
    except Exception as exc:
        logger.warning("Failed to generate hop plan: %s", exc)
        return dict(DEFAULT_HOP_PLAN)


def _parse_hop_plan_json(raw: str, question: str) -> Dict[str, Any]:
    """Robustly parse LLM output into a hop plan dict.

    Why: LLMs sometimes wrap JSON in markdown code blocks or add extra text.
    We try multiple cleanup strategies before falling back to default.
    """
    # Strip markdown code fences
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Try to extract JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        logger.warning("No JSON object found in hop plan response")
        return dict(DEFAULT_HOP_PLAN)

    try:
        plan = json.loads(match.group())
    except json.JSONDecodeError:
        # Try fixing common issues
        cleaned = match.group().replace("'", '"')
        try:
            plan = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Failed to parse hop plan JSON")
            return dict(DEFAULT_HOP_PLAN)

    # Validate structure
    if not isinstance(plan.get("hops"), list) or len(plan["hops"]) == 0:
        return dict(DEFAULT_HOP_PLAN)

    # Normalize hop_count
    hop_count = plan.get("hop_count", len(plan["hops"]))
    if isinstance(hop_count, str):
        # Handle Chinese: "两跳" -> 2, "三跳及以上" -> 3
        if "三" in hop_count or "3" in str(hop_count):
            hop_count = 3
        elif "两" in hop_count or "二" in hop_count or "2" in str(hop_count):
            hop_count = 2
        else:
            hop_count = 1
    plan["hop_count"] = min(int(hop_count), 4)  # Cap at 4 hops

    # Ensure each hop has required fields
    for hop in plan["hops"]:
        hop.setdefault("target", "搜索相关信息")
        hop.setdefault("stop_condition", "获取到相关信息")
        hop.setdefault("hop_num", plan["hops"].index(hop) + 1)

    plan.setdefault("total_stop_condition", "获取到完整答案")
    return plan


# ---------------------------------------------------------------------------
# Think step (reflection between hops)
# ---------------------------------------------------------------------------

def generate_inter_hop_reflection(
    question: str,
    current_hop: int,
    total_hops: int,
    hop_target: str,
    hop_result: str,
    accumulated_evidence: str,
    llm_fn: Optional[Callable[[str, str], str]] = None,
) -> Dict[str, Any]:
    """Force a reflection pause between hops to assess progress and plan next step.

    Why (from open_deep_research think_tool): Without reflection, the agent
    blindly generates follow-up queries.  Pausing to assess what was found
    and what is still missing significantly improves hop-2 query quality.

    Returns:
        Dict with keys: reflection, next_action, confidence, missing_info
    """
    if not llm_fn:
        return {
            "reflection": "No LLM available for reflection",
            "next_action": "continue",
            "confidence": 0.3,
            "missing_info": "unknown",
        }

    system_prompt = (
        "You are a research progress evaluator. Assess the current state of a "
        "multi-hop reasoning task and decide the next action.\n\n"
        "Output a JSON object with these fields:\n"
        '- "reflection": Brief assessment of what was found (1-2 sentences)\n'
        '- "next_action": One of "continue" (need more hops), "sufficient" (have enough info), or "retry" (current hop failed)\n'
        '- "confidence": Float 0-1 how confident the current evidence answers the question\n'
        '- "missing_info": What specific information is still needed (or "none" if sufficient)\n\n'
        "Output ONLY the JSON, no extra text."
    )
    user_prompt = (
        f"Original question: {question}\n\n"
        f"Current hop: {current_hop}/{total_hops}\n"
        f"Hop target: {hop_target}\n"
        f"Hop result: {hop_result[:2000]}\n\n"
        f"Accumulated evidence so far:\n{accumulated_evidence[:3000]}\n\n"
        f"Assess progress and decide next action:"
    )

    try:
        raw = llm_fn(system_prompt, user_prompt)
        if not raw:
            return {"reflection": "", "next_action": "continue", "confidence": 0.3, "missing_info": ""}
        # Parse JSON
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            result.setdefault("reflection", "")
            result.setdefault("next_action", "continue")
            result.setdefault("confidence", 0.3)
            result.setdefault("missing_info", "")
            return result
    except Exception as exc:
        logger.debug("Reflection parse failed: %s", exc)

    return {"reflection": raw[:200] if raw else "", "next_action": "continue", "confidence": 0.3, "missing_info": ""}


# ---------------------------------------------------------------------------
# Generate queries for a specific hop
# ---------------------------------------------------------------------------

def generate_queries_for_hop(
    question: str,
    hop_target: str,
    previous_hop_results: List[str],
    llm_fn: Optional[Callable[[str, str], str]] = None,
) -> List[str]:
    """Generate search queries tailored to a specific hop target.

    Why: Generic query generation ignores hop-specific goals.  By telling
    the LLM exactly what to search for (the hop target) and providing
    previous hop results, we get more focused queries.
    """
    if not llm_fn:
        return [hop_target]

    prev_context = ""
    if previous_hop_results:
        prev_context = "Previous research findings:\n" + "\n".join(
            f"- Hop {i+1}: {r[:300]}" for i, r in enumerate(previous_hop_results)
        ) + "\n\n"

    system_prompt = (
        "Generate 2-3 precise search queries for a specific research objective.\n"
        "Rules:\n"
        "- Each query should be 5-20 words\n"
        "- Include specific names, dates, terms from the question and previous findings\n"
        "- For Chinese topics use Chinese queries, for English topics use English queries\n"
        "- Output ONLY search queries, one per line, no numbering or explanations\n"
    )
    user_prompt = (
        f"Original question: {question}\n\n"
        f"{prev_context}"
        f"Current search objective: {hop_target}\n\n"
        f"Generate search queries:"
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
