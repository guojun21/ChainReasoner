"""LLM interaction methods for the API server.

Why: All LLM prompt engineering and HTTP call logic lives here so the
main server module stays focused on routing and orchestration.
"""

import time
from typing import Any, Dict, Optional, Tuple

import requests


def send_chat_completion_request_with_retry(base_model: dict, system_prompt: str, user_prompt: str,
                     temperature: float = 0.0, max_tokens: int = 256,
                     purpose: str = "", logger=None, trace_logger=None) -> str:
    """Send a chat completion request with retry logic.

    Why: All LLM calls share the same retry/rate-limit/logging pattern;
    centralising it avoids duplicated error handling.
    """
    api_url = base_model.get("api_url")
    api_key = base_model.get("api_key")
    model_id = base_model.get("model_id")
    if not api_url or not api_key or not model_id:
        return ""

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    max_retries = 3
    start_time = time.time()
    for attempt in range(max_retries):
        try:
            if logger:
                logger.info("LLM generic call: purpose=%s model=%s temp=%s max_tokens=%s system_len=%d user_len=%d",
                            purpose or "generic", model_id, temperature, max_tokens,
                            len(system_prompt), len(user_prompt))
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            if "choices" in result and result["choices"]:
                response_text = result["choices"][0]["message"]["content"].strip()
                elapsed_ms = int((time.time() - start_time) * 1000)
                if trace_logger:
                    trace_logger.record_llm_api_call(
                        purpose=purpose or "generic", model_id=model_id,
                        system_prompt=system_prompt, user_prompt=user_prompt,
                        temperature=temperature, max_tokens=max_tokens,
                        response_text=response_text, elapsed_ms=elapsed_ms,
                        status="success",
                    )
                return response_text
            if result.get("status") in ("449", "429") or "rate limit" in result.get("msg", "").lower():
                if logger:
                    logger.warning("LLM rate limited (attempt %d), retrying...", attempt + 1)
                time.sleep(5 * (attempt + 1))
                continue
            if logger:
                logger.error("LLM missing choices: %s", str(result)[:400])
            elapsed_ms = int((time.time() - start_time) * 1000)
            if trace_logger:
                trace_logger.record_llm_api_call(
                    purpose=purpose or "generic", model_id=model_id,
                    system_prompt=system_prompt, user_prompt=user_prompt,
                    temperature=temperature, max_tokens=max_tokens,
                    response_text=str(result)[:2000], elapsed_ms=elapsed_ms,
                    status="error", error="missing choices in response",
                )
            return ""
        except Exception as exc:
            if logger:
                logger.error("LLM call error (attempt %d): %s", attempt + 1, exc)
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                elapsed_ms = int((time.time() - start_time) * 1000)
                if trace_logger:
                    trace_logger.record_llm_api_call(
                        purpose=purpose or "generic", model_id=model_id,
                        system_prompt=system_prompt, user_prompt=user_prompt,
                        temperature=temperature, max_tokens=max_tokens,
                        response_text="", elapsed_ms=elapsed_ms,
                        status="error", error=str(exc),
                    )
                return ""
    return ""


def send_structured_reasoning_request_to_llm(base_model: dict, question: str,
                                             logger=None, trace_logger=None) -> Dict[str, Any]:
    """Structured LLM call that returns separate reasoning steps and answer."""
    system_prompt = (
        "You are a helpful AI assistant that answers questions accurately and concisely.\n"
        "Please provide your reasoning process step by step before giving the final answer.\n"
        "Format your response as:\n"
        "REASONING PROCESS:\n[Step 1: ...]\n[Step 2: ...]\n\nFINAL ANSWER:\n[Your final answer here]"
    )
    raw_response = send_chat_completion_request_with_retry(base_model, system_prompt, question,
                                     temperature=0.7, max_tokens=2048,
                                     purpose="reasoning", logger=logger,
                                     trace_logger=trace_logger)
    if not raw_response:
        return {"reasoning_steps": ["Error: empty response"], "answer": ""}

    reasoning_steps = []
    final_answer = raw_response
    if "REASONING PROCESS:" in raw_response:
        parts = raw_response.split("REASONING PROCESS:")[1].split("FINAL ANSWER:")
        for line in parts[0].strip().split("\n"):
            if line.strip():
                reasoning_steps.append(line.strip())
        if len(parts) > 1:
            final_answer = parts[1].strip()
    return {"reasoning_steps": reasoning_steps, "answer": final_answer}


def get_knowledge_only_answer_from_llm(base_model: dict, question: str,
                                       logger=None, trace_logger=None) -> str:
    """LLM knowledge-only answer (no search evidence).

    Why max_tokens=2500: Qwen3-max often produces long reasoning chains for
    complex multi-hop questions.  With 1200 tokens the response was frequently
    truncated *before* the ``Answer:`` line, making knowledge_answer empty and
    disabling the knowledge-anchored dual-chain recovery.
    """
    system_prompt = (
        "You are an expert at answering complex multi-hop questions using your own knowledge.\n\n"
        "TASK: Reason step by step through the question, then give ONLY the final answer.\n\n"
        "RULES:\n"
        "1. Think through each clue in the question systematically\n"
        "2. Use your encyclopedic knowledge to identify entities, dates, events\n"
        "3. For Chinese questions, answer in Chinese; for English questions, answer in English\n"
        "4. Your final answer must be a specific name, number, year, or short phrase (1-15 words)\n"
        "5. NEVER say 'Unknown' or 'I don't know' — always give your best answer\n"
        "6. NEVER include explanations in your answer — just the answer itself\n"
        "7. If the question shows a format example (e.g. '格式形如：XXX'), it only shows the answer TYPE.\n"
        "   Do NOT copy the example's specific words. Use the EXACT official name from your knowledge.\n"
        "   For example: if example says 'Limited' but the real name uses 'Ltd', output 'Ltd'.\n\n"
        "IMPORTANT: Keep your reasoning concise (≤15 numbered steps). "
        "You MUST end with the Answer: line.\n\n"
        "FORMAT:\nReasoning: [your step-by-step reasoning]\nAnswer: [your concise answer]\n"
    )
    raw_response = send_chat_completion_request_with_retry(
        base_model, system_prompt, f"Question: {question}",
        temperature=0.0, max_tokens=2500,
        purpose="knowledge_answer", logger=logger,
        trace_logger=trace_logger)
    if not raw_response:
        return ""
    # Primary extraction: look for "Answer:" line
    if "Answer:" in raw_response:
        return raw_response.split("Answer:")[-1].strip().split("\n")[0].strip()
    # Fallback: truncated response — try to extract the best-guess entity from
    # the last few lines of reasoning, which typically contain the conclusion.
    lines = [line.strip() for line in raw_response.strip().split("\n") if line.strip()]
    if not lines:
        return raw_response.strip()
    # Look for a line that looks like a conclusion (mentions "answer", "therefore",
    # a proper noun, or is the last short line).
    import re as _re
    for line in reversed(lines[-5:]):
        # Check for explicit answer-like patterns in reasoning
        m = _re.search(
            r"(?:the answer is|answer[:\s]+|therefore[:\s,]+|因此|答案[是为：:]+)\s*(.+)",
            line, _re.IGNORECASE,
        )
        if m:
            candidate = m.group(1).strip().rstrip(".")
            if candidate and len(candidate) < 150:
                return candidate
    # Last resort: return last line if it's short enough to be an entity
    last = lines[-1]
    if len(last) < 150:
        return last
    return ""


def decompose_multi_hop_question_into_search_queries(base_model: dict, question: str,
                                                     logger=None, trace_logger=None) -> str:
    """Decompose multi-hop question into search queries."""
    system_prompt = (
        "你是多跳推理问题的搜索查询专家。\n\n"
        "## 任务\n"
        "给定一个复杂问题，你需要：\n"
        "1. 分析推理链：识别问题需要几跳推理（A→B→C 型）\n"
        "2. 生成第一跳搜索词：3-5 个精准的搜索查询，用于找到推理链的第一个中间实体\n\n"
        "## 关键规则\n"
        "- 每个查询 5-20 个词，必须包含问题中的专有名词、年份、技术术语\n"
        "- 禁止使用单个泛化词如 'author', 'spiritual', 'thesis'\n"
        "- 中文问题用中文查询，英文问题用英文查询\n"
        "- 对于多跳问题，第一跳的查询应聚焦于找到中间实体，而非最终答案\n\n"
        "## 输出格式\n只输出搜索查询，每行一个。不要编号、不要符号、不要解释。\n"
    )
    return send_chat_completion_request_with_retry(base_model, system_prompt, f"Question: {question}",
                            temperature=0.0, max_tokens=400,
                            purpose="decompose_question", logger=logger,
                            trace_logger=trace_logger)


def extract_concise_answer_from_evidence_using_llm(base_model: dict, question: str, evidence: str,
                                                   logger=None, trace_logger=None,
                                                   format_constraints: str = "") -> str:
    """Extract concise answer from question + evidence."""
    system_prompt = (
        "You are a precise question answering system. You MUST use the provided evidence AND your own knowledge to answer.\n\n"
        "ABSOLUTE RULES:\n"
        "1. Return ONLY the answer — a specific name, number, year, title, or short phrase\n"
        "2. NEVER return 'Unknown' or 'Cannot be determined' — ALWAYS give your best answer\n"
        "3. NEVER return generic role words like 'Author', 'Director' — return the ACTUAL name\n"
        "4. NEVER include explanations, reasoning, preamble, or sentences\n"
        "5. Your answer should typically be 1-10 words maximum\n\n"
        "FORMAT AWARENESS (CRITICAL):\n"
        "- If the question contains '格式形如' or a format example, it shows the GENERAL TYPE of answer expected\n"
        "- ALWAYS use the EXACT entity name as it appears in the web search evidence\n"
        "- The format example is just a style guide — do NOT copy its specific words into your answer\n"
        "- If the evidence says 'Ltd', output 'Ltd' — do NOT expand to 'Limited'\n"
        "- If the evidence says 'Limited', output 'Limited' — do NOT abbreviate to 'Ltd'\n"
        "- When a preliminary answer conflicts with evidence, ALWAYS prefer the form from evidence\n\n"
        "FORMAT RULES by question type:\n"
        "- Person name: Return full name exactly as in evidence\n"
        "- Company/org name: Return the EXACT COMPLETE official name from evidence — include ALL parts of the name\n"
        "- Number/count: Return ONLY the number\n- Year: Return ONLY 4-digit year\n"
        "- Chinese answer: Respond in Chinese matching the question's expected format\n"
        "- English answer: Respond in English matching the question's expected format\n"
    )
    user_prompt = f"Question: {question}\n\nEvidence:\n{evidence}\n\n"
    if format_constraints:
        user_prompt += f"{format_constraints}\n"
    user_prompt += "Answer (just the answer, nothing else):"
    return send_chat_completion_request_with_retry(base_model, system_prompt, user_prompt,
                            temperature=0.0, max_tokens=150,
                            purpose="extract_answer", logger=logger,
                            trace_logger=trace_logger)


def verify_answer_against_evidence_via_llm(base_model: dict, question: str, answer: str,
                      evidence: str, logger=None, trace_logger=None) -> Tuple[str, float]:
    """Verify answer against evidence with structured few-shot verification.

    Why (P0-c SelfVerifier enhancement, from Enhancing-Multi-Hop-QA):
    The old prompt was too simple (single-line "LABEL|CONFIDENCE").
    The enhanced prompt uses 3 few-shot examples covering SUPPORTS, REFUTES,
    and INSUFFICIENT verdicts, producing more calibrated confidence scores
    and better reasoning about whether the evidence actually supports the answer.

    Returns (label, confidence) where label is SUPPORTS/REFUTES/INSUFFICIENT.
    """
    system_prompt = (
        "You are a rigorous answer verification agent. Given a question, a proposed answer, "
        "and evidence, determine whether the evidence supports the answer.\n\n"
        "## Output Format\n"
        "Verdict: SUPPORTS | REFUTES | INSUFFICIENT\n"
        "Confidence: 0.00-1.00\n"
        "Reasoning: <one sentence explaining why>\n"
        "Better_entity: <if evidence mentions a DIFFERENT entity that better matches the question, name it here; otherwise leave empty>\n\n"
        "## Rules\n"
        "- SUPPORTS: Evidence clearly and directly confirms the answer (confidence >= 0.7)\n"
        "- REFUTES: Evidence clearly contradicts the answer or names a different entity (confidence >= 0.7)\n"
        "- INSUFFICIENT: Evidence does not mention the answer or is ambiguous (confidence < 0.7)\n"
        "- Pay special attention to EXACT entity names — 'RepRap' ≠ 'RepRapPro', '福临' ≠ '皇太极'\n"
        "- If the answer is a partial match of the correct entity, mark as REFUTES\n"
        "- CRITICAL: If the evidence mentions a DIFFERENT entity that better matches ALL constraints "
        "in the question, you MUST name it in Better_entity and mark as REFUTES\n\n"
        "## Examples\n"
        "Q: Who founded RepRapPro? A: Adrian Bowyer. Evidence: 'Adrian Bowyer founded RepRapPro Ltd in 2011'\n"
        "Verdict: SUPPORTS\nConfidence: 0.95\nReasoning: Evidence directly names Adrian Bowyer as founder\nBetter_entity:\n\n"
        "Q: What company did he create? A: RepRap Ltd. Evidence: 'He created RepRapPro Ltd'\n"
        "Verdict: REFUTES\nConfidence: 0.92\nReasoning: Evidence says RepRapPro Ltd, not RepRap Ltd — different entity\nBetter_entity: RepRapPro Ltd\n\n"
        "Q: When was it dissolved? A: 2016. Evidence: 'The company was established in Bath'\n"
        "Verdict: INSUFFICIENT\nConfidence: 0.20\nReasoning: Evidence mentions location but not dissolution date\nBetter_entity:\n"
    )
    user_prompt = (
        f"Question: {question}\n"
        f"Proposed Answer: {answer}\n"
        f"Evidence:\n{evidence[:6000]}\n\n"
        f"Provide your verdict:"
    )
    raw_response = send_chat_completion_request_with_retry(base_model, system_prompt, user_prompt,
                                     temperature=0.0, max_tokens=120,
                                     purpose="verify_answer", logger=logger,
                                     trace_logger=trace_logger)
    label, confidence, reasoning = "INSUFFICIENT", 0.0, ""
    if raw_response:
        # Try structured parsing first
        import re as _re_verify
        label_match = _re_verify.search(r"Verdict:\s*(SUPPORTS|REFUTES|INSUFFICIENT)", raw_response, _re_verify.IGNORECASE)
        conf_match = _re_verify.search(r"Confidence:\s*([\d.]+)", raw_response)
        reason_match = _re_verify.search(r"Reasoning:\s*(.+)", raw_response)
        if label_match:
            label = label_match.group(1).upper()
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except ValueError:
                pass
        if reason_match:
            reasoning = reason_match.group(1).strip()
        # Fallback to legacy pipe-delimited format
        if not label_match:
            parts = raw_response.strip().split("|")
            if parts:
                first = parts[0].strip().upper()
                if first in ("SUPPORTS", "REFUTES", "INSUFFICIENT"):
                    label = first
            if len(parts) > 1:
                try:
                    confidence = float(parts[1].strip())
                except ValueError:
                    pass
        # If no structured reasoning found, use the full response as context
        if not reasoning:
            reasoning = raw_response.strip()
    return label, confidence, reasoning


def fuse_multi_hop_evidence_via_llm(
    base_model: dict,
    question: str,
    hop_evidence_list: list,
    logger=None,
    trace_logger=None,
) -> str:
    """Fuse evidence from multiple hops into a coherent summary.

    Why (P0-c, from Research_Agent fuse_hop_evidence): Instead of naively
    concatenating all hop evidence, we use the LLM to filter noise, resolve
    contradictions, and produce a focused evidence summary that directly
    addresses the question.
    """
    system_prompt = (
        "You are an evidence synthesis expert. Given evidence collected from multiple "
        "search rounds for a complex question, produce a focused evidence summary.\n\n"
        "Rules:\n"
        "1. Merge overlapping information, remove duplicates\n"
        "2. Resolve contradictions by noting both claims with sources\n"
        "3. Keep EXACT names, dates, numbers — do not paraphrase proper nouns\n"
        "4. Remove irrelevant information that doesn't help answer the question\n"
        "5. Output a concise 200-500 word summary of the key facts\n"
        "6. Preserve the language of the evidence (Chinese stays Chinese, English stays English)\n"
    )
    evidence_parts = []
    for i, hop_ev in enumerate(hop_evidence_list, 1):
        evidence_parts.append(f"--- Hop {i} Evidence ---\n{hop_ev[:3000]}")
    user_prompt = (
        f"Question: {question}\n\n"
        + "\n\n".join(evidence_parts)
        + "\n\nSynthesize the key facts into a focused evidence summary:"
    )
    return send_chat_completion_request_with_retry(
        base_model, system_prompt, user_prompt,
        temperature=0.0, max_tokens=1500,
        purpose="evidence_fusion",
        logger=logger, trace_logger=trace_logger,
    )


def extract_corrected_entity_from_refutation_via_llm(
    base_model: dict,
    question: str,
    refutation_reasoning: str,
    logger=None,
    trace_logger=None,
) -> str:
    """Extract the correct entity from a REFUTES verification reasoning.

    Why (P0-f, from Enhancing-Multi-Hop-QA self_verification recovery):
    When verify_answer returns REFUTES, its reasoning often names the *correct*
    entity (e.g. "Evidence says RepRapPro Ltd, not RepRap Ltd").  Instead of
    discarding this intelligence, we extract the correct entity as a corrected
    answer — this is the highest-ROI recovery path since the verification
    stage already did the hard work of finding the truth.
    """
    system_prompt = (
        "A verification step found that a proposed answer is WRONG. "
        "The verification reasoning below names or implies the CORRECT answer.\n\n"
        "TASK: Extract the correct entity/answer from the reasoning text.\n\n"
        "RULES:\n"
        "1. Return ONLY the correct entity — a name, number, year, or short phrase\n"
        "2. Use the EXACT form as it appears in the reasoning (do not paraphrase)\n"
        "3. If the reasoning does not clearly name a correct entity, return EMPTY\n"
        "4. Your output must be 1-15 words maximum\n"
        "5. Do NOT include any explanation\n"
    )
    user_prompt = (
        f"Original question: {question}\n\n"
        f"Verification reasoning (answer was REFUTED):\n{refutation_reasoning}\n\n"
        f"Correct entity (or EMPTY if not found):"
    )
    raw = send_chat_completion_request_with_retry(
        base_model, system_prompt, user_prompt,
        temperature=0.0, max_tokens=60,
        purpose="extract_corrected_entity_from_refutation",
        logger=logger, trace_logger=trace_logger,
    )
    if not raw:
        return ""
    cleaned = raw.strip().strip('"').strip("'")
    # Reject if the LLM returned "EMPTY" or a refusal
    if cleaned.upper() in ("EMPTY", "NONE", "N/A", "UNKNOWN", ""):
        return ""
    if len(cleaned) > 200:
        return ""
    return cleaned


def arbitrate_among_candidate_answers_via_llm(
    base_model: dict,
    system_prompt: str,
    user_prompt: str,
    logger=None,
    trace_logger=None,
) -> str:
    """LLM arbitration: pick the best answer from multiple candidates.

    Why (P1-a): When consistency voting finds no consensus among candidate
    answers, this lightweight LLM call evaluates which candidate is most
    likely correct — inspired by MetaGPT ScEnsemble.
    """
    return send_chat_completion_request_with_retry(
        base_model, system_prompt, user_prompt,
        temperature=0.0, max_tokens=64,
        purpose="answer_arbitration",
        logger=logger, trace_logger=trace_logger,
    )
