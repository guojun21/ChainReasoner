"""LLM prompt factories for the answer pipeline.

Why: Each callback that ConstrainedMultiHopSearchAgent needs (knowledge,
decompose, extract, verify) has a specific system prompt.  These
factories adapt the generic LLMClient.chat() to those signatures
and keep prompt engineering in one place.
"""

from typing import Callable, Tuple

from src.llm.client import LLMClient
from src.agents.question_answer_format_hint_parsing_and_alignment import (
    build_format_sensitive_answer_constraints_for_prompt,
)


def create_knowledge_only_answer_callback(llm: LLMClient) -> Callable[[str], str]:
    """Callback: answer from LLM's own knowledge (no evidence)."""

    def knowledge_answer(question: str) -> str:
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
            "FORMAT:\n"
            "Reasoning: [your step-by-step reasoning]\n"
            "Answer: [your concise answer]\n"
        )
        raw_response = llm.chat(system_prompt, f"Question: {question}",
                                temperature=0.0, max_tokens=1200, purpose="knowledge_answer")
        if not raw_response:
            return ""
        if "Answer:" in raw_response:
            return raw_response.split("Answer:")[-1].strip().split("\n")[0].strip()
        lines = [line.strip() for line in raw_response.strip().split("\n") if line.strip()]
        return lines[-1] if lines else raw_response.strip()

    return knowledge_answer


def create_question_decomposition_callback(llm: LLMClient) -> Callable[[str], str]:
    """Callback: decompose a multi-hop question into search queries."""

    def decompose_question(question: str) -> str:
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
            "- 思考哪个 Wikipedia 条目或权威来源能回答这个问题\n"
            "- 对于多跳问题，第一跳的查询应聚焦于找到中间实体，而非最终答案\n\n"
            "## 输出格式\n"
            "只输出搜索查询，每行一个。不要编号、不要符号、不要解释。\n"
        )
        return llm.chat(system_prompt, f"Question: {question}",
                        temperature=0.0, max_tokens=400, purpose="decompose_question")

    return decompose_question


def create_evidence_based_answer_extraction_callback(llm: LLMClient) -> Callable[[str, str], str]:
    """Callback: extract a concise answer from question + evidence."""

    def extract_answer(question: str, evidence: str) -> str:
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
            "- Number/count: Return ONLY the number\n"
            "- Year: Return ONLY 4-digit year\n"
            "- Chinese answer: Respond in Chinese matching the question's expected format\n"
            "- English answer: Respond in English matching the question's expected format\n"
        )
        format_constraints = build_format_sensitive_answer_constraints_for_prompt(question)
        user_prompt = f"Question: {question}\n\nEvidence:\n{evidence}\n\n{format_constraints}\nAnswer (just the answer, nothing else):"
        return llm.chat(system_prompt, user_prompt, temperature=0.0, max_tokens=150, purpose="extract_answer")

    return extract_answer


def create_answer_verification_callback(llm: LLMClient) -> Callable[[str, str, str], Tuple[str, float]]:
    """Callback: verify an answer against evidence (SUPPORTS/REFUTES/INSUFFICIENT)."""

    def verify_answer(question: str, answer: str, evidence: str) -> Tuple[str, float]:
        system_prompt = (
            "You are a verifier. Given question, answer, and evidence, respond with "
            "one label SUPPORTS, REFUTES, or INSUFFICIENT and a confidence 0-1. "
            "Format: LABEL|CONFIDENCE"
        )
        user_prompt = f"Question: {question}\nAnswer: {answer}\nEvidence:\n{evidence}\nVerdict:"
        raw_response = llm.chat(system_prompt, user_prompt, temperature=0.0, max_tokens=32, purpose="verify_answer")
        label = "INSUFFICIENT"
        confidence = 0.0
        if raw_response:
            parts = raw_response.strip().split("|")
            if parts:
                label = parts[0].strip().upper()
            if len(parts) > 1:
                try:
                    confidence = float(parts[1].strip())
                except ValueError:
                    pass
        return label, confidence

    return verify_answer
