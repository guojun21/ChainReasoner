"""
Standalone answer pipeline for ChainReasoner.

This module decouples the core answer logic from Flask / api_server.py so that
it can be called from:
  - The existing Flask server  (``api_server.py``)
  - The evaluation script      (``run_progressive_eval.py``)
  - A future LangStudio Python node

Usage::

    from src.pipeline import build_pipeline, answer_question

    pipeline = build_pipeline()            # uses default configs
    result   = answer_question(pipeline, "What is ...?")
    print(result["answer"])
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import yaml

from src.agents.constrained_search import ConstrainedSearchAgent
from src.llm.client import LLMClient, OpenAICompatibleClient
from src.search.client import BraveSearchClient, HybridSearchClient, IQSSearchClient, SearchClient

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
DEFAULT_MCP_CONFIG_PATH = BASE_DIR / "configs" / "mcp_config.json"


# ---------------------------------------------------------------------------
# LLM prompt helpers (extracted from api_server.py for reuse)
# ---------------------------------------------------------------------------

def _make_knowledge_fn(llm: LLMClient) -> Callable[[str], str]:
    """Build the ``llm_knowledge_fn`` callback for ConstrainedSearchAgent."""

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
            "6. NEVER include explanations in your answer — just the answer itself\n\n"
            "FORMAT:\n"
            "Reasoning: [your step-by-step reasoning]\n"
            "Answer: [your concise answer]\n"
        )
        raw = llm.chat(system_prompt, f"Question: {question}", temperature=0.0, max_tokens=500, purpose="knowledge_answer")
        if not raw:
            return ""
        if "Answer:" in raw:
            answer_part = raw.split("Answer:")[-1].strip().split("\n")[0].strip()
            return answer_part
        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
        return lines[-1] if lines else raw.strip()

    return knowledge_answer


def _make_decompose_fn(llm: LLMClient) -> Callable[[str], str]:
    """Build the ``llm_decompose_fn`` callback."""

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
        return llm.chat(system_prompt, f"Question: {question}", temperature=0.0, max_tokens=400, purpose="decompose_question")

    return decompose_question


def _make_extract_fn(llm: LLMClient) -> Callable[[str, str], str]:
    """Build the ``llm_answer_fn`` callback."""

    def extract_answer(question: str, evidence: str) -> str:
        system_prompt = (
            "You are a precise question answering system. You MUST use the provided evidence AND your own knowledge to answer.\n\n"
            "ABSOLUTE RULES:\n"
            "1. Return ONLY the answer — a specific name, number, year, title, or short phrase\n"
            "2. NEVER return 'Unknown' or 'Cannot be determined' — ALWAYS give your best answer\n"
            "3. NEVER return generic role words like 'Author', 'Director' — return the ACTUAL name\n"
            "4. NEVER include explanations, reasoning, preamble, or sentences\n"
            "5. Your answer should typically be 1-10 words maximum\n\n"
            "FORMAT RULES by question type:\n"
            "- Person name: Return full name\n"
            "- Company/org name: Return official name\n"
            "- Number/count: Return ONLY the number\n"
            "- Year: Return ONLY 4-digit year\n"
            "- Chinese answer: Respond in Chinese matching the question's expected format\n"
            "- English answer: Respond in English matching the question's expected format\n"
        )
        user_prompt = f"Question: {question}\n\nEvidence:\n{evidence}\n\nAnswer (just the answer, nothing else):"
        return llm.chat(system_prompt, user_prompt, temperature=0.0, max_tokens=150, purpose="extract_answer")

    return extract_answer


def _make_verify_fn(llm: LLMClient) -> Callable[[str, str, str], Tuple[str, float]]:
    """Build the ``llm_verify_fn`` callback."""

    def verify_answer(question: str, answer: str, evidence: str) -> Tuple[str, float]:
        system_prompt = (
            "You are a verifier. Given question, answer, and evidence, respond with "
            "one label SUPPORTS, REFUTES, or INSUFFICIENT and a confidence 0-1. "
            "Format: LABEL|CONFIDENCE"
        )
        user_prompt = f"Question: {question}\nAnswer: {answer}\nEvidence:\n{evidence}\nVerdict:"
        raw = llm.chat(system_prompt, user_prompt, temperature=0.0, max_tokens=32, purpose="verify_answer")
        label = "INSUFFICIENT"
        confidence = 0.0
        if raw:
            parts = raw.strip().split("|")
            if parts:
                label = parts[0].strip().upper()
            if len(parts) > 1:
                try:
                    confidence = float(parts[1].strip())
                except ValueError:
                    pass
        return label, confidence

    return verify_answer


# ---------------------------------------------------------------------------
# Pipeline data class
# ---------------------------------------------------------------------------

class Pipeline:
    """Holds all components needed to answer a question."""

    def __init__(
        self,
        search_agent: ConstrainedSearchAgent,
        llm: LLMClient,
        search: SearchClient,
    ):
        self.search_agent = search_agent
        self.llm = llm
        self.search = search


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_pipeline(
    config_path: Optional[str] = None,
    mcp_config_path: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    search_client: Optional[SearchClient] = None,
) -> Pipeline:
    """Build a fully-wired Pipeline from config files.

    You can override ``llm_client`` and/or ``search_client`` for testing
    or LangStudio migration.
    """
    # --- load configs ---
    cfg_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not cfg_file.is_absolute():
        cfg_file = BASE_DIR / cfg_file
    with open(cfg_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    mcp_file = Path(mcp_config_path) if mcp_config_path else DEFAULT_MCP_CONFIG_PATH
    if not mcp_file.is_absolute():
        mcp_file = BASE_DIR / mcp_file
    mcp_config: dict = {}
    if mcp_file.exists():
        with open(mcp_file, "r", encoding="utf-8") as f:
            mcp_config = json.load(f)

    # --- LLM ---
    if llm_client is None:
        llm_client = OpenAICompatibleClient.from_config(config.get("base_model", {}))
        if llm_client is None:
            raise RuntimeError("Cannot create LLM client — check base_model in config.yaml")

    # --- Search (Hybrid: Brave MCP for English, IQS for Chinese) ---
    if search_client is None:
        iqs_client = IQSSearchClient.from_mcp_config(mcp_config)
        if iqs_client is None:
            raise RuntimeError("Cannot create IQS search client — check mcp_config.json")
        # Prefer MCP-based Brave client, fall back to config-based (HTTP-only)
        brave_client = BraveSearchClient.from_mcp_config(mcp_config)
        if brave_client is None:
            brave_client = BraveSearchClient.from_config(config)
        if brave_client:
            logger.info("Brave search available — using HybridSearchClient (EN→Brave MCP, ZH→IQS)")
            search_client = HybridSearchClient(iqs=iqs_client, brave=brave_client)
        else:
            logger.info("No Brave key — using IQS only")
            search_client = iqs_client

    # --- ConstrainedSearchAgent ---
    search_cfg = config.get("search_agent", {}) if isinstance(config, dict) else {}

    agent = ConstrainedSearchAgent(
        search_fn=search_client.search,
        max_queries=max(search_cfg.get("max_queries", 5), 5),
        per_query_delay=search_cfg.get("per_query_delay", 0.2),
        max_results_per_query=max(search_cfg.get("max_results_per_query", 10), 10),
        max_evidence=max(search_cfg.get("max_evidence", 12), 12),
        adaptive_threshold_n=search_cfg.get("adaptive_threshold_n", 0.5),
        rewrite_fn=None,
        llm_answer_fn=_make_extract_fn(llm_client),
        llm_verify_fn=_make_verify_fn(llm_client),
        llm_decompose_fn=_make_decompose_fn(llm_client),
        llm_knowledge_fn=_make_knowledge_fn(llm_client),
    )

    return Pipeline(search_agent=agent, llm=llm_client, search=search_client)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def answer_question(pipeline: Pipeline, question: str, use_mcp: bool = True) -> Dict[str, Any]:
    """Answer a single question using the pipeline.

    Returns
    -------
    dict
        ``{"question": str, "answer": str, "reasoning_steps": list, ...}``
    """
    start = time.time()
    logger.info("Pipeline: question=%s", question[:100])

    if use_mcp:
        result = pipeline.search_agent.answer(question)
    else:
        # Fallback: LLM-only (no search)
        knowledge_fn = _make_knowledge_fn(pipeline.llm)
        answer_text = knowledge_fn(question) or "Unknown"
        result = {"answer": answer_text, "reasoning_steps": ["LLM-only"], "search_traces": []}

    elapsed = time.time() - start
    logger.info("Pipeline: answer=%s elapsed=%.1fs", result.get("answer", "")[:60], elapsed)

    return {
        "question": question,
        "answer": result.get("answer", "Unknown"),
        "reasoning_steps": result.get("reasoning_steps", []),
        "mcp_results": result.get("search_traces", []),
        "use_mcp": use_mcp,
        "timestamp": datetime.now().isoformat(),
    }
