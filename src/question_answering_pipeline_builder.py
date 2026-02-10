"""Standalone answer pipeline — decoupled from Flask for reuse.

Why: The same reasoning logic must run in Flask (api_server.py),
evaluation (run_progressive_eval.py), and future LangStudio nodes.
This module wires LLM + search + agent without any web framework.

Usage::

    from src.question_answering_pipeline_builder import build_question_answering_pipeline, answer_single_question
    pipeline = build_question_answering_pipeline()
    result   = answer_single_question(pipeline, "What is ...?")
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.agents.constrained_multi_hop_search_agent import ConstrainedMultiHopSearchAgent
from src.llm.client import LLMClient, OpenAICompatibleClient
from src.search.brave_web_search_client import BraveWebSearchClient
from src.search.google_custom_search_api_client import GoogleCustomSearchApiClient
from src.search.language_aware_hybrid_search_dispatcher import LanguageAwareHybridSearchDispatcher
from src.search.alibaba_iqs_search_client import AlibabaIQSSearchClient
from src.search.abstract_search_client_interface import AbstractSearchClientInterface
from src.pipeline_llm_prompt_factory_functions import (
    create_knowledge_only_answer_callback,
    create_question_decomposition_callback,
    create_evidence_based_answer_extraction_callback,
    create_answer_verification_callback,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
DEFAULT_MCP_CONFIG_PATH = BASE_DIR / "configs" / "mcp_config.json"


class QuestionAnsweringPipeline:
    """Holds all wired components needed to answer a question."""

    def __init__(self, search_agent: ConstrainedMultiHopSearchAgent,
                 llm: LLMClient, search: AbstractSearchClientInterface):
        self.search_agent = search_agent
        self.llm = llm
        self.search = search


def build_question_answering_pipeline(
    config_path: Optional[str] = None,
    mcp_config_path: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    search_client: Optional[AbstractSearchClientInterface] = None,
) -> QuestionAnsweringPipeline:
    """Factory: load configs, create clients, wire the agent.

    Override ``llm_client`` / ``search_client`` for testing or LangStudio.
    """
    config = _load_yaml(config_path or DEFAULT_CONFIG_PATH)
    mcp_config = _load_json(mcp_config_path or DEFAULT_MCP_CONFIG_PATH)

    if llm_client is None:
        llm_client = OpenAICompatibleClient.from_config(config.get("base_model", {}))
        if llm_client is None:
            raise RuntimeError("Cannot create LLM client — check base_model in config.yaml")

    if search_client is None:
        search_client = _build_hybrid_search(config, mcp_config)

    search_config = config.get("search_agent", {}) if isinstance(config, dict) else {}
    agent = ConstrainedMultiHopSearchAgent(
        search_fn=search_client.execute_search_query,
        max_queries=max(search_config.get("max_queries", 5), 5),
        per_query_delay=search_config.get("per_query_delay", 0.2),
        max_results_per_query=max(search_config.get("max_results_per_query", 10), 10),
        max_evidence=max(search_config.get("max_evidence", 12), 12),
        adaptive_threshold_n=search_config.get("adaptive_threshold_n", 0.5),
        llm_answer_fn=create_evidence_based_answer_extraction_callback(llm_client),
        llm_verify_fn=create_answer_verification_callback(llm_client),
        llm_decompose_fn=create_question_decomposition_callback(llm_client),
        llm_knowledge_fn=create_knowledge_only_answer_callback(llm_client),
    )
    return QuestionAnsweringPipeline(search_agent=agent, llm=llm_client, search=search_client)


def answer_single_question(pipeline: QuestionAnsweringPipeline, question: str, use_mcp: bool = True) -> Dict[str, Any]:
    """Answer a single question; returns {question, answer, reasoning_steps, ...}."""
    start_time = time.time()
    logger.info("Pipeline: question=%s", question[:100])

    if use_mcp:
        result = pipeline.search_agent.answer(question)
    else:
        knowledge_fn = create_knowledge_only_answer_callback(pipeline.llm)
        answer_text = knowledge_fn(question) or "Unknown"
        result = {"answer": answer_text, "reasoning_steps": ["LLM-only"], "search_traces": []}

    elapsed = time.time() - start_time
    logger.info("Pipeline: answer=%s elapsed=%.1fs", result.get("answer", "")[:60], elapsed)
    return {
        "question": question,
        "answer": result.get("answer", "Unknown"),
        "reasoning_steps": result.get("reasoning_steps", []),
        "mcp_results": result.get("search_traces", []),
        "use_mcp": use_mcp,
        "timestamp": datetime.now().isoformat(),
    }


# ── Internal helpers ────────────────────────────────────────────────────

def _load_yaml(path) -> dict:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = BASE_DIR / file_path
    with open(file_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_json(path) -> dict:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = BASE_DIR / file_path
    if not file_path.exists():
        return {"mcpServers": {}}
    with open(file_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_hybrid_search(config: dict, mcp_config: dict) -> AbstractSearchClientInterface:
    """Wire up IQS + Google + Brave into a LanguageAwareHybridSearchDispatcher."""
    iqs_client = AlibabaIQSSearchClient.from_mcp_config(mcp_config)
    if iqs_client is None:
        raise RuntimeError("Cannot create IQS search client — check mcp_config.json")
    google_client = GoogleCustomSearchApiClient.from_mcp_config(mcp_config)
    brave_client = BraveWebSearchClient.from_mcp_config(mcp_config) or BraveWebSearchClient.from_config(config)
    if google_client and brave_client:
        logger.info("Hybrid search: EN -> Google -> Brave -> IQS, ZH -> IQS -> Google")
    elif google_client:
        logger.info("Hybrid search: EN -> Google -> IQS, ZH -> IQS -> Google")
    elif brave_client:
        logger.info("Hybrid search: EN -> Brave -> IQS, ZH -> IQS")
    else:
        logger.info("IQS only (no Google/Brave key)")
    return LanguageAwareHybridSearchDispatcher(iqs=iqs_client, brave=brave_client, google=google_client)
