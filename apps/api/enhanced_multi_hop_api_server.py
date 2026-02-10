#!/usr/bin/env python3
"""Flask API server with multi-hop reasoning and MCP integration.

Why: This is the HTTP entry point for the competition — it receives
questions via POST and returns answers.  Also used by the eval script
which calls ``server._multi_hop_reasoning(question, use_mcp=True)``.
"""

import json
import re as _re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import requests
import yaml
from flask import Flask, Response, jsonify, request, stream_with_context

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
DEFAULT_MCP_CONFIG_PATH = BASE_DIR / "configs" / "mcp_config.json"
INDEX_HTML_PATH = BASE_DIR / "apps" / "web" / "index.html"
sys.path.insert(0, str(BASE_DIR))

from src.utils.logger_config import get_logger, MultiHopLogger
from src.agents.constrained_multi_hop_search_agent import ConstrainedMultiHopSearchAgent
from src.search.brave_web_search_client import BraveWebSearchClient
from src.search.google_custom_search_api_client import GoogleCustomSearchApiClient
from src.search.duckduckgo_web_search_client import DuckDuckGoWebSearchClient
from src.search.language_aware_hybrid_search_dispatcher import LanguageAwareHybridSearchDispatcher
from src.search.alibaba_iqs_search_client import AlibabaIQSSearchClient

from apps.api.large_language_model_call_handlers import (
    send_chat_completion_request_with_retry,
    send_structured_reasoning_request_to_llm,
    get_knowledge_only_answer_from_llm,
    decompose_multi_hop_question_into_search_queries,
    extract_concise_answer_from_evidence_using_llm,
    verify_answer_against_evidence_via_llm,
    arbitrate_among_candidate_answers_via_llm,
)
from src.agents.question_answer_format_hint_parsing_and_alignment import (
    build_format_sensitive_answer_constraints_for_prompt,
    apply_format_aware_answer_postprocessing_pipeline,
)
from apps.api.mcp_service_dispatch_and_stdio_callers import dispatch_mcp_service_call_to_appropriate_transport
from apps.api.legacy_search_api_handlers_and_logging import (
    resolve_brave_api_key_from_env_or_config,
    redact_api_key_middle_portion_for_logging,
    append_search_trace_record_to_jsonl_audit_log,
    call_iqs_common_search_via_legacy_mcp_path,
    call_brave_search_api_with_rate_limiting,
)


# ── Legacy MCP HTTP client (kept for _call_mcp_service HTTP path) ───────

class MCPHttpClient:
    """HTTP MCP client — JSON-RPC 2.0 over HTTP for cloud MCP services."""

    def __init__(self, url: str, api_key: str, logger, *, timeout: int = 30):
        self.url = url
        self.api_key = api_key
        self.logger = logger
        self.timeout = timeout
        self._req_id = 0
        self._lock = threading.Lock()
        self._initialized = False

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _build_headers(self) -> dict:
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _ensure_initialized(self) -> None:
        """JSON-RPC handshake — thread-safe, runs once."""
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            try:
                response = requests.post(self.url, headers=self._build_headers(), json={
                    "jsonrpc": "2.0", "id": self._next_id(), "method": "initialize",
                    "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                               "clientInfo": {"name": "ChainReasoner", "version": "1.0.0"}},
                }, timeout=15)
                response.raise_for_status()
                requests.post(self.url, headers=self._build_headers(),
                              json={"jsonrpc": "2.0", "method": "notifications/initialized"}, timeout=10)
                self._initialized = True
                self.logger.info("MCP HTTP initialized: %s", self.url)
            except Exception as exc:
                self.logger.error("MCP HTTP init failed (%s): %s", self.url, exc)

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Invoke an MCP tool; returns concatenated text content."""
        self._ensure_initialized()
        try:
            response = requests.post(self.url, headers=self._build_headers(), json={
                "jsonrpc": "2.0", "id": self._next_id(), "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }, timeout=self.timeout)
            response.raise_for_status()
            body = response.json()
            content_items = body.get("result", {}).get("content", [])
            return "\n".join(c.get("text", "") for c in content_items if c.get("type") == "text")
        except Exception as exc:
            self.logger.error("MCP call_tool(%s) error: %s", tool_name, exc)
            self._initialized = False
            return ""


def _load_mcp_http_clients(mcp_config: dict, logger) -> Dict[str, MCPHttpClient]:
    """Build MCPHttpClient instances for all HTTP-type MCP servers in config."""
    clients: Dict[str, MCPHttpClient] = {}
    for name, cfg in mcp_config.get("mcpServers", {}).items():
        if cfg.get("type") != "http" or not cfg.get("url"):
            continue
        clients[name] = MCPHttpClient(cfg["url"], cfg.get("api_key", ""), logger)
        logger.info("MCP HTTP client registered: %s -> %s", name, cfg["url"])
    return clients


def _parse_search_markdown(markdown_text: str) -> list:
    """Parse IQS markdown into [{title, url, content}]."""
    results = []
    for block in _re.split(r"\n---\n?", markdown_text):
        block = block.strip()
        if not block:
            continue
        title = url = content = ""
        match = _re.search(r"##\s*(?:标题\s*)?(.+)", block)
        if match:
            title = match.group(1).strip()
        match = _re.search(r"\*\*url\*\*[：:]\s*(https?://\S+)", block)
        if match:
            url = match.group(1).strip()
        match = _re.search(r"\*\*摘要\*\*[：:]\s*(.+)", block, _re.DOTALL)
        if match:
            content = _re.split(r"\n-\s*\*\*", match.group(1).strip())[0].strip()
        if title or content:
            results.append({"title": title, "url": url, "content": content})
    return results


# ── Main server class ───────────────────────────────────────────────────

class EnhancedMultiHopReasoningApiServer:
    """Flask server: receives questions, runs multi-hop reasoning, returns answers."""

    def __init__(self, config_path: Optional[str] = None, trace_logger=None):
        self.logger = get_logger("api", "api.log")
        self.logger.info("=" * 70)
        self.logger.info("Enhanced MultiHop API Server - Starting")
        self.logger.info("=" * 70)

        # Per-run faithful API call trace logger (optional, injected by eval scripts)
        self.trace_logger = trace_logger

        self.config = self._load_config(config_path)
        self.base_model = self.config.get("base_model", {})
        self.api_token = self.config.get("api_token", "default_token_123456")
        self.mcp_config = self._load_mcp_config()
        search_config = self.config.get("search_agent", {}) if isinstance(self.config, dict) else {}

        self._mcp_http = _load_mcp_http_clients(self.mcp_config, self.logger)
        self._brave_min_interval = min(max(search_config.get("brave_min_interval_seconds", 1.5), 1.2), 2.0)
        self._brave_last_call_ts = 0.0
        self._brave_lock = threading.Lock()
        self._hybrid_search = self._build_hybrid_search()
        self._last_search_backend_preflight: Dict[str, Any] = {}

        # Inject trace_logger into all search client components
        self._inject_trace_logger_into_search_components()
        # P0-b: Inject LLM refinement function into dispatcher for Layer 3 cleaning
        self._inject_llm_refine_into_dispatcher()

        self.search_agent = ConstrainedMultiHopSearchAgent(
            self._hybrid_search_fn,
            max_queries=max(search_config.get("max_queries", 5), 5),
            per_query_delay=search_config.get("per_query_delay", 0.2),
            max_results_per_query=max(search_config.get("max_results_per_query", 10), 10),
            max_evidence=max(search_config.get("max_evidence", 12), 12),
            adaptive_threshold_n=search_config.get("adaptive_threshold_n", 0.5),
            llm_answer_fn=lambda q, e: extract_concise_answer_from_evidence_using_llm(
                self.base_model, q, e, self.logger, trace_logger=self.trace_logger,
                format_constraints=build_format_sensitive_answer_constraints_for_prompt(q)),
            llm_verify_fn=lambda q, a, e: verify_answer_against_evidence_via_llm(
                self.base_model, q, a, e, self.logger, trace_logger=self.trace_logger),
            llm_decompose_fn=lambda q: decompose_multi_hop_question_into_search_queries(
                self.base_model, q, self.logger, trace_logger=self.trace_logger),
            llm_knowledge_fn=lambda q: get_knowledge_only_answer_from_llm(
                self.base_model, q, self.logger, trace_logger=self.trace_logger),
        )
        # P1-a: Inject LLM arbitration function + trace logger into search agent
        self.search_agent.llm_arbitrate_fn = lambda sys_p, usr_p: arbitrate_among_candidate_answers_via_llm(
            self.base_model, sys_p, usr_p, self.logger, trace_logger=self.trace_logger)
        self.search_agent.trace_logger = self.trace_logger
        self.app = Flask(__name__)
        self._setup_routes()
        self.logger.info("Model: %s", self.base_model.get("model_id", "unknown"))
        self.logger.info("MCP Services: %d available", len(self.mcp_config.get("mcpServers", {})))
        if self.trace_logger:
            self.logger.info("Trace logger: ACTIVE -> %s", self.trace_logger.run_directory)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        config_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if not config_file.is_absolute():
            config_file = BASE_DIR / config_file
        with open(config_file, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def _load_mcp_config(self) -> Dict[str, Any]:
        if DEFAULT_MCP_CONFIG_PATH.exists():
            with open(DEFAULT_MCP_CONFIG_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {"mcpServers": {}}

    def _inject_trace_logger_into_search_components(self) -> None:
        """Propagate the per-run trace logger into all search and MCP transport layers.

        Why: The trace logger is created externally (by the eval script) and must
        reach every component that makes external API calls, including nested MCP
        transport clients inside the search clients.
        """
        if not self.trace_logger:
            return
        dispatcher = self._hybrid_search
        if dispatcher is None:
            return
        # Set trace_logger on the dispatcher itself
        dispatcher.trace_logger = self.trace_logger
        # Set on IQS client and its MCP transport clients
        if hasattr(dispatcher, 'iqs') and dispatcher.iqs:
            dispatcher.iqs.trace_logger = self.trace_logger
            if hasattr(dispatcher.iqs, 'search_client') and dispatcher.iqs.search_client:
                dispatcher.iqs.search_client.trace_logger = self.trace_logger
            if hasattr(dispatcher.iqs, 'readpage_client') and dispatcher.iqs.readpage_client:
                dispatcher.iqs.readpage_client.trace_logger = self.trace_logger
        # Set on Brave client and its MCP stdio transport
        if hasattr(dispatcher, 'brave') and dispatcher.brave:
            dispatcher.brave.trace_logger = self.trace_logger
            if hasattr(dispatcher.brave, 'mcp_client') and dispatcher.brave.mcp_client:
                dispatcher.brave.mcp_client.trace_logger = self.trace_logger
        # Set on Google client
        if hasattr(dispatcher, 'google') and dispatcher.google:
            dispatcher.google.trace_logger = self.trace_logger
        # Set on DuckDuckGo client
        if hasattr(dispatcher, 'duckduckgo') and dispatcher.duckduckgo:
            dispatcher.duckduckgo.trace_logger = self.trace_logger

    def _inject_llm_refine_into_dispatcher(self) -> None:
        """P0-b: Inject LLM page-content refinement function into the search dispatcher.

        This enables Layer 3 of the three-layer cleaning pipeline when the
        dispatcher enriches search results with full page content.
        """
        dispatcher = self._hybrid_search
        if dispatcher is None:
            return
        dispatcher._llm_refine_fn = self._llm_refine_page_content
        dispatcher._enable_llm_refinement = True
        self.logger.info("P0-b: LLM page-content refinement injected into dispatcher")

    def _llm_refine_page_content(self, query: str, raw_text: str) -> str:
        """Layer 3 LLM refinement: extract query-relevant facts from page content.

        Uses a short, focused prompt to filter noise from cleaned web page text.
        Keeps temperature=0 for deterministic output.  Max 2000 tokens to stay fast.
        """
        system_prompt = (
            "You are a web content filter. Given a search query and web page text, "
            "extract ONLY the facts, names, dates, numbers, and statements directly "
            "relevant to the query. Remove all unrelated content, navigation text, "
            "and noise. Output the relevant information as concise paragraphs. "
            "If nothing relevant is found, output 'NO_RELEVANT_CONTENT'."
        )
        user_prompt = (
            f"Search query: {query}\n\n"
            f"Web page content:\n{raw_text[:8000]}\n\n"
            f"Extract only the facts relevant to the search query above:"
        )
        result = send_chat_completion_request_with_retry(
            base_model=self.base_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=2000,
            purpose="page_content_refinement",
            logger=self.logger,
            trace_logger=self.trace_logger,
        )
        if result and "NO_RELEVANT_CONTENT" not in result:
            return result
        return raw_text

    def _build_hybrid_search(self) -> LanguageAwareHybridSearchDispatcher:
        """Wire IQS + Google + Brave + DuckDuckGo into LanguageAwareHybridSearchDispatcher."""
        iqs = AlibabaIQSSearchClient.from_mcp_config(self.mcp_config)
        google = GoogleCustomSearchApiClient.from_mcp_config(self.mcp_config)
        brave = BraveWebSearchClient.from_mcp_config(self.mcp_config) or BraveWebSearchClient.from_config(self.config)
        duckduckgo = DuckDuckGoWebSearchClient(timeout=15)
        engines = []
        if google:
            engines.append("Google")
        if brave:
            engines.append("Brave")
        engines.append("DuckDuckGo")
        engines.append("IQS")
        self.logger.info("Hybrid mode: EN -> %s, ZH -> IQS -> %s",
                         " -> ".join(engines),
                         " -> ".join([e for e in engines if e != "IQS"]))
        return LanguageAwareHybridSearchDispatcher(
            iqs=iqs, brave=brave, google=google, duckduckgo=duckduckgo)

    def _hybrid_search_fn(self, query: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Primary search dispatch — routes through LanguageAwareHybridSearchDispatcher."""
        if self._hybrid_search:
            return self._hybrid_search.execute_search_query(query, meta)
        return call_iqs_common_search_via_legacy_mcp_path(query, self._mcp_http, meta, self.logger)

    def run_search_backend_preflight_checks(self) -> Dict[str, Any]:
        """Probe search backends once and disable unavailable engines.

        Why: If a backend is known unavailable (quota exhausted, auth failure,
        transport failure), repeatedly trying it for every query only adds
        noise and latency. We detect this at startup and skip that backend.
        """
        dispatcher = self._hybrid_search
        if not dispatcher:
            summary = {"status": "no_dispatcher", "engines": {}}
            self._last_search_backend_preflight = summary
            return summary

        self.logger.info("Search backend preflight: starting")
        checks = [
            ("google", "test query"),
            ("brave", "open source hardware project"),
            ("iqs", "北京 天气"),
        ]
        engines: Dict[str, Any] = {}

        for engine_name, probe_query in checks:
            client = getattr(dispatcher, engine_name, None)
            if client is None:
                dispatcher.set_engine_enabled(engine_name, False, reason="not configured")
                engines[engine_name] = {
                    "configured": False,
                    "enabled": False,
                    "status": "not_configured",
                    "reason": "not configured",
                }
                continue
            try:
                result = client.execute_search_query(
                    probe_query,
                    {"preflight_check": True, "source": "run_search_backend_preflight_checks"},
                )
                error = str(result.get("error", "")).strip()
                error_code = str(result.get("error_code", "")).strip()
                unavailable = bool(error)
                if unavailable:
                    reason = error_code or error
                    dispatcher.set_engine_enabled(engine_name, False, reason=reason)
                    status = "disabled"
                else:
                    dispatcher.set_engine_enabled(engine_name, True)
                    status = "enabled"
                engines[engine_name] = {
                    "configured": True,
                    "enabled": bool(dispatcher.engine_enabled.get(engine_name, False)),
                    "status": status,
                    "reason": error if unavailable else "",
                    "error_code": error_code,
                    "probe_service": result.get("service", ""),
                    "probe_count": int(result.get("count", 0)),
                }
            except Exception as exc:
                dispatcher.set_engine_enabled(engine_name, False, reason=str(exc))
                engines[engine_name] = {
                    "configured": True,
                    "enabled": False,
                    "status": "disabled",
                    "reason": str(exc),
                    "error_code": "preflight_exception",
                }

        # Also report DDG status (not actively probed by requirement).
        ddg_configured = bool(getattr(dispatcher, "duckduckgo", None))
        engines["duckduckgo"] = {
            "configured": ddg_configured,
            "enabled": bool(dispatcher.engine_enabled.get("duckduckgo", False)),
            "status": "enabled" if dispatcher.engine_enabled.get("duckduckgo", False) else "disabled",
            "reason": dispatcher.engine_disable_reasons.get("duckduckgo", ""),
            "note": "not preflight probed",
        }

        snapshot = dispatcher.get_engine_status_snapshot()
        summary = {"status": "ok", "engines": engines, "snapshot": snapshot}
        self._last_search_backend_preflight = summary

        for name in ("google", "brave", "iqs", "duckduckgo"):
            info = engines.get(name, {})
            self.logger.info(
                "Preflight %-10s configured=%s enabled=%s status=%s reason=%s",
                name,
                info.get("configured"),
                info.get("enabled"),
                info.get("status"),
                (info.get("reason", "") or "-")[:160],
            )
        return summary

    def _multi_hop_reasoning(self, question: str, use_mcp: bool = False) -> Dict[str, Any]:
        """Run full multi-hop reasoning pipeline on a question."""
        start_time = time.time()
        self.logger.info("=" * 70)
        self.logger.info("Multi-Hop Reasoning - Starting")
        self.logger.info("Question: %s...", question[:100])
        self.logger.info("MCP Enabled: %s", use_mcp)
        self.logger.info("=" * 70)

        if use_mcp:
            self.logger.info("Multi-Hop Step 1: Constrained search pipeline")
            search_result = self.search_agent.answer(question)
            reasoning_steps = ["Step 1: Constrained search pipeline"] + search_result.get("reasoning_steps", [])
            mcp_results = search_result.get("search_traces", [])
            final_answer = search_result.get("answer", "Unknown")
        else:
            self.logger.info("Multi-Hop Step 1: LLM analysis")
            llm_result = send_structured_reasoning_request_to_llm(
                self.base_model, question, self.logger, trace_logger=self.trace_logger)
            reasoning_steps = ["Step 1: LLM analysis"] + llm_result.get("reasoning_steps", [])
            mcp_results = []
            final_answer = llm_result.get("answer", "")

        # P0 optimization: apply format-aware answer post-processing
        raw_before_postprocess = final_answer
        final_answer, postprocess_trace = apply_format_aware_answer_postprocessing_pipeline(
            final_answer, question)
        if raw_before_postprocess != final_answer:
            reasoning_steps.append(
                f"Format post-process: '{raw_before_postprocess}' → '{final_answer}'")
            self.logger.info("Answer post-processed: '%s' → '%s'",
                             raw_before_postprocess, final_answer)
        if self.trace_logger:
            self.trace_logger.record_answer_postprocess_trace(
                question_text=question[:200],
                raw_answer=raw_before_postprocess,
                final_answer=final_answer,
                trace=postprocess_trace,
            )

        self.logger.info("Multi-Hop Step 2: Synthesizing final answer")
        reasoning_steps.append("Step 2: Synthesizing final answer")
        duration = time.time() - start_time
        self.logger.info("Multi-Hop Reasoning - Completed (Duration: %.2fs)", duration)
        self.logger.info("Total reasoning steps: %d", len(reasoning_steps))
        self.logger.info("MCP results: %d", len(mcp_results))
        return {
            "question": question, "answer": final_answer,
            "reasoning_steps": reasoning_steps,
            "mcp_results": mcp_results if use_mcp else [],
            "use_mcp": use_mcp, "timestamp": datetime.now().isoformat(),
        }

    def _generate_event_stream(self, question: str, use_mcp: bool = False) -> Generator[str, None, None]:
        """SSE stream: reasoning steps -> MCP results -> final answer."""
        result = self._multi_hop_reasoning(question, use_mcp)
        for idx, step in enumerate(result.get("reasoning_steps", []), 1):
            yield f"data: {json.dumps({'type': 'reasoning', 'step': idx, 'content': step}, ensure_ascii=False)}\n\n"
            time.sleep(0.3)
        if result.get("mcp_results"):
            yield f"data: {json.dumps({'type': 'mcp_results', 'results': result['mcp_results']}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'answer', 'answer': result['answer'], 'use_mcp': use_mcp, 'timestamp': datetime.now().isoformat()}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    def _setup_routes(self) -> None:
        """Register all Flask routes."""

        @self.app.after_request
        def add_cors_headers(response):
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,Accept"
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
            return response

        @self.app.route("/", methods=["GET"])
        def index():
            try:
                return open(INDEX_HTML_PATH, "r", encoding="utf-8").read()
            except Exception:
                return jsonify({"error": "Failed to load web interface"}), 500

        @self.app.route("/health", methods=["GET"])
        def health_check():
            return jsonify({"status": "healthy", "service": "Enhanced MultiHop Agent API",
                            "mcp_services": list(self.mcp_config.get("mcpServers", {}).keys())})

        @self.app.route("/api/v1/answer", methods=["POST"])
        def answer_endpoint():
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth.replace("Bearer ", "") != self.api_token:
                return jsonify({"error": "Unauthorized"}), 401
            data = request.get_json()
            if not data or "question" not in data:
                return jsonify({"error": "Missing 'question'"}), 400
            question = data["question"]
            use_mcp = data.get("use_mcp", False)
            if "text/event-stream" in request.headers.get("Accept", ""):
                return Response(stream_with_context(self._generate_event_stream(question, use_mcp)),
                                mimetype="text/event-stream",
                                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})
            return jsonify(self._multi_hop_reasoning(question, use_mcp))

        @self.app.route("/api/v1/mcp/list", methods=["GET"])
        def mcp_list():
            services = self.mcp_config.get("mcpServers", {})
            return jsonify({"mcp_services": services, "count": len(services)})

        @self.app.route("/api/v1/mcp/call", methods=["POST"])
        def mcp_call():
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth.replace("Bearer ", "") != self.api_token:
                return jsonify({"error": "Unauthorized"}), 401
            data = request.get_json()
            if not data or "service" not in data or "query" not in data:
                return jsonify({"error": "Bad Request"}), 400
            return jsonify(dispatch_mcp_service_call_to_appropriate_transport(
                data["service"], data["query"],
                self.mcp_config, self._mcp_http, self.logger))

    def run(self, host: str = "0.0.0.0", port: int = 5000, ssl_context=None) -> None:
        """Start the Flask server."""
        print(f"\nEnhanced MultiHop API Server on {host}:{port}")
        print(f"Model: {self.base_model.get('model_id', 'unknown')}")
        print(f"MCP Services: {len(self.mcp_config.get('mcpServers', {}))}")
        self.app.run(host=host, port=port, ssl_context=ssl_context, threaded=True)


def main():
    server = EnhancedMultiHopReasoningApiServer()
    try:
        import ssl
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain("server.crt", "server.key")
    except Exception:
        ssl_context = None
    server.run(ssl_context=ssl_context)


if __name__ == "__main__":
    main()
