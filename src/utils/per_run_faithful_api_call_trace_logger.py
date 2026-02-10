"""Per-run faithful API call trace logger — records every external call with full I/O.

Why: During evaluation debugging, we need to see *exactly* what was sent to
each LLM / search engine / MCP service and what came back, without truncation.
Each evaluation run gets its own timestamped folder so logs never collide.
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class PerRunFaithfulApiCallTraceLogger:
    """Thread-safe JSONL trace logger that creates per-run log directories.

    Why: The existing logger_config.py only logs summary info (query[:80], count).
    This class records *complete* request and response payloads for every
    external API call, enabling post-mortem analysis of exactly which evidence
    the system saw and which LLM prompts it used.

    Each run creates a timestamped directory under the base log path::

        autorunAtnight/logs/run_20260207_150000/
            llm_api_call_trace.jsonl
            search_api_call_trace.jsonl
            mcp_transport_trace.jsonl
    """

    def __init__(self, run_directory: Path):
        run_directory.mkdir(parents=True, exist_ok=True)
        self._run_directory = run_directory
        self._llm_trace_file = open(
            run_directory / "llm_api_call_trace.jsonl", "w", encoding="utf-8")
        self._search_trace_file = open(
            run_directory / "search_api_call_trace.jsonl", "w", encoding="utf-8")
        self._mcp_trace_file = open(
            run_directory / "mcp_transport_trace.jsonl", "w", encoding="utf-8")
        self._answer_postprocess_trace_file = open(
            run_directory / "answer_postprocess_trace.jsonl", "w", encoding="utf-8")
        self._page_cleaning_trace_file = open(
            run_directory / "page_cleaning_trace.jsonl", "w", encoding="utf-8")
        self._answer_voting_trace_file = open(
            run_directory / "answer_voting_trace.jsonl", "w", encoding="utf-8")
        self._call_counter = 0
        self._lock = threading.Lock()

    @property
    def run_directory(self) -> Path:
        """The per-run log directory path."""
        return self._run_directory

    # ── Internal helpers ─────────────────────────────────────────────────

    def _next_call_id(self) -> int:
        """Atomically increment and return the next call sequence number."""
        with self._lock:
            self._call_counter += 1
            return self._call_counter

    def _write_record(self, file_handle, record: Dict[str, Any]) -> None:
        """Write one JSONL line and flush immediately for crash safety."""
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            file_handle.write(line)
            file_handle.flush()

    # ── Public recording methods ─────────────────────────────────────────

    def record_llm_api_call(
        self,
        purpose: str,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        response_text: str,
        elapsed_ms: int,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Record a complete LLM chat-completion call with full prompts and response.

        Why: Seeing the exact system/user prompts and LLM output is essential
        for diagnosing wrong answers — was the evidence good but the prompt bad,
        or was the evidence itself insufficient?
        """
        self._write_record(self._llm_trace_file, {
            "timestamp": datetime.now().isoformat(),
            "call_id": self._next_call_id(),
            "type": "llm_api_call",
            "purpose": purpose,
            "request": {
                "model_id": model_id,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            "response": {
                "text": response_text,
            },
            "elapsed_ms": elapsed_ms,
            "status": status,
            "error": error,
        })

    def record_search_api_call(
        self,
        service_name: str,
        query: str,
        request_params: Dict[str, Any],
        response_data: Dict[str, Any],
        result_count: int,
        elapsed_ms: int,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Record a complete search engine call with full query and all results.

        Why: Comparing search results across engines (IQS vs Google vs Brave)
        reveals which source provided the evidence that led to correct/wrong answers.
        """
        self._write_record(self._search_trace_file, {
            "timestamp": datetime.now().isoformat(),
            "call_id": self._next_call_id(),
            "type": "search_api_call",
            "service_name": service_name,
            "query": query,
            "request": request_params,
            "response": response_data,
            "result_count": result_count,
            "elapsed_ms": elapsed_ms,
            "status": status,
            "error": error,
        })

    def record_mcp_transport_call(
        self,
        transport_type: str,
        url_or_command: str,
        tool_name: str,
        arguments: Dict[str, Any],
        response_text: str,
        elapsed_ms: int,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Record a low-level MCP JSON-RPC call with full arguments and response.

        Why: MCP is the layer between our search clients and the actual services.
        If a search returns empty, the MCP trace shows whether the issue was
        transport-level (timeout, malformed JSON-RPC) or service-level (empty index).
        """
        self._write_record(self._mcp_trace_file, {
            "timestamp": datetime.now().isoformat(),
            "call_id": self._next_call_id(),
            "type": "mcp_transport_call",
            "transport_type": transport_type,
            "url_or_command": url_or_command,
            "tool_name": tool_name,
            "request": {
                "arguments": arguments,
            },
            "response": {
                "text": response_text[:50000] if response_text else "",
            },
            "elapsed_ms": elapsed_ms,
            "status": status,
            "error": error,
        })

    def record_answer_postprocess_trace(
        self,
        question_text: str,
        raw_answer: str,
        final_answer: str,
        trace: Dict[str, Any],
    ) -> None:
        """Record the full answer post-processing pipeline trace.

        Why: To diagnose 'close but wrong' answers, we need to see every
        transformation step from raw LLM output to the final submitted answer,
        including which format hints were detected and how alignment was applied.
        """
        self._write_record(self._answer_postprocess_trace_file, {
            "timestamp": datetime.now().isoformat(),
            "call_id": self._next_call_id(),
            "type": "answer_postprocess",
            "question_text": question_text,
            "raw_answer": raw_answer,
            "final_answer": final_answer,
            "changed": raw_answer != final_answer,
            "pipeline_trace": trace,
        })

    def record_page_content_cleaning_trace(
        self,
        url: str,
        domain: str,
        extractor_used: str,
        raw_html_chars: int,
        layer1_chars: int,
        layer2_chars: int,
        layer3_chars: int,
        final_chars: int,
        cleaning_elapsed_ms: int,
    ) -> None:
        """Record page content cleaning metrics for post-mortem analysis.

        Why (P0-b): To understand whether the 3-layer cleaning pipeline
        is improving content quality, we need to see per-URL metrics:
        which extractor was used, how much content survived each layer,
        and how long cleaning took.
        """
        self._write_record(self._page_cleaning_trace_file, {
            "timestamp": datetime.now().isoformat(),
            "call_id": self._next_call_id(),
            "type": "page_content_cleaning",
            "url": url,
            "domain": domain,
            "extractor_used": extractor_used,
            "raw_html_chars": raw_html_chars,
            "layer1_chars": layer1_chars,
            "layer2_chars": layer2_chars,
            "layer3_chars": layer3_chars,
            "final_chars": final_chars,
            "cleaning_elapsed_ms": cleaning_elapsed_ms,
        })

    def record_answer_consistency_voting_trace(
        self,
        question: str,
        candidates: dict,
        similarity_matrix: dict,
        consensus_found: bool,
        llm_arbitration_used: bool,
        final_answer: str,
        answer_source: str,
        decision_reason: str,
    ) -> None:
        """Record answer consistency voting decision for post-mortem analysis.

        Why (P1-a): To understand whether the consistency voting mechanism is
        selecting better answers, we need per-question traces of all candidates,
        their pairwise similarities, and how the final answer was chosen.
        """
        self._write_record(self._answer_voting_trace_file, {
            "timestamp": datetime.now().isoformat(),
            "call_id": self._next_call_id(),
            "type": "answer_consistency_voting",
            "question": question[:200],
            "candidates": candidates,
            "similarity_matrix": similarity_matrix,
            "consensus_found": consensus_found,
            "llm_arbitration_used": llm_arbitration_used,
            "final_answer": final_answer,
            "answer_source": answer_source,
            "decision_reason": decision_reason,
        })

    # ── Lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush and close all trace files."""
        for fh in (self._llm_trace_file, self._search_trace_file,
                    self._mcp_trace_file, self._answer_postprocess_trace_file,
                    self._page_cleaning_trace_file, self._answer_voting_trace_file):
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def create_per_run_trace_logger_and_directory(
    base_log_directory: Path,
    run_timestamp: Optional[str] = None,
) -> PerRunFaithfulApiCallTraceLogger:
    """Factory: create a per-run trace logger with a timestamped directory.

    Why: Both the eval script and the test script need the same setup logic —
    create a ``run_YYYYMMDD_HHMMSS/`` folder and return a ready-to-use logger.
    """
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = base_log_directory / f"run_{run_timestamp}"
    return PerRunFaithfulApiCallTraceLogger(run_directory)
