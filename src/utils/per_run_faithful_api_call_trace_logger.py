"""Per-run faithful API call trace logger — records every external call with full I/O.

Why: During evaluation debugging, we need to see *exactly* what was sent to
each LLM / search engine / MCP service and what came back, without truncation.
Each evaluation run gets its own timestamped folder so logs never collide.
"""

import json
import os
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
        # Extract run_id from directory name (e.g. "run_20260216_212140" -> "20260216_212140")
        dir_name = run_directory.name
        self._run_id = dir_name.replace("run_", "") if dir_name.startswith("run_") else dir_name
        # Current question context (set per-question via set_question_context)
        self._current_qid: Optional[int] = None
        self._current_question_text: str = ""
        self._llm_trace_file = open(
            run_directory / "llm_api_call_trace.jsonl", "w", encoding="utf-8", buffering=1)
        self._search_trace_file = open(
            run_directory / "search_api_call_trace.jsonl", "w", encoding="utf-8", buffering=1)
        self._mcp_trace_file = open(
            run_directory / "mcp_transport_trace.jsonl", "w", encoding="utf-8", buffering=1)
        self._answer_postprocess_trace_file = open(
            run_directory / "answer_postprocess_trace.jsonl", "w", encoding="utf-8", buffering=1)
        self._page_cleaning_trace_file = open(
            run_directory / "page_cleaning_trace.jsonl", "w", encoding="utf-8", buffering=1)
        self._answer_voting_trace_file = open(
            run_directory / "answer_voting_trace.jsonl", "w", encoding="utf-8", buffering=1)
        self._hop_planning_trace_file = open(
            run_directory / "hop_planning_trace.jsonl", "w", encoding="utf-8", buffering=1)
        self._scratchpad_trace_file = open(
            run_directory / "scratchpad_trace.jsonl", "w", encoding="utf-8", buffering=1)
        self._mini_agent_iteration_trace_file = open(
            run_directory / "mini_agent_iteration_trace.jsonl", "w", encoding="utf-8", buffering=1)
        self._per_hop_retrieval_trace_file = open(
            run_directory / "per_hop_retrieval_trace.jsonl", "w", encoding="utf-8", buffering=1)
        self._per_hop_summary_trace_file = open(
            run_directory / "per_hop_summary_trace.jsonl", "w", encoding="utf-8", buffering=1)
        self._unified_event_file = open(
            run_directory / "unified_event_log.jsonl", "w", encoding="utf-8", buffering=1)
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
        """Write one JSONL line and flush immediately for real-time visibility.

        Also writes a compact summary to the unified event log with run_id and qid.
        """
        # Inject run_id and qid into every record
        record["run_id"] = self._run_id
        record["qid"] = self._current_qid
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            file_handle.write(line)
            file_handle.flush()
            os.fsync(file_handle.fileno())
            # Write compact summary to unified event log
            unified = {
                "ts": record.get("timestamp", datetime.now().isoformat()),
                "run_id": self._run_id,
                "qid": self._current_qid,
                "seq": record.get("call_id", 0),
                "event_type": record.get("type", "unknown"),
                "summary": self._build_summary(record),
            }
            uline = json.dumps(unified, ensure_ascii=False, default=str) + "\n"
            self._unified_event_file.write(uline)
            self._unified_event_file.flush()
            os.fsync(self._unified_event_file.fileno())

    def _build_summary(self, record: Dict[str, Any]) -> str:
        """Build a human-readable one-line summary from a trace record."""
        rtype = record.get("type", "")
        try:
            if rtype == "llm_api_call":
                return (f"purpose={record.get('purpose','')} "
                        f"model={record.get('request',{}).get('model_id','')} "
                        f"elapsed={record.get('elapsed_ms','')}ms "
                        f"status={record.get('status','')}")
            if rtype == "search_api_call":
                return (f"engine={record.get('service_name','')} "
                        f"q='{record.get('query','')[:80]}' "
                        f"results={record.get('result_count',0)} "
                        f"elapsed={record.get('elapsed_ms','')}ms")
            if rtype == "mcp_transport_call":
                return (f"tool={record.get('tool_name','')} "
                        f"elapsed={record.get('elapsed_ms','')}ms "
                        f"status={record.get('status','')}")
            if rtype == "hop_planning":
                return (f"hop_count={record.get('hop_count',0)} "
                        f"question='{record.get('question','')[:60]}'")
            if rtype == "per_hop_retrieval_cycle":
                d = record.get("details", {})
                net = d.get("network_retrieval", {})
                ev = d.get("evaluation", {})
                return (f"Hop {record.get('hop_num','')} Cycle {record.get('cycle','')}: "
                        f"{net.get('query_count',0)} queries, "
                        f"{net.get('total_snippets',0)} snippets, "
                        f"entity={ev.get('extracted_entity','')[:40]}, "
                        f"conf={ev.get('confidence','')}")
            if rtype == "per_hop_summary":
                d = record.get("details", {})
                return (f"Hop {record.get('hop_num','')}: "
                        f"entity={d.get('final_entity','')[:40]} "
                        f"cycles={d.get('total_cycles','')} "
                        f"elapsed={d.get('elapsed_ms','')}ms")
            if rtype == "mini_agent_iteration":
                d = record.get("details", {})
                return (f"iter={record.get('iteration','')} "
                        f"status={d.get('status','')} "
                        f"tool={d.get('tool_name','N/A')} "
                        f"result_len={d.get('result_length','')}")
            if rtype == "answer_consistency_voting":
                return (f"source={record.get('answer_source','')} "
                        f"consensus={record.get('consensus_found','')} "
                        f"answer='{record.get('final_answer','')[:60]}'")
            if rtype == "answer_postprocess":
                return (f"changed={record.get('changed',False)} "
                        f"raw='{record.get('raw_answer','')[:40]}' "
                        f"final='{record.get('final_answer','')[:40]}'")
            if rtype == "page_content_cleaning":
                return (f"url={record.get('url','')[:60]} "
                        f"final_chars={record.get('final_chars',0)}")
            if rtype == "scratchpad":
                return (f"op={record.get('operation','')} "
                        f"details_keys={list(record.get('details',{}).keys())}")
            if rtype == "agent_event":
                return record.get("summary", "")
            return json.dumps(record, ensure_ascii=False, default=str)[:200]
        except Exception:
            return str(record)[:200]

    # ── Question context ──────────────────────────────────────────────────

    def set_question_context(self, question_id: Optional[int], question_text: str = "") -> None:
        """Set the current question being processed (thread-safe).

        Call this at the start of each question so all subsequent records
        automatically include the question_id.
        """
        with self._lock:
            self._current_qid = question_id
            self._current_question_text = question_text[:200]

    # ── Generic event recording ───────────────────────────────────────────

    def record_event(self, event_type: str, summary: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Record a generic agent decision/event that doesn't fit other trace categories.

        This is the catch-all for the ~20 decision points that were previously unlogged.
        Written directly to unified log using the same ts/run_id/qid/seq/event_type format.
        """
        seq = self._next_call_id()
        unified = {
            "ts": datetime.now().isoformat(),
            "run_id": self._run_id,
            "qid": self._current_qid,
            "seq": seq,
            "event_type": event_type,
            "summary": summary,
        }
        if details:
            unified["details"] = details
        line = json.dumps(unified, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            self._unified_event_file.write(line)
            self._unified_event_file.flush()
            os.fsync(self._unified_event_file.fileno())

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

    def record_hop_planning_trace(
        self,
        question: str,
        hop_plan: dict,
    ) -> None:
        """Record the structured hop plan for post-mortem analysis.

        Why (P0-c): To understand how the planner decomposed each question
        and whether the hop targets were appropriate, we record every plan.
        """
        self._write_record(self._hop_planning_trace_file, {
            "timestamp": datetime.now().isoformat(),
            "call_id": self._next_call_id(),
            "type": "hop_planning",
            "question": question[:200],
            "hop_count": hop_plan.get("hop_count", 0),
            "hops": hop_plan.get("hops", []),
            "total_stop_condition": hop_plan.get("total_stop_condition", ""),
            "raw_llm_response": hop_plan.get("raw_llm_response", ""),
        })

    def record_scratchpad_operation_trace(
        self,
        operation: str,
        details: dict,
    ) -> None:
        """Record scratchpad read/write operations for debugging and analysis.

        Why: The scratchpad's local BM25 retrieval and file persistence are
        new components.  Logging every operation lets us verify that evidence
        is being correctly stored and retrieved, and measure the hit rate.
        """
        self._write_record(self._scratchpad_trace_file, {
            "timestamp": datetime.now().isoformat(),
            "call_id": self._next_call_id(),
            "type": "scratchpad",
            "operation": operation,
            "details": details,
        })

    def record_mini_agent_iteration_trace(
        self,
        question: str,
        iteration: int,
        details: Dict[str, Any],
    ) -> None:
        """Record one iteration-level event from the MiniAgent loop.

        Why: The existing ``mini_agent_loop`` record is end-of-loop only.
        Iteration-level traces are needed to diagnose parse failures, tool latency,
        offload decisions, and early-stop behavior without reading raw prompts.
        """
        self._write_record(self._mini_agent_iteration_trace_file, {
            "timestamp": datetime.now().isoformat(),
            "call_id": self._next_call_id(),
            "type": "mini_agent_iteration",
            "question": question[:200],
            "iteration": iteration,
            "details": details,
        })

    def record_per_hop_retrieval_cycle_trace(
        self,
        question: str,
        hop_num: int,
        cycle: int,
        details: Dict[str, Any],
    ) -> None:
        """Record one cycle event from per-hop iterative retrieval.

        Why: Multi-hop failures usually happen at the cycle level (wrong query,
        weak evidence, bad refine decision).  This event captures local retrieval,
        network retrieval, extraction, and evaluation in one structured record.
        """
        self._write_record(self._per_hop_retrieval_trace_file, {
            "timestamp": datetime.now().isoformat(),
            "call_id": self._next_call_id(),
            "type": "per_hop_retrieval_cycle",
            "question": question[:200],
            "hop_num": hop_num,
            "cycle": cycle,
            "details": details,
        })

    def record_per_hop_summary_trace(
        self,
        question: str,
        hop_num: int,
        hop_target: str,
        details: Dict[str, Any],
    ) -> None:
        """Record one hop-level summary after iterative retrieval completes.

        Why: Hop-level summaries allow fast post-mortem analysis of where time
        and drift happened, without replaying every cycle log line.
        """
        self._write_record(self._per_hop_summary_trace_file, {
            "timestamp": datetime.now().isoformat(),
            "call_id": self._next_call_id(),
            "type": "per_hop_summary",
            "question": question[:200],
            "hop_num": hop_num,
            "hop_target": hop_target[:200],
            "details": details,
        })

    # ── Lifecycle ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush and close all trace files."""
        for fh in (self._llm_trace_file, self._search_trace_file,
                    self._mcp_trace_file, self._answer_postprocess_trace_file,
                    self._page_cleaning_trace_file, self._answer_voting_trace_file,
                    self._hop_planning_trace_file, self._scratchpad_trace_file,
                    self._mini_agent_iteration_trace_file,
                    self._per_hop_retrieval_trace_file,
                    self._per_hop_summary_trace_file,
                    self._unified_event_file):
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
