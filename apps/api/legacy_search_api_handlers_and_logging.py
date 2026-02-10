"""Legacy search handlers for the API server.

Why: The API server still has its own IQS and Brave search methods
(used by _call_mcp_service and as fallbacks).  These are separate
from the new SearchClient abstraction in src/search/.
"""

import json
import os
import time
from typing import Any, Dict, Optional

import requests

from src.utils.logger_config import MultiHopLogger


def resolve_brave_api_key_from_env_or_config(config: dict, mcp_config: dict) -> Optional[str]:
    """Resolve Brave API key from env -> config.yaml -> mcp_config.json."""
    if os.getenv("BRAVE_API_KEY"):
        return os.getenv("BRAVE_API_KEY")
    api_keys = config.get("api_keys", {}) if isinstance(config, dict) else {}
    if api_keys.get("brave_api_key"):
        return api_keys["brave_api_key"]
    return mcp_config.get("mcpServers", {}).get("brave-search", {}).get("env", {}).get("BRAVE_API_KEY")


def redact_api_key_middle_portion_for_logging(value: str) -> str:
    """Redact middle portion of API key for safe logging."""
    if not value or len(value) <= 8:
        return "*" * len(value) if value else ""
    return f"{value[:4]}***{value[-4:]}"


def append_search_trace_record_to_jsonl_audit_log(record: Dict[str, Any], logger=None) -> None:
    """Append a search trace record to the JSONL audit log."""
    try:
        trace_path = MultiHopLogger._log_dir / "search_trace.jsonl"
        trace_path.parent.mkdir(exist_ok=True)
        with open(trace_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        if logger:
            logger.error("Failed to write search trace: %s", exc)


def call_iqs_common_search_via_legacy_mcp_path(query: str, mcp_http_clients: dict,
                    meta: Optional[Dict[str, Any]] = None,
                    logger=None) -> Dict[str, Any]:
    """IQS MCP common_search + readpage_scrape (legacy path).

    Why: This is the original IQS integration used by _call_mcp_service.
    The new AlibabaIQSSearchClient in src/search/ is preferred for pipeline usage.
    """
    start_time = time.time()
    if logger:
        logger.info("MCP search: query=%s", query[:80])

    search_client = mcp_http_clients.get("iqs-search")
    readpage_client = mcp_http_clients.get("iqs-readpage")
    if not search_client:
        if logger:
            logger.error("MCP search client 'iqs-search' not found")
        return {"service": "mcp-search", "query": query, "results": [], "count": 0}

    from apps.api.enhanced_multi_hop_api_server import _parse_search_markdown
    markdown_text = search_client.call_tool("common_search", {"query": query})
    if not markdown_text or markdown_text.strip() == "# 搜索结果":
        return {"service": "mcp-search", "query": query, "results": [], "count": 0}

    results = _parse_search_markdown(markdown_text)
    if readpage_client and results:
        for idx in range(min(5, len(results))):
            page_url = results[idx].get("url", "")
            if not page_url:
                continue
            try:
                full_text = readpage_client.call_tool("readpage_scrape", {"url": page_url})
                if full_text and len(full_text) > len(results[idx].get("content", "")):
                    results[idx]["content"] = full_text[:2000]
            except Exception as exc:
                if logger:
                    logger.warning("readpage_scrape failed for %s: %s", page_url[:60], exc)

    elapsed_ms = int((time.time() - start_time) * 1000)
    if logger:
        logger.info("MCP search ok: elapsed_ms=%d query=%s results=%d", elapsed_ms, query[:80], len(results))
    append_search_trace_record_to_jsonl_audit_log({
        "type": "mcp_common_search", "query": query,
        "original_query": (meta or {}).get("original_query"),
        "elapsed_ms": elapsed_ms, "result_count": len(results),
    }, logger)
    return {"service": "mcp-search", "query": query, "results": results, "count": len(results)}


def call_brave_search_api_with_rate_limiting(query: str, api_key: str, brave_lock, brave_last_call_ts: float,
                      brave_min_interval: float,
                      meta: Optional[Dict[str, Any]] = None,
                      logger=None) -> Dict[str, Any]:
    """Brave Search API with thread-safe rate limiting (legacy path)."""
    if not api_key:
        return {"service": "brave-search", "query": query, "error": "Missing Brave API key"}

    endpoint = "https://api.search.brave.com/res/v1/web/search"
    params = {"q": query, "count": 10}
    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
    start_time = time.time()
    response = None
    try:
        for retry in range(4):
            with brave_lock:
                now = time.time()
                wait_needed = brave_min_interval - (now - brave_last_call_ts)
                if wait_needed > 0:
                    time.sleep(wait_needed)
                response = requests.get(endpoint, headers=headers, params=params, timeout=30)
            if response.status_code == 429:
                wait_time = 10 * (2 ** retry)
                if logger:
                    logger.warning("Brave 429, retrying in %ds (attempt %d/4)", wait_time, retry + 1)
                time.sleep(wait_time)
                continue
            break
        response.raise_for_status()
        data = response.json()
        results = [
            {"title": item.get("title", ""), "url": item.get("url", ""), "content": item.get("description", "")}
            for item in data.get("web", {}).get("results", [])[:10]
        ]
        elapsed_ms = int((time.time() - start_time) * 1000)
        append_search_trace_record_to_jsonl_audit_log({
            "type": "brave_search", "query": query,
            "status_code": response.status_code, "elapsed_ms": elapsed_ms,
            "result_count": len(results),
        }, logger)
        return {"service": "brave-search", "query": query, "results": results, "count": len(results)}
    except Exception as exc:
        if logger:
            logger.error("Brave search error: %s", exc)
        return {"service": "brave-search", "query": query, "error": str(exc)}
