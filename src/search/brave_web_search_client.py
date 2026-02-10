"""Brave Web Search client — best for English queries.

Why: Brave has superior English-language coverage and fast response times.
Supports two modes: MCP stdio (preferred) and direct HTTP API (fallback).
"""

import json as _json
import logging
import re as _re
import time
from typing import Any, Dict, List, Optional

from src.search.abstract_search_client_interface import AbstractSearchClientInterface
from src.search.model_context_protocol_transport_clients import ModelContextProtocolStdioTransportClient

logger = logging.getLogger(__name__)

BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"


def _parse_brave_mcp_tool_output_into_search_results(text: str) -> List[Dict[str, str]]:
    """Parse brave_web_search tool output into [{title, url, content}].

    Why: The MCP server returns either JSON array or markdown-like text
    depending on version; we must handle both formats gracefully.

    The Brave MCP server outputs results in this format::

        Title: Some Title
        Description: Some description...
        URL: https://example.com

        Title: Next Result
        Description: ...
        URL: ...

    Blocks are separated by blank lines before each ``Title:`` line.
    """
    results: List[Dict[str, str]] = []
    # Attempt 1: JSON array (some MCP versions return this)
    try:
        data = _json.loads(text)
        if isinstance(data, list):
            for item in data:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("description", item.get("snippet", "")),
                })
            return results
    except (_json.JSONDecodeError, TypeError):
        pass

    # Attempt 2: Split on blank-line-before-Title or --- separator
    # The key fix: Brave outputs "Title:\n Description:\n URL:\n\nTitle:..."
    # so we split on any blank line followed by "Title:" OR on "---" separators
    for block in _re.split(r"\n---\n|\n\n(?=Title[：:])", text):
        block = block.strip()
        if not block:
            continue
        title_match = _re.search(r"(?:Title|标题)[：:]\s*(.+)", block)
        url_match = _re.search(r"(?:URL|链接|url)[：:]\s*(https?://\S+)", block)
        desc_match = _re.search(r"(?:Description|描述|Snippet|摘要)[：:]\s*(.+)", block, _re.DOTALL)
        if title_match or url_match:
            # For description: stop at the URL line if description captured too much
            content = ""
            if desc_match:
                content = desc_match.group(1).strip()
                # Trim off trailing URL: line that may have been captured by DOTALL
                content = _re.split(r"\nURL[：:]", content)[0].strip()[:500]
            results.append({
                "title": (title_match.group(1).strip() if title_match else ""),
                "url": (url_match.group(1).strip() if url_match else ""),
                "content": content or block.strip()[:500],
            })
    if not results and text.strip():
        results.append({"title": "", "url": "", "content": text.strip()[:2000]})
    return results


class BraveWebSearchClient(AbstractSearchClientInterface):
    """Brave Web Search — MCP stdio first, HTTP API fallback."""

    def __init__(self, api_key: str,
                 mcp_client: Optional[ModelContextProtocolStdioTransportClient] = None,
                 count: int = 10, timeout: int = 15):
        self.api_key = api_key
        self.mcp_client = mcp_client
        self.count = count
        self.timeout = timeout

    @classmethod
    def from_config(cls, config: dict) -> Optional["BraveWebSearchClient"]:
        """Factory: build from config.yaml ``api_keys.brave_api_key``."""
        key = config.get("api_keys", {}).get("brave_api_key", "")
        if not key or key.startswith("YOUR_"):
            return None
        return cls(api_key=key)

    @classmethod
    def from_mcp_config(cls, mcp_config: dict) -> Optional["BraveWebSearchClient"]:
        """Factory: build from ``mcpServers.brave-search`` in mcp_config.json."""
        servers = mcp_config.get("mcpServers", {})
        brave_cfg = servers.get("brave-search")
        if not brave_cfg or brave_cfg.get("type") != "stdio":
            return None
        env_vars = brave_cfg.get("env", {})
        api_key = env_vars.get("BRAVE_API_KEY", "")
        if not api_key:
            return None
        command = brave_cfg.get("command", "npx")
        args = brave_cfg.get("args", [])
        mcp = ModelContextProtocolStdioTransportClient(
            command=command, args=args, env=env_vars, timeout=30)
        logger.info("BraveWebSearchClient: MCP stdio (%s %s)", command, args)
        return cls(api_key=api_key, mcp_client=mcp, count=10)

    def _search_via_mcp(self, query: str) -> Optional[Dict[str, Any]]:
        """Try the MCP stdio server first — faster, no HTTP overhead."""
        if not self.mcp_client:
            return None
        try:
            raw_text = self.mcp_client.invoke_mcp_tool_via_stdio(
                "brave_web_search", {"query": query, "count": self.count})
            if not raw_text:
                return None
            results = _parse_brave_mcp_tool_output_into_search_results(raw_text)
            return {"service": "brave-mcp", "query": query, "results": results, "count": len(results)}
        except Exception as exc:
            logger.warning("Brave MCP failed, will try HTTP: %s", exc)
            return None

    def _search_via_http(self, query: str) -> Dict[str, Any]:
        """Direct HTTP fallback when MCP is unavailable or rate-limited."""
        try:
            import requests
            response = requests.get(
                BRAVE_API_URL,
                headers={"Accept": "application/json", "Accept-Encoding": "gzip",
                         "X-Subscription-Token": self.api_key},
                params={"q": query, "count": self.count},
                timeout=self.timeout,
            )
            if response.status_code == 429:
                logger.warning("Brave HTTP rate limited")
                return {"service": "brave-http", "query": query, "results": [], "count": 0, "error": "rate_limited"}
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.error("Brave HTTP error: %s", exc)
            return {"service": "brave-http", "query": query, "results": [], "count": 0, "error": str(exc)}

        results = [
            {"title": item.get("title", ""), "url": item.get("url", ""), "content": item.get("description", "")}
            for item in data.get("web", {}).get("results", [])
        ]
        return {"service": "brave-http", "query": query, "results": results, "count": len(results)}

    def execute_search_query(self, query: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        start_time = time.time()
        logger.info("Brave search: query=%s", query[:80])

        result = self._search_via_mcp(query)
        if result is None or result.get("count", 0) == 0:
            result = self._search_via_http(query)

        elapsed_ms = int((time.time() - start_time) * 1000)
        top_titles = [r["title"][:60] for r in result.get("results", [])[:3]]
        logger.info("Brave search ok: elapsed_ms=%d service=%s query=%s results=%d top=%s",
                     elapsed_ms, result.get("service", "?"), query[:80], result.get("count", 0), top_titles)
        if self.trace_logger:
            self.trace_logger.record_search_api_call(
                service_name=result.get("service", "brave"), query=query,
                request_params={"q": query, "count": self.count},
                response_data=result, result_count=result.get("count", 0),
                elapsed_ms=elapsed_ms,
                status="error" if result.get("error") else "success",
                error=result.get("error"),
            )
        return result
