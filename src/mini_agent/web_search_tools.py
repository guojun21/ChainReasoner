"""Web search and fetch tools — network tools for MiniAgent.

Why: When local scratchpad evidence is insufficient, the agent needs to search
the web or fetch specific pages.  These tools wrap the existing search_fn and
MCP-based page fetching with rate limiting and error handling.

Tools defined here:
  1. WebSearchTool — search the web (rate-limited)
  2. FetchPageTool — fetch and clean a web page by URL (rate-limited)
  3. DeepWikiSearchTool — search DeepWiki for open-source project info
"""

import logging
from typing import Any, Callable, Dict, Optional

from src.mini_agent.tool_base_and_registry import MiniAgentToolBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_WEB_SEARCH_CALLS_PER_QUESTION = 3
# Raised from 2 to 4: direct HTTP is faster and more reliable than IQS readpage,
# so the agent can afford more page fetches within the time budget.
MAX_FETCH_PAGE_CALLS_PER_QUESTION = 4


# ---------------------------------------------------------------------------
# 1. WebSearchTool
# ---------------------------------------------------------------------------

class WebSearchTool(MiniAgentToolBase):
    """Search the web for additional information."""

    name = "web_search"
    description = f"Search the web for additional information (max {MAX_WEB_SEARCH_CALLS_PER_QUESTION} times)."
    when_to_use = "Local evidence is insufficient and you need external information."
    when_not_to_use = "You haven't checked local evidence yet — always check local first."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        search_fn: Optional[Callable] = None,
        language_priority: str = "bilingual_equal",
        trace_logger: Optional[Any] = None,
    ) -> None:
        self._search_fn = search_fn
        self._call_count = 0
        self._language_priority = language_priority
        self._trace_logger = trace_logger

    def execute(self, query: str = "", **kwargs: Any) -> str:
        """Execute a web search with rate limiting."""
        if self._call_count >= MAX_WEB_SEARCH_CALLS_PER_QUESTION:
            if self._trace_logger and hasattr(self._trace_logger, "record_event"):
                self._trace_logger.record_event("tool_rate_limit_hit",
                    f"web_search limit={MAX_WEB_SEARCH_CALLS_PER_QUESTION} count={self._call_count}")
            return (
                f"ERROR: web_search limit exhausted ({MAX_WEB_SEARCH_CALLS_PER_QUESTION}/"
                f"{MAX_WEB_SEARCH_CALLS_PER_QUESTION} calls used). "
                "Do NOT call web_search again. Use submit_answer or local tools (grep_evidence, read_file) instead."
            )
        if self._search_fn is None:
            if self._trace_logger and hasattr(self._trace_logger, "record_event"):
                self._trace_logger.record_event("search_fn_unavailable", "web_search search_fn is None")
            return "(web search not available)"
        if not query:
            return "(web_search requires a 'query' argument)"

        self._call_count += 1

        try:
            try:
                search_result = self._search_fn(query, {"original_query": query})
            except TypeError:
                search_result = self._search_fn(query)

            results_list = search_result.get("results", []) if isinstance(search_result, dict) else []
            if results_list:
                parts = []
                for r in results_list[:5]:
                    parts.append(
                        f"- {r.get('title', 'Untitled')}\n"
                        f"  URL: {r.get('url', '')}\n"
                        f"  {r.get('content', '')[:300]}"
                    )
                return "\n".join(parts)
            return f"(no web results for: {query})"

        except Exception as exc:
            logger.warning("web_search error for query '%s': %s", query[:80], exc)
            return f"(web_search error: {exc})"

    @property
    def call_count(self) -> int:
        """Return current call count for logging."""
        return self._call_count


# ---------------------------------------------------------------------------
# 2. FetchPageTool
# ---------------------------------------------------------------------------

class FetchPageByUrlTool(MiniAgentToolBase):
    """Fetch and clean a web page by URL."""

    name = "fetch_page"
    description = f"Fetch and clean a web page by URL (max {MAX_FETCH_PAGE_CALLS_PER_QUESTION} times)."
    when_to_use = "You found a promising URL in evidence or search results but need the full page content."
    when_not_to_use = "You don't have a specific URL — use web_search first to find URLs."
    parameters_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch",
            },
        },
        "required": ["url"],
    }

    def __init__(
        self,
        mcp_config: Optional[Dict] = None,
        mcp_http_clients: Optional[Dict] = None,
        trace_logger: Optional[Any] = None,
    ) -> None:
        self._mcp_config = mcp_config
        self._mcp_http_clients = mcp_http_clients
        self._call_count = 0
        self._trace_logger = trace_logger

    def execute(self, url: str = "", **kwargs: Any) -> str:
        """Fetch a page via direct HTTP (primary) or MCP readpage (fallback).

        Why priority reversal: The direct HTTP fetcher has a mature 3-layer
        cleaning pipeline (rule-based -> density pruning -> optional LLM) and
        handles English pages well.  IQS readpage is slow (5-22s) and returns
        poor content for non-Chinese URLs.  IQS readpage is now only used as
        a fallback when direct HTTP fails.
        """
        if self._call_count >= MAX_FETCH_PAGE_CALLS_PER_QUESTION:
            if self._trace_logger and hasattr(self._trace_logger, "record_event"):
                self._trace_logger.record_event("tool_rate_limit_hit",
                    f"fetch_page limit={MAX_FETCH_PAGE_CALLS_PER_QUESTION} count={self._call_count}")
            return f"(fetch_page limit reached: {MAX_FETCH_PAGE_CALLS_PER_QUESTION} calls max)"
        if not url:
            return "(fetch_page requires a 'url' argument)"

        self._call_count += 1

        # Primary: direct HTTP fetcher with 3-layer cleaning pipeline
        try:
            from src.search.direct_http_web_page_content_fetcher import (
                _fetch_and_clean_single_page,
            )
            text = _fetch_and_clean_single_page(url)
            if text and len(text.strip()) > 50:
                logger.info("fetch_page via direct HTTP: url=%s len=%d", url[:80], len(text))
                return text
        except Exception as exc:
            logger.warning("fetch_page direct HTTP error for %s: %s", url[:80], exc)

        # Fallback: IQS readpage MCP (useful for Chinese pages behind firewalls)
        if self._mcp_http_clients:
            client = self._mcp_http_clients.get("iqs-readpage")
            if client:
                try:
                    text = client.call_tool("readpage_scrape", {"url": url})
                    if text and len(text.strip()) > 50:
                        logger.info("fetch_page via iqs-readpage (fallback): url=%s len=%d", url[:80], len(text))
                        return text
                except Exception as exc:
                    logger.warning("fetch_page iqs-readpage fallback error for %s: %s", url[:80], exc)

        return f"(fetch_page: page content too short or empty for {url})"

    @property
    def call_count(self) -> int:
        """Return current call count for logging."""
        return self._call_count


# ---------------------------------------------------------------------------
# 3. DeepWikiSearchTool
# ---------------------------------------------------------------------------

class DeepWikiSearchTool(MiniAgentToolBase):
    """Search DeepWiki for open-source project / GitHub repo information."""

    name = "deepwiki_search"
    description = "Search DeepWiki for open-source project / GitHub repo information."
    when_to_use = "The question involves GitHub repos, open-source projects, or software tools."
    when_not_to_use = "The question is about people, companies, history, or non-tech topics."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for DeepWiki",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        mcp_config: Optional[Dict] = None,
        mcp_http_clients: Optional[Dict] = None,
    ) -> None:
        self._mcp_config = mcp_config
        self._mcp_http_clients = mcp_http_clients

    def execute(self, query: str = "", **kwargs: Any) -> str:
        """Search DeepWiki via MCP stdio transport."""
        if not query:
            return "(deepwiki_search requires a 'query' argument)"
        if not self._mcp_config:
            return "(deepwiki_search: MCP config not available)"

        servers = self._mcp_config.get("mcpServers", {})
        if "mcp-deepwiki" not in servers:
            return "(deepwiki_search: mcp-deepwiki service not configured)"

        try:
            from apps.api.mcp_service_dispatch_and_stdio_callers import (
                dispatch_mcp_service_call_to_appropriate_transport,
            )
            result = dispatch_mcp_service_call_to_appropriate_transport(
                "mcp-deepwiki", query, self._mcp_config,
                self._mcp_http_clients or {}, logger,
            )
            if "error" in result:
                return f"(deepwiki_search error: {result['error']})"
            results_list = result.get("results", [])
            if results_list:
                parts = []
                for r in results_list[:5]:
                    parts.append(
                        f"- {r.get('title', 'DeepWiki Result')}\n"
                        f"  URL: {r.get('url', '')}\n"
                        f"  {r.get('content', '')[:500]}"
                    )
                return "\n".join(parts)
            return f"(no deepwiki results for: {query})"
        except Exception as exc:
            logger.warning("deepwiki_search error: %s", exc)
            return f"(deepwiki_search error: {exc})"
