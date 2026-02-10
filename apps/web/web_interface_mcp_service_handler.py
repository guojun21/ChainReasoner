"""MCP service handling for the web interface.

Why: The enhanced web interface calls MCP services (SearXNG, web-search)
as part of multi-hop reasoning.  Extracting these keeps the main
web_interface.py focused on routing and template rendering.
"""

from typing import Any, Dict

import requests


def dispatch_web_mcp_service_call(service_name: str, query: str, mcp_config: dict) -> Dict[str, Any]:
    """Dispatch an MCP call for the web interface."""
    servers = mcp_config.get("mcpServers", {})
    if service_name not in servers:
        return {"error": f"MCP service '{service_name}' not found",
                "available_services": list(servers.keys())}
    if service_name == "searxng":
        return _call_searxng_meta_search_engine(query)
    if service_name == "web-search":
        return _call_duckduckgo_html_fallback_search(query)
    return {"error": f"MCP service '{service_name}' not yet implemented"}


def _call_searxng_meta_search_engine(query: str) -> Dict[str, Any]:
    """SearXNG meta-search engine."""
    try:
        response = requests.get("https://searx.stream/search",
                                params={"q": query, "format": "json"}, timeout=30)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])[:5]
        return {"service": "searxng", "query": query, "results": results,
                "count": len(data.get("results", []))}
    except Exception as exc:
        return {"error": f"SearXNG error: {exc}"}


def _call_duckduckgo_html_fallback_search(query: str) -> Dict[str, Any]:
    """DuckDuckGo HTML fallback."""
    try:
        response = requests.get("https://duckduckgo.com/html/", params={"q": query}, timeout=30)
        response.raise_for_status()
        return {"service": "web-search", "query": query, "status": "success"}
    except Exception as exc:
        return {"error": f"Web search error: {exc}"}
