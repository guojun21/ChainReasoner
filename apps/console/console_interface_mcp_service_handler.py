"""MCP service handling for the console interface.

Why: The console interface calls MCP services (SearXNG, web-search) to
gather context before sending to the LLM.  Extracting this keeps the
main console_interface.py focused on the REPL loop and user interaction.
"""

import time
from typing import Any, Dict

import requests

from src.utils.logger_config import MultiHopLogger


def dispatch_console_mcp_service_call(service_name: str, query: str, mcp_config: dict,
                     logger=None) -> Dict[str, Any]:
    """Dispatch an MCP call for the console interface."""
    start_time = time.time()
    if logger:
        logger.info("MCP Service Call - %s", service_name)

    servers = mcp_config.get("mcpServers", {})
    if service_name not in servers:
        return {"service": service_name, "error": f"Service {service_name} not found"}

    service_cfg = servers[service_name]
    service_url = service_cfg.get("url", "")
    if not service_url:
        return {"service": service_name, "error": f"No URL for {service_name}"}

    try:
        if service_name == "searxng":
            return _call_searxng_meta_search_engine(service_url, query, logger)
        if service_name == "web-search":
            return _call_generic_web_search_service(service_url, query, logger)
        return {"service": service_name, "error": f"Service {service_name} not supported"}
    except requests.exceptions.Timeout:
        _log_mcp_service_error(logger, service_name, start_time, "timeout")
        return {"service": service_name, "error": "Request timeout"}
    except requests.exceptions.ConnectionError:
        _log_mcp_service_error(logger, service_name, start_time, "connection error")
        return {"service": service_name, "error": "Connection error"}
    except Exception as exc:
        _log_mcp_service_error(logger, service_name, start_time, str(exc))
        return {"service": service_name, "error": str(exc)}


def _call_searxng_meta_search_engine(base_url: str, query: str, logger=None) -> Dict[str, Any]:
    """SearXNG meta-search with rate-limit awareness."""
    time.sleep(1)  # basic rate-limit courtesy
    response = requests.get(f"{base_url}/search",
                            params={"q": query, "format": "json", "engines": "duckduckgo"},
                            headers={"Accept": "application/json"}, timeout=30)
    if response.status_code == 429:
        return {"service": "searxng", "error": "Rate limit exceeded (429)"}
    response.raise_for_status()
    results = response.json().get("results", [])
    formatted = [{"title": r.get("title", ""), "url": r.get("url", ""),
                   "snippet": r.get("content", "")} for r in results[:5]]
    if logger:
        logger.info("SearXNG: %d results", len(formatted))
    return {"service": "searxng", "count": len(formatted), "results": formatted}


def _call_generic_web_search_service(base_url: str, query: str, logger=None) -> Dict[str, Any]:
    """Generic web-search MCP service."""
    time.sleep(1)
    response = requests.post(f"{base_url}/api/search",
                             json={"query": query, "limit": 5}, timeout=30)
    if response.status_code == 429:
        return {"service": "web-search", "error": "Rate limit exceeded (429)"}
    response.raise_for_status()
    results = response.json().get("results", [])
    formatted = [{"title": r.get("title", ""), "url": r.get("url", ""),
                   "snippet": r.get("snippet", "")} for r in results]
    if logger:
        logger.info("Web search: %d results", len(formatted))
    return {"service": "web-search", "count": len(formatted), "results": formatted}


def _log_mcp_service_error(logger, service_name: str, start_time: float, msg: str) -> None:
    if logger:
        elapsed = time.time() - start_time
        logger.error("MCP %s %s (%.1fs)", service_name, msg, elapsed)
