"""MCP service dispatcher and individual service callers.

Why: The server needs to call many different MCP services (IQS, Brave,
SearXNG, DeepWiki, etc.).  This module dispatches by service name
and handles both HTTP and stdio MCP transports.
"""

import json
import re
import subprocess
import time
from typing import Any, Dict, Optional

import requests


def dispatch_mcp_service_call_to_appropriate_transport(service_name: str, query: str,
                     mcp_config: dict, mcp_http_clients: dict,
                     logger=None) -> Dict[str, Any]:
    """Route an MCP call to the right transport and service."""
    servers = mcp_config.get("mcpServers", {})
    if service_name not in servers:
        return {"error": f"MCP service '{service_name}' not found",
                "available_services": list(servers.keys())}

    cfg = servers[service_name]
    service_type = cfg.get("type", "stdio")
    start_time = time.time()

    try:
        if service_type == "http":
            return _call_mcp_service_via_http_transport(service_name, query, cfg, mcp_http_clients, logger)
        if service_name == "searxng":
            return _call_searxng_meta_search_engine(query, logger)
        if service_name == "web-search":
            return _call_duckduckgo_html_fallback_search(query, logger)
        command = cfg.get("command", "")
        args = cfg.get("args", [])
        if not command:
            return {"error": f"MCP service '{service_name}' has no command"}
        return _call_mcp_service_via_stdio_subprocess(service_name, [command] + args, query, logger)
    except Exception as exc:
        elapsed = time.time() - start_time
        if logger:
            logger.error("MCP %s exception (%.1fs): %s", service_name, elapsed, exc)
        return {"error": f"MCP service error: {exc}"}


def _call_mcp_service_via_http_transport(service_name: str, query: str, cfg: dict,
                   mcp_http_clients: dict, logger=None) -> Dict[str, Any]:
    """Dispatch to a pre-initialised MCPHttpClient."""
    client = mcp_http_clients.get(service_name)
    if not client:
        return {"error": f"HTTP MCP client '{service_name}' not initialized"}
    tools = cfg.get("tools", [])
    tool_name = tools[0] if tools else service_name
    text = client.call_tool(tool_name, {"query": query})
    return {
        "service": service_name, "query": query,
        "results": [{"title": service_name, "url": "", "content": text[:3000]}] if text else [],
        "count": 1 if text else 0,
    }


def _call_searxng_meta_search_engine(query: str, logger=None) -> Dict[str, Any]:
    """SearXNG public meta-search engine."""
    try:
        response = requests.get("https://searx.stream/search",
                                params={"q": query, "format": "json"}, timeout=30)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])[:5]
        return {"service": "searxng", "query": query, "results": results, "count": len(results)}
    except Exception as exc:
        if logger:
            logger.error("SearXNG error: %s", exc)
        return {"error": f"SearXNG error: {exc}"}


def _call_duckduckgo_html_fallback_search(query: str, logger=None) -> Dict[str, Any]:
    """DuckDuckGo HTML fallback search."""
    try:
        response = requests.get("https://duckduckgo.com/html/", params={"q": query}, timeout=30)
        response.raise_for_status()
        return {"service": "web-search", "query": query, "status": "success", "note": "Web search completed"}
    except Exception as exc:
        if logger:
            logger.error("Web search error: %s", exc)
        return {"error": f"Web search error: {exc}"}


def _call_mcp_service_via_stdio_subprocess(service_name: str, command: list, query: str,
                        logger=None) -> Dict[str, Any]:
    """Launch a stdio MCP server as a subprocess and extract results.

    Why: Many npm-based MCP servers use stdin/stdout JSON-RPC.
    This handles the full lifecycle: start -> send -> read -> kill.
    """
    mcp_request = {"id": f"test-{service_name}", "function": service_name,
                   "arguments": {"query": query, "count": 10}}
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8")
        process.stdin.write(json.dumps(mcp_request) + "\n")
        process.stdin.flush()

        start_time = time.time()
        stdout_lines = []
        while time.time() - start_time < 30:
            if process.stdout.closed:
                break
            line = process.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
            stdout_lines.append(line)
            try:
                response = json.loads(line.strip())
                if "result" in response:
                    results = _extract_search_results_from_mcp_tool_response(response["result"], service_name)
                    _gracefully_terminate_then_force_kill_process(process)
                    return {"service": service_name, "query": query,
                            "results": results, "count": len(results)}
            except json.JSONDecodeError:
                continue

        _gracefully_terminate_then_force_kill_process(process)
        all_output = "".join(stdout_lines)
        results = _scan_raw_stdout_for_json_result_blocks(all_output, service_name)
        return {"service": service_name, "query": query, "results": results, "count": len(results)}
    except subprocess.TimeoutExpired:
        if "process" in locals():
            process.kill()
        return {"error": f"{service_name} timeout"}
    except Exception as exc:
        if "process" in locals():
            _gracefully_terminate_then_force_kill_process(process)
        return {"error": f"{service_name} error: {exc}"}


def _extract_search_results_from_mcp_tool_response(result: dict, service_name: str) -> list:
    """Parse MCP tool result content into search result dicts."""
    search_results = []
    for item in result.get("content", []):
        if item.get("type") != "text":
            continue
        try:
            content_json = json.loads(item["text"])
            for entry in content_json.get("results", []):
                search_results.append({
                    "title": entry.get("title", ""),
                    "url": entry.get("url", ""),
                    "content": entry.get("content", ""),
                })
        except json.JSONDecodeError:
            search_results.append({"title": f"{service_name} Result", "url": "", "content": item["text"]})
    return search_results


def _scan_raw_stdout_for_json_result_blocks(output: str, service_name: str) -> list:
    """Last-resort: scan raw stdout for JSON result blocks."""
    for match in re.findall(r'\{[^}]*"result"[^}]*\}', output):
        try:
            response = json.loads(match)
            if "result" in response:
                return _extract_search_results_from_mcp_tool_response(response["result"], service_name)
        except json.JSONDecodeError:
            continue
    return []


def _gracefully_terminate_then_force_kill_process(process) -> None:
    """Gracefully terminate, then force-kill if needed."""
    try:
        process.terminate()
        process.wait(timeout=2)
    except (subprocess.TimeoutExpired, Exception):
        process.kill()
