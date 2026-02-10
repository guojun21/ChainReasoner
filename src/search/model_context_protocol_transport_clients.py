"""Low-level MCP transport clients (HTTP and Stdio).

Why: MCP servers come in two flavours — cloud HTTP endpoints (like IQS)
and local stdio subprocesses (like brave-search-mcp).  This module
hides the transport difference behind a uniform ``invoke_mcp_tool_*`` interface.
"""

import json as _json
import logging
import os
import select
import subprocess
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _build_argument_preview(arguments: Dict[str, Any], max_len: int = 220) -> str:
    """Build a short, safe argument preview for runtime logs."""
    try:
        preview = _json.dumps(arguments, ensure_ascii=False)
    except Exception:
        preview = str(arguments)
    if len(preview) > max_len:
        return preview[:max_len] + "...(truncated)"
    return preview


# ── HTTP transport ──────────────────────────────────────────────────────────

class ModelContextProtocolHttpTransportClient:
    """Talks to a streamable-HTTP MCP server (e.g. Alibaba IQS)."""

    def __init__(self, url: str, api_key: str, timeout: int = 30):
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self._initialized = False
        self._id_counter = 0
        self.trace_logger = None  # Optional: set by API server for faithful call recording

    def _build_authentication_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _generate_next_request_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def _ensure_mcp_session_is_initialized(self) -> None:
        """Lazy JSON-RPC handshake — only runs once per connection."""
        if self._initialized:
            return
        try:
            import requests
            response = requests.post(
                self.url,
                headers=self._build_authentication_headers(),
                json={
                    "jsonrpc": "2.0",
                    "id": self._generate_next_request_id(),
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "ChainReasoner", "version": "1.0"},
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            requests.post(
                self.url,
                headers=self._build_authentication_headers(),
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                timeout=self.timeout,
            )
            self._initialized = True
        except Exception as exc:
            logger.error("MCP HTTP init failed for %s: %s", self.url[:60], exc)

    def invoke_mcp_tool_via_http(self, tool_name: str, arguments: dict) -> str:
        """Invoke an MCP tool; returns concatenated text content."""
        import time as _time
        arg_preview = _build_argument_preview(arguments)
        if self.trace_logger:
            print(
                f"[MCP][HTTP][CALL] tool={tool_name} url={self.url[:80]} args={arg_preview}",
                flush=True,
            )
        self._ensure_mcp_session_is_initialized()
        start_time = _time.time()
        try:
            import requests
            response = requests.post(
                self.url,
                headers=self._build_authentication_headers(),
                json={
                    "jsonrpc": "2.0",
                    "id": self._generate_next_request_id(),
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            body = response.json()
            content_items = body.get("result", {}).get("content", [])
            texts = [c.get("text", "") for c in content_items if c.get("type") == "text"]
            result_text = "\n".join(texts)
            elapsed_ms = int((_time.time() - start_time) * 1000)
            if self.trace_logger:
                print(
                    f"[MCP][HTTP][DONE] tool={tool_name} status=success elapsed_ms={elapsed_ms} response_chars={len(result_text)}",
                    flush=True,
                )
            if self.trace_logger:
                self.trace_logger.record_mcp_transport_call(
                    transport_type="http", url_or_command=self.url,
                    tool_name=tool_name, arguments=arguments,
                    response_text=result_text, elapsed_ms=elapsed_ms,
                    status="success",
                )
            return result_text
        except Exception as exc:
            logger.error("MCP HTTP invoke_mcp_tool_via_http(%s) error: %s", tool_name, exc)
            elapsed_ms = int((_time.time() - start_time) * 1000)
            if self.trace_logger:
                print(
                    f"[MCP][HTTP][DONE] tool={tool_name} status=error elapsed_ms={elapsed_ms} error={str(exc)[:200]}",
                    flush=True,
                )
            if self.trace_logger:
                self.trace_logger.record_mcp_transport_call(
                    transport_type="http", url_or_command=self.url,
                    tool_name=tool_name, arguments=arguments,
                    response_text="", elapsed_ms=elapsed_ms,
                    status="error", error=str(exc),
                )
            self._initialized = False
            return ""


# ── Stdio transport ─────────────────────────────────────────────────────────

class ModelContextProtocolStdioTransportClient:
    """Talks to a local stdio MCP server (launched as a subprocess).

    Why: npm-based MCP servers (brave-search-mcp, deepwiki, etc.) use
    stdin/stdout JSON-RPC.  The subprocess is started lazily and kept
    alive for reuse.  Thread-safe via an internal lock.
    """

    def __init__(self, command: str, args: List[str],
                 env: Optional[Dict[str, str]] = None, timeout: int = 30):
        self.command = command
        self.args = args
        self.env = env or {}
        self.timeout = timeout
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._id_counter = 0
        self._initialized = False
        self.trace_logger = None  # Optional: set by API server for faithful call recording

    def _generate_next_request_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def _ensure_subprocess_is_running(self) -> None:
        """Launch subprocess if not already running."""
        if self._proc is not None and self._proc.poll() is None:
            return
        merged_env = {**os.environ, **self.env}
        try:
            self._proc = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=merged_env, text=True, bufsize=1,
            )
            logger.info("MCPStdio started %s %s (pid=%d)", self.command, self.args[:3], self._proc.pid)
        except Exception as exc:
            logger.error("MCPStdio failed to start %s: %s", self.command, exc)
            self._proc = None
            raise

    def _send_jsonrpc_request_and_read_response(self, payload: dict) -> dict:
        """Send one JSON-RPC message, read one response line."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            raise RuntimeError("MCPStdio process not running")
        proc.stdin.write(_json.dumps(payload) + "\n")
        proc.stdin.flush()
        ready, _, _ = select.select([proc.stdout], [], [], self.timeout)
        if not ready:
            raise TimeoutError(f"MCPStdio: no response within {self.timeout}s")
        response_line = proc.stdout.readline()
        if not response_line:
            raise RuntimeError("MCPStdio: EOF from server")
        return _json.loads(response_line)

    def _ensure_mcp_session_is_initialized(self) -> None:
        """JSON-RPC handshake — only once per process lifetime."""
        if self._initialized:
            return
        self._ensure_subprocess_is_running()
        self._send_jsonrpc_request_and_read_response({
            "jsonrpc": "2.0",
            "id": self._generate_next_request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "ChainReasoner", "version": "1.0"},
            },
        })
        # Some servers ignore this notification; send it anyway for spec compliance
        self._proc.stdin.write(
            _json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        )
        self._proc.stdin.flush()
        self._initialized = True

    def invoke_mcp_tool_via_stdio(self, tool_name: str, arguments: dict) -> str:
        """Invoke an MCP tool; returns concatenated text content."""
        import time as _time
        with self._lock:
            start_time = _time.time()
            arg_preview = _build_argument_preview(arguments)
            if self.trace_logger:
                print(
                    f"[MCP][STDIO][CALL] tool={tool_name} command={self.command} args={arg_preview}",
                    flush=True,
                )
            try:
                self._ensure_mcp_session_is_initialized()
                response = self._send_jsonrpc_request_and_read_response({
                    "jsonrpc": "2.0",
                    "id": self._generate_next_request_id(),
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                })
                content_items = response.get("result", {}).get("content", [])
                texts = [c.get("text", "") for c in content_items if c.get("type") == "text"]
                result_text = "\n".join(texts)
                elapsed_ms = int((_time.time() - start_time) * 1000)
                if self.trace_logger:
                    print(
                        f"[MCP][STDIO][DONE] tool={tool_name} status=success elapsed_ms={elapsed_ms} response_chars={len(result_text)}",
                        flush=True,
                    )
                if self.trace_logger:
                    self.trace_logger.record_mcp_transport_call(
                        transport_type="stdio",
                        url_or_command=f"{self.command} {' '.join(self.args[:3])}",
                        tool_name=tool_name, arguments=arguments,
                        response_text=result_text, elapsed_ms=elapsed_ms,
                        status="success",
                    )
                return result_text
            except Exception as exc:
                logger.error("MCPStdio invoke_mcp_tool_via_stdio(%s) error: %s", tool_name, exc)
                elapsed_ms = int((_time.time() - start_time) * 1000)
                if self.trace_logger:
                    print(
                        f"[MCP][STDIO][DONE] tool={tool_name} status=error elapsed_ms={elapsed_ms} error={str(exc)[:200]}",
                        flush=True,
                    )
                if self.trace_logger:
                    self.trace_logger.record_mcp_transport_call(
                        transport_type="stdio",
                        url_or_command=f"{self.command} {' '.join(self.args[:3])}",
                        tool_name=tool_name, arguments=arguments,
                        response_text="", elapsed_ms=elapsed_ms,
                        status="error", error=str(exc),
                    )
                self._initialized = False
                if self._proc and self._proc.poll() is None:
                    self._proc.kill()
                self._proc = None
                return ""

    def close(self) -> None:
        """Terminate the subprocess."""
        with self._lock:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            self._proc = None
            self._initialized = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
