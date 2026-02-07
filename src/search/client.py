"""
Search Client abstraction for ChainReasoner.

Provides a pluggable interface so the search backend can be swapped without
changing any business logic.  Current implementation wraps the IQS MCP
HTTP service.  For LangStudio migration, implement ``LangStudioSearchClient``.

Includes:
- IQSSearchClient  -- Alibaba Cloud IQS MCP (best for Chinese queries)
- BraveSearchClient -- Brave Web Search API (best for English queries)
- HybridSearchClient -- auto-dispatches based on query language
"""

import json as _json
import logging
import os
import re as _re
import subprocess
import threading
import time
import unicodedata
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SearchClient(ABC):
    """Abstract search client interface.

    Subclass this and implement ``search`` to plug in a new search backend.
    """

    @abstractmethod
    def search(self, query: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a web search and return results.

        Parameters
        ----------
        query : str
            The search query.
        meta : dict, optional
            Extra metadata (e.g. original_query, rewritten_query).

        Returns
        -------
        dict
            ``{"service": str, "query": str, "results": list[dict], "count": int}``
            where each result has keys ``title``, ``url``, ``content``.
        """
        ...


# ---------------------------------------------------------------------------
# Helper: parse IQS markdown into structured results
# ---------------------------------------------------------------------------

def _parse_search_markdown(md_text: str) -> List[Dict[str, str]]:
    """Parse IQS common_search / litesearch markdown into [{title, url, content}]."""
    results: List[Dict[str, str]] = []
    blocks = _re.split(r"\n---\n?", md_text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        title = url = content = ""
        m = _re.search(r"##\s*(?:标题\s*)?(.+)", block)
        if m:
            title = m.group(1).strip()
        m = _re.search(r"\*\*url\*\*[：:]\s*(https?://\S+)", block)
        if m:
            url = m.group(1).strip()
        m = _re.search(r"\*\*摘要\*\*[：:]\s*(.+)", block, _re.DOTALL)
        if m:
            content = m.group(1).strip()
            content = _re.split(r"\n-\s*\*\*", content)[0].strip()
        if title or content:
            results.append({"title": title, "url": url, "content": content})
    return results


# ---------------------------------------------------------------------------
# MCP HTTP helper (moved from api_server.py for reuse)
# ---------------------------------------------------------------------------

class MCPHttpClient:
    """Generic MCP HTTP client for streamable-HTTP mode servers."""

    def __init__(self, url: str, api_key: str, timeout: int = 30):
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self._initialized = False
        self._id_counter = 0

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def _ensure_initialized(self):
        if self._initialized:
            return
        try:
            import requests
            r = requests.post(
                self.url,
                headers=self._headers(),
                json={
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "ChainReasoner", "version": "1.0"},
                    },
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            # Send initialized notification
            requests.post(
                self.url,
                headers=self._headers(),
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                timeout=self.timeout,
            )
            self._initialized = True
        except Exception as e:
            logger.error("MCP init error for %s: %s", self.url[:60], e)

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool and return concatenated text content."""
        self._ensure_initialized()
        try:
            import requests
            r = requests.post(
                self.url,
                headers=self._headers(),
                json={
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            body = r.json()
            contents = body.get("result", {}).get("content", [])
            texts = [c.get("text", "") for c in contents if c.get("type") == "text"]
            return "\n".join(texts)
        except Exception as e:
            logger.error("MCP call_tool(%s) error: %s", tool_name, e)
            self._initialized = False
            return ""


# ---------------------------------------------------------------------------
# MCP Stdio helper (for local stdio-based MCP servers like brave-search-mcp)
# ---------------------------------------------------------------------------

class MCPStdioClient:
    """MCP client that communicates with a stdio-based MCP server via subprocess.

    The server is started lazily on first ``call_tool`` and kept alive for reuse.
    Thread-safe: all stdin/stdout access is serialised by a lock.
    """

    def __init__(self, command: str, args: List[str], env: Optional[Dict[str, str]] = None, timeout: int = 30):
        self.command = command
        self.args = args
        self.env = env or {}
        self.timeout = timeout
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._id_counter = 0
        self._initialized = False

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def _ensure_started(self):
        """Start the subprocess if not already running."""
        if self._proc is not None and self._proc.poll() is None:
            return
        merged_env = {**os.environ, **self.env}
        try:
            self._proc = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=merged_env,
                text=True,
                bufsize=1,
            )
            logger.info("MCPStdio: started %s %s (pid=%d)", self.command, self.args[:3], self._proc.pid)
        except Exception as e:
            logger.error("MCPStdio: failed to start %s: %s", self.command, e)
            self._proc = None
            raise

    def _send_recv(self, payload: dict) -> dict:
        """Send a JSON-RPC message and read one JSON-RPC response line."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            raise RuntimeError("MCPStdio process not running")
        line = _json.dumps(payload) + "\n"
        proc.stdin.write(line)
        proc.stdin.flush()

        # Read one line of response (with timeout)
        import select
        ready, _, _ = select.select([proc.stdout], [], [], self.timeout)
        if not ready:
            raise TimeoutError(f"MCPStdio: no response within {self.timeout}s")
        resp_line = proc.stdout.readline()
        if not resp_line:
            raise RuntimeError("MCPStdio: EOF from server")
        return _json.loads(resp_line)

    def _ensure_initialized(self):
        if self._initialized:
            return
        self._ensure_started()
        # Send initialize
        resp = self._send_recv({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "ChainReasoner", "version": "1.0"},
            },
        })
        logger.debug("MCPStdio init response: %s", str(resp)[:200])
        # Send initialized notification (no response expected, but some servers may respond)
        notif = _json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        self._proc.stdin.write(notif)
        self._proc.stdin.flush()
        self._initialized = True

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool and return concatenated text content."""
        with self._lock:
            try:
                self._ensure_initialized()
                resp = self._send_recv({
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                })
                contents = resp.get("result", {}).get("content", [])
                texts = [c.get("text", "") for c in contents if c.get("type") == "text"]
                return "\n".join(texts)
            except Exception as e:
                logger.error("MCPStdio call_tool(%s) error: %s", tool_name, e)
                self._initialized = False
                # Kill broken process
                if self._proc and self._proc.poll() is None:
                    self._proc.kill()
                self._proc = None
                return ""

    def close(self):
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


# ---------------------------------------------------------------------------
# IQS MCP search implementation
# ---------------------------------------------------------------------------

class IQSSearchClient(SearchClient):
    """Search client backed by Alibaba Cloud IQS MCP services."""

    def __init__(
        self,
        search_client: MCPHttpClient,
        readpage_client: Optional[MCPHttpClient] = None,
        readpage_top_n: int = 5,
    ):
        self.search_client = search_client
        self.readpage_client = readpage_client
        self.readpage_top_n = readpage_top_n

    @classmethod
    def from_mcp_config(cls, mcp_config: dict) -> Optional["IQSSearchClient"]:
        """Build from the ``mcpServers`` section of mcp_config.json."""
        servers = mcp_config.get("mcpServers", {})

        search_cfg = servers.get("iqs-search")
        if not search_cfg or search_cfg.get("type") != "http":
            return None
        search_mc = MCPHttpClient(
            url=search_cfg["url"],
            api_key=search_cfg.get("api_key", ""),
            timeout=30,
        )

        readpage_mc = None
        rp_cfg = servers.get("iqs-readpage")
        if rp_cfg and rp_cfg.get("type") == "http":
            readpage_mc = MCPHttpClient(
                url=rp_cfg["url"],
                api_key=rp_cfg.get("api_key", ""),
                timeout=30,
            )

        return cls(search_client=search_mc, readpage_client=readpage_mc)

    def search(self, query: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        start = time.time()
        logger.info("IQS search: query=%s", query[:80])

        md_text = self.search_client.call_tool("common_search", {"query": query})
        if not md_text or md_text.strip() == "# 搜索结果":
            logger.warning("IQS common_search empty for: %s", query[:60])
            return {"service": "iqs-mcp", "query": query, "results": [], "count": 0}

        formatted = _parse_search_markdown(md_text)

        # Enrich top results with full page text
        if self.readpage_client and formatted:
            top_n = min(self.readpage_top_n, len(formatted))
            for i in range(top_n):
                page_url = formatted[i].get("url", "")
                if not page_url:
                    continue
                try:
                    page_text = self.readpage_client.call_tool(
                        "readpage_scrape", {"url": page_url}
                    )
                    if page_text and len(page_text) > len(formatted[i].get("content", "")):
                        formatted[i]["content"] = page_text[:2000]
                except Exception as e:
                    logger.warning("readpage_scrape failed for %s: %s", page_url[:60], e)

        elapsed_ms = int((time.time() - start) * 1000)
        logger.info(
            "IQS search ok: elapsed_ms=%d query=%s results=%d",
            elapsed_ms, query[:80], len(formatted),
        )

        return {
            "service": "iqs-mcp",
            "query": query,
            "results": formatted,
            "count": len(formatted),
        }


# ---------------------------------------------------------------------------
# Brave Web Search implementation
# ---------------------------------------------------------------------------

class BraveSearchClient(SearchClient):
    """Search client backed by Brave Web Search via MCP stdio server.

    Uses the ``brave-search-mcp`` (or ``@modelcontextprotocol/server-brave-search``)
    npm package as a stdio MCP server.  Falls back to direct HTTP API if MCP
    fails or is unavailable.

    Best for English queries.
    """

    API_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(
        self,
        api_key: str,
        mcp_client: Optional[MCPStdioClient] = None,
        count: int = 10,
        timeout: int = 15,
    ):
        self.api_key = api_key
        self.mcp_client = mcp_client
        self.count = count
        self.timeout = timeout

    @classmethod
    def from_config(cls, config: dict) -> Optional["BraveSearchClient"]:
        """Build from config.yaml's ``api_keys.brave_api_key``."""
        key = config.get("api_keys", {}).get("brave_api_key", "")
        if not key or key.startswith("YOUR_"):
            return None
        return cls(api_key=key)

    @classmethod
    def from_mcp_config(cls, mcp_config: dict) -> Optional["BraveSearchClient"]:
        """Build from the ``mcpServers.brave-search`` section of mcp_config.json.

        Sets up the MCPStdioClient to launch the brave-search-mcp subprocess.
        """
        servers = mcp_config.get("mcpServers", {})
        brave_cfg = servers.get("brave-search")
        if not brave_cfg:
            return None
        if brave_cfg.get("type") != "stdio":
            return None
        env_vars = brave_cfg.get("env", {})
        api_key = env_vars.get("BRAVE_API_KEY", "")
        if not api_key:
            return None
        command = brave_cfg.get("command", "npx")
        args = brave_cfg.get("args", [])
        mcp = MCPStdioClient(command=command, args=args, env=env_vars, timeout=30)
        logger.info("BraveSearchClient: configured via MCP stdio (%s %s)", command, args)
        return cls(api_key=api_key, mcp_client=mcp, count=10)

    def _search_via_mcp(self, query: str) -> Optional[Dict[str, Any]]:
        """Try searching via the MCP stdio server.  Returns None on failure."""
        if not self.mcp_client:
            return None
        try:
            raw_text = self.mcp_client.call_tool("brave_web_search", {"query": query, "count": self.count})
            if not raw_text:
                return None
            # brave_web_search returns markdown-like text or JSON; parse results
            formatted = self._parse_brave_mcp_output(raw_text)
            return {
                "service": "brave-mcp",
                "query": query,
                "results": formatted,
                "count": len(formatted),
            }
        except Exception as e:
            logger.warning("Brave MCP search failed, will try HTTP fallback: %s", e)
            return None

    @staticmethod
    def _parse_brave_mcp_output(text: str) -> List[Dict[str, str]]:
        """Parse the brave_web_search MCP tool output into [{title, url, content}]."""
        results = []
        # Try JSON first
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
        # Fallback: parse markdown-style output
        # Typical format: "Title: ...\nURL: ...\nDescription: ..."
        blocks = _re.split(r"\n---\n|\n\n(?=\d+\.)", text)
        for block in blocks:
            title_m = _re.search(r"(?:Title|标题)[：:]\s*(.+)", block)
            url_m = _re.search(r"(?:URL|链接|url)[：:]\s*(https?://\S+)", block)
            desc_m = _re.search(r"(?:Description|描述|Snippet|摘要)[：:]\s*(.+)", block, _re.DOTALL)
            if title_m or url_m:
                results.append({
                    "title": (title_m.group(1).strip() if title_m else ""),
                    "url": (url_m.group(1).strip() if url_m else ""),
                    "content": (desc_m.group(1).strip()[:500] if desc_m else block.strip()[:500]),
                })
        # Last resort: if nothing parsed, treat whole text as one result
        if not results and text.strip():
            results.append({"title": "", "url": "", "content": text.strip()[:2000]})
        return results

    def _search_via_http(self, query: str) -> Dict[str, Any]:
        """Direct HTTP fallback to Brave API."""
        try:
            import requests
            resp = requests.get(
                self.API_URL,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self.api_key,
                },
                params={"q": query, "count": self.count},
                timeout=self.timeout,
            )
            if resp.status_code == 429:
                logger.warning("Brave HTTP rate limited")
                return {"service": "brave-http", "query": query, "results": [], "count": 0, "error": "rate_limited"}
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error("Brave HTTP search error: %s", e)
            return {"service": "brave-http", "query": query, "results": [], "count": 0, "error": str(e)}

        web_results = data.get("web", {}).get("results", [])
        formatted = []
        for item in web_results:
            formatted.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("description", ""),
            })
        return {
            "service": "brave-http",
            "query": query,
            "results": formatted,
            "count": len(formatted),
        }

    def search(self, query: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        start = time.time()
        logger.info("Brave search: query=%s", query[:80])

        # Try MCP first, then HTTP fallback
        result = self._search_via_mcp(query)
        if result is None or result.get("count", 0) == 0:
            result = self._search_via_http(query)

        elapsed_ms = int((time.time() - start) * 1000)
        top_titles = [r["title"][:60] for r in result.get("results", [])[:3]]
        logger.info(
            "Brave search ok: elapsed_ms=%d service=%s query=%s results=%d top=%s",
            elapsed_ms, result.get("service", "?"), query[:80], result.get("count", 0), top_titles,
        )
        return result


# ---------------------------------------------------------------------------
# Language detection helper
# ---------------------------------------------------------------------------

def _is_mainly_chinese(text: str, threshold: float = 0.3) -> bool:
    """Return True if more than *threshold* fraction of characters are CJK."""
    if not text:
        return False
    cjk = 0
    total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf":
            cjk += 1
    return (cjk / max(total, 1)) >= threshold


# ---------------------------------------------------------------------------
# Hybrid search: auto-dispatch by language
# ---------------------------------------------------------------------------

class HybridSearchClient(SearchClient):
    """Dispatches to Brave for English queries, IQS for Chinese queries.

    Falls back to IQS if Brave returns an error (e.g. rate limit).
    """

    def __init__(self, iqs: IQSSearchClient, brave: Optional[BraveSearchClient] = None):
        self.iqs = iqs
        self.brave = brave

    def search(self, query: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # If no Brave client, always use IQS
        if not self.brave:
            return self.iqs.search(query, meta)

        # Detect language
        if _is_mainly_chinese(query):
            return self.iqs.search(query, meta)

        # English query -> try Brave first
        result = self.brave.search(query, meta)
        if result.get("error") or result.get("count", 0) == 0:
            logger.info("Brave failed/empty, falling back to IQS for: %s", query[:60])
            return self.iqs.search(query, meta)

        return result
