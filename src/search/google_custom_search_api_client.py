"""Google Custom Search API client — high-quality global web search.

Why: Google has the broadest web index and highest-quality English results.
Unlike Brave which has rate-limit issues, this self-hosted proxy endpoint
at google.ydcloud.org provides stable access without stdio MCP overhead.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import requests

from src.search.abstract_search_client_interface import AbstractSearchClientInterface

logger = logging.getLogger(__name__)

GOOGLE_SEARCH_API_DEFAULT_URL = "http://google.ydcloud.org/api/search"
GOOGLE_UNAVAILABLE_ERROR_CODE = "GOOGLE_UNAVAILABLE"


def _parse_google_organic_results_into_structured_list(organic_items: List[dict]) -> List[Dict[str, str]]:
    """Convert Google API ``organic`` array into [{title, url, content}].

    Why: The google.ydcloud.org proxy returns results in a ``organic`` array
    with ``title``, ``link``, ``snippet`` fields — different from both IQS
    markdown and Brave JSON, so we need a dedicated parser.
    """
    results: List[Dict[str, str]] = []
    for item in organic_items:
        title = item.get("title", "")
        url = item.get("link", "")
        content = item.get("snippet", "")
        if title or content:
            results.append({"title": title, "url": url, "content": content})
    return results


class GoogleCustomSearchApiClient(AbstractSearchClientInterface):
    """Google web search via self-hosted REST proxy at google.ydcloud.org.

    Why: Provides a third search backend alongside IQS and Brave.
    Google's index is the most comprehensive for both English and Chinese,
    giving better coverage on obscure entities the competition may test.
    """

    def __init__(self, api_url: str, api_key: str,
                 count: int = 10, timeout: int = 15):
        self.api_url = api_url
        self.api_key = api_key
        self.count = count
        self.timeout = timeout

    @classmethod
    def from_mcp_config(cls, mcp_config: dict) -> Optional["GoogleCustomSearchApiClient"]:
        """Factory: build from ``mcpServers.google-search`` in mcp_config.json."""
        servers = mcp_config.get("mcpServers", {})
        google_cfg = servers.get("google-search")
        if not google_cfg:
            return None
        api_url = google_cfg.get("url", GOOGLE_SEARCH_API_DEFAULT_URL)
        api_key = google_cfg.get("api_key", "")
        if not api_key:
            return None
        logger.info("GoogleCustomSearchApiClient: url=%s", api_url)
        return cls(api_url=api_url, api_key=api_key, count=10)

    def execute_search_query(self, query: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send query to Google proxy and return standardised results."""
        start_time = time.time()
        logger.info("Google search: query=%s", query[:80])

        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Content-Type": "application/json",
                    "X-API-KEY": self.api_key,
                },
                json={"q": query},
                timeout=self.timeout,
            )
            if response.status_code == 429:
                logger.warning("Google search rate limited")
                elapsed_ms = int((time.time() - start_time) * 1000)
                rate_result = {"service": "google-search", "query": query,
                               "results": [], "count": 0, "error": "rate_limited",
                               "error_code": GOOGLE_UNAVAILABLE_ERROR_CODE}
                if self.trace_logger:
                    self.trace_logger.record_search_api_call(
                        service_name="google-search", query=query,
                        request_params={"q": query, "api_url": self.api_url},
                        response_data=rate_result, result_count=0,
                        elapsed_ms=elapsed_ms, status="error", error="rate_limited",
                    )
                return rate_result
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.error("Google search error: %s", exc)
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_result = {"service": "google-search", "query": query,
                            "results": [], "count": 0, "error": str(exc),
                            "error_code": GOOGLE_UNAVAILABLE_ERROR_CODE}
            if self.trace_logger:
                self.trace_logger.record_search_api_call(
                    service_name="google-search", query=query,
                    request_params={"q": query, "api_url": self.api_url},
                    response_data=error_result, result_count=0,
                    elapsed_ms=elapsed_ms, status="error", error=str(exc),
                )
            return error_result

        # Google proxy returns {"code": 400, "msg": "..."} on quota/auth errors
        # instead of a proper HTTP status code — must check JSON body explicitly
        api_code = data.get("code")
        if api_code is not None and api_code != 200:
            error_msg = data.get("msg", f"API returned code {api_code}")
            logger.warning("Google search API error: code=%s msg=%s", api_code, error_msg)
            elapsed_ms = int((time.time() - start_time) * 1000)
            api_error_result = {"service": "google-search", "query": query,
                                "results": [], "count": 0, "error": error_msg,
                                "error_code": GOOGLE_UNAVAILABLE_ERROR_CODE}
            if self.trace_logger:
                self.trace_logger.record_search_api_call(
                    service_name="google-search", query=query,
                    request_params={"q": query, "api_url": self.api_url},
                    response_data=api_error_result, result_count=0,
                    elapsed_ms=elapsed_ms, status="error", error=error_msg,
                )
            return api_error_result

        organic_items = data.get("organic", [])
        results = _parse_google_organic_results_into_structured_list(organic_items)

        elapsed_ms = int((time.time() - start_time) * 1000)
        top_titles = [r["title"][:60] for r in results[:3]]
        logger.info("Google search ok: elapsed_ms=%d query=%s results=%d top=%s",
                     elapsed_ms, query[:80], len(results), top_titles)
        result_dict = {"service": "google-search", "query": query,
                       "results": results, "count": len(results)}
        if self.trace_logger:
            self.trace_logger.record_search_api_call(
                service_name="google-search", query=query,
                request_params={"q": query, "api_url": self.api_url},
                response_data=result_dict, result_count=len(results),
                elapsed_ms=elapsed_ms, status="success",
            )
        return result_dict
