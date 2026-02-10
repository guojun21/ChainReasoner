"""DuckDuckGo Web Search client — free, unlimited, no API key required.

Why: Google API key has a hard quota limit, and SearXNG public instances
are unreliable. DuckDuckGo's HTML lite endpoint provides stable, free
web search results with decent quality for both English and Chinese queries.
This serves as a reliable fallback when Google is unavailable.
"""

import logging
import re as _re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import unquote as _unquote

import requests

from src.search.abstract_search_client_interface import AbstractSearchClientInterface

logger = logging.getLogger(__name__)

DUCKDUCKGO_HTML_SEARCH_URL = "https://html.duckduckgo.com/html/"


def _parse_duckduckgo_html_response_into_structured_results(html_text: str) -> List[Dict[str, str]]:
    """Parse DuckDuckGo HTML lite search results into [{title, url, content}].

    Why: DuckDuckGo's lite HTML endpoint returns results in a structured
    HTML format with ``class="result__a"`` for links and ``class="result__snippet"``
    for snippets. We parse these with regex to avoid heavy HTML parser dependencies.
    """
    results: List[Dict[str, str]] = []

    # Extract (href, title_html, snippet_html) tuples from the result page
    matches = _re.findall(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
        r'.*?class="result__snippet"[^>]*>(.*?)</(?:td|span)',
        html_text, _re.DOTALL,
    )
    for raw_url, title_html, snippet_html in matches:
        # Strip HTML tags
        title = _re.sub(r"<[^>]+>", "", title_html).strip()
        snippet = _re.sub(r"<[^>]+>", "", snippet_html).strip()

        # DuckDuckGo wraps URLs in a redirect: //duckduckgo.com/l/?uddg=<encoded_url>&...
        url = raw_url
        uddg_match = _re.search(r"uddg=([^&]+)", raw_url)
        if uddg_match:
            url = _unquote(uddg_match.group(1))

        if title or snippet:
            results.append({"title": title, "url": url, "content": snippet})

    return results


class DuckDuckGoWebSearchClient(AbstractSearchClientInterface):
    """DuckDuckGo web search via HTML lite endpoint — free, no API key.

    Why: Provides a reliable search backend that never runs out of quota.
    Used as a fallback when Google API key is exhausted, or as a primary
    engine for queries where IQS/Brave don't have good coverage.
    """

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def execute_search_query(self, query: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send query to DuckDuckGo HTML lite and return standardised results."""
        start_time = time.time()
        logger.info("DuckDuckGo search: query=%s", query[:80])

        try:
            response = requests.get(
                DUCKDUCKGO_HTML_SEARCH_URL,
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0 (compatible; ChainReasoner/1.0)"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            html_text = response.text
        except Exception as exc:
            logger.error("DuckDuckGo search error: %s", exc)
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_result = {"service": "duckduckgo", "query": query,
                            "results": [], "count": 0, "error": str(exc)}
            if self.trace_logger:
                self.trace_logger.record_search_api_call(
                    service_name="duckduckgo", query=query,
                    request_params={"q": query},
                    response_data=error_result, result_count=0,
                    elapsed_ms=elapsed_ms, status="error", error=str(exc),
                )
            return error_result

        results = _parse_duckduckgo_html_response_into_structured_results(html_text)

        elapsed_ms = int((time.time() - start_time) * 1000)
        top_titles = [r["title"][:60] for r in results[:3]]
        logger.info("DuckDuckGo search ok: elapsed_ms=%d query=%s results=%d top=%s",
                     elapsed_ms, query[:80], len(results), top_titles)
        result_dict = {"service": "duckduckgo", "query": query,
                       "results": results, "count": len(results)}
        if self.trace_logger:
            self.trace_logger.record_search_api_call(
                service_name="duckduckgo", query=query,
                request_params={"q": query},
                response_data=result_dict, result_count=len(results),
                elapsed_ms=elapsed_ms, status="success",
            )
        return result_dict
