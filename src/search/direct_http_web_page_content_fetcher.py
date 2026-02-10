"""Direct HTTP web page content fetcher — enrich search snippets with full page text.

Why: Brave and DuckDuckGo only return short snippets (200-500 chars),
while many competition questions require specific facts (names, numbers,
dates) buried deep in the page. This fetcher grabs the full HTML, runs it
through a 3-layer cleaning pipeline (rule-based + density pruning + optional
LLM refinement) with domain-aware extractors, giving the LLM high-quality
evidence per result.

P0-b upgrade: Replaced the old regex-only cleaner with the three-layer
pipeline in three_layer_html_content_cleaning_pipeline.py.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import requests

from src.search.three_layer_html_content_cleaning_pipeline import (
    clean_html_with_three_layer_pipeline,
)

logger = logging.getLogger(__name__)

# Limits — raised from 3000 to 5000 because cleaned content has better signal-to-noise
MAX_PAGE_TEXT_LENGTH = 5000  # chars to keep per page
FETCH_TIMEOUT_SECONDS = 10
MAX_CONCURRENT_FETCHES = 3


def _fetch_and_clean_single_page(
    url: str,
    query: str = "",
    timeout: int = FETCH_TIMEOUT_SECONDS,
    llm_refine_fn: Optional[Callable[[str, str], str]] = None,
    enable_llm_refinement: bool = False,
    trace_logger: Optional[Any] = None,
) -> str:
    """Fetch a single URL and return cleaned text via the 3-layer pipeline.

    Args:
        url: Page URL to fetch.
        query: Search query for query-aware filtering (Layer 2/3).
        timeout: HTTP timeout in seconds.
        llm_refine_fn: Optional LLM refinement callable for Layer 3.
        enable_llm_refinement: Whether Layer 3 is active.
        trace_logger: Optional trace logger for cleaning metrics.

    Returns empty string on any error (timeout, 4xx/5xx, encoding issues).
    """
    if not url or not url.startswith("http"):
        return ""
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            },
            timeout=timeout,
            allow_redirects=True,
        )
        if response.status_code != 200:
            return ""
        content_type = response.headers.get("Content-Type", "")
        if "html" not in content_type.lower() and "text" not in content_type.lower():
            return ""
        return clean_html_with_three_layer_pipeline(
            html=response.text,
            url=url,
            query=query,
            llm_refine_fn=llm_refine_fn,
            enable_llm_refinement=enable_llm_refinement,
            trace_logger=trace_logger,
        )
    except Exception as exc:
        logger.debug("Fetch failed for %s: %s", url[:80], exc)
        return ""


# Keep the old name as a thin wrapper for backward compatibility
def fetch_single_web_page_as_clean_text(url: str, timeout: int = FETCH_TIMEOUT_SECONDS) -> str:
    """Fetch a single URL and return cleaned text content (legacy API).

    Note: This wrapper does not pass query/llm_refine_fn — use
    _fetch_and_clean_single_page() directly for full pipeline features.
    """
    return _fetch_and_clean_single_page(url=url, query="", timeout=timeout)


def enrich_search_results_with_full_page_content(
    results: List[Dict[str, str]],
    top_n: int = 3,
    max_content_length: int = MAX_PAGE_TEXT_LENGTH,
    query: str = "",
    llm_refine_fn: Optional[Callable[[str, str], str]] = None,
    enable_llm_refinement: bool = False,
    trace_logger: Optional[Any] = None,
) -> List[Dict[str, str]]:
    """Fetch full page text for the top N search results and replace/extend content.

    Why: Search engine snippets are typically 200-500 chars. For multi-hop
    reasoning questions, we need the actual page content to find specific
    facts. This function fetches the top N URLs concurrently, cleans them
    via the 3-layer pipeline, and replaces the short snippet with the
    cleaned text (truncated to max_content_length).

    Only replaces content if the fetched text is longer than the existing snippet.

    Args:
        results: List of search result dicts with 'url' and 'content' keys.
        top_n: Number of top results to fetch.
        max_content_length: Max chars to keep per page.
        query: Search query for query-aware cleaning.
        llm_refine_fn: Optional LLM refinement callable for Layer 3.
        enable_llm_refinement: Whether Layer 3 is active.
        trace_logger: Optional trace logger for cleaning metrics.
    """
    if not results:
        return results

    urls_to_fetch = []
    for idx in range(min(top_n, len(results))):
        url = results[idx].get("url", "")
        if url and url.startswith("http"):
            urls_to_fetch.append((idx, url))

    if not urls_to_fetch:
        return results

    start_time = time.time()

    # Fetch and clean pages concurrently
    fetched = {}
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FETCHES) as executor:
        future_to_idx = {
            executor.submit(
                _fetch_and_clean_single_page,
                url=url,
                query=query,
                llm_refine_fn=llm_refine_fn,
                enable_llm_refinement=enable_llm_refinement,
                trace_logger=trace_logger,
            ): idx
            for idx, url in urls_to_fetch
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                text = future.result()
                if text:
                    fetched[idx] = text
            except Exception:
                pass

    # Replace content if fetched text is more informative
    enriched_count = 0
    for idx, full_text in fetched.items():
        existing_content = results[idx].get("content", "")
        if len(full_text) > len(existing_content):
            results[idx]["content"] = full_text[:max_content_length]
            enriched_count += 1

    elapsed_ms = int((time.time() - start_time) * 1000)
    logger.info("Page enrichment: fetched=%d/%d enriched=%d elapsed_ms=%d",
                len(fetched), len(urls_to_fetch), enriched_count, elapsed_ms)

    if trace_logger and fetched:
        trace_logger.record_search_api_call(
            service_name="page-content-fetcher",
            query=f"enrich top {top_n} URLs (query={query[:40]})",
            request_params={"urls": [url for _, url in urls_to_fetch],
                            "query": query, "llm_refinement": enable_llm_refinement},
            response_data={"fetched_count": len(fetched), "enriched_count": enriched_count,
                           "avg_length": sum(len(t) for t in fetched.values()) // max(len(fetched), 1)},
            result_count=enriched_count,
            elapsed_ms=elapsed_ms, status="success",
        )

    return results
