"""IQS MCP search client — best for Chinese queries.

Why: Alibaba Cloud IQS has no rate limit and returns high-quality
Chinese-language results with full-page scraping support.

P0-b: readpage_scrape results now pass through Layer 1 rule-based cleaning
to remove HTML residue and noise.
"""

import logging
import re as _re
import time
from typing import Any, Dict, List, Optional

from src.search.abstract_search_client_interface import AbstractSearchClientInterface
from src.search.model_context_protocol_transport_clients import ModelContextProtocolHttpTransportClient
from src.search.three_layer_html_content_cleaning_pipeline import apply_layer1_only

logger = logging.getLogger(__name__)
IQS_UNAVAILABLE_ERROR_CODE = "IQS_UNAVAILABLE"


def parse_iqs_search_result_markdown_into_structured_list(markdown_text: str) -> List[Dict[str, str]]:
    """Convert IQS common_search markdown into [{title, url, content}].

    Why: IQS returns a custom markdown format with ``## title``,
    ``**url**: ...``, ``**摘要**: ...`` blocks separated by ``---``.
    """
    results: List[Dict[str, str]] = []
    for block in _re.split(r"\n---\n?", markdown_text):
        block = block.strip()
        if not block:
            continue
        title = url = content = ""
        match = _re.search(r"##\s*(?:标题\s*)?(.+)", block)
        if match:
            title = match.group(1).strip()
        match = _re.search(r"\*\*url\*\*[：:]\s*(https?://\S+)", block)
        if match:
            url = match.group(1).strip()
        match = _re.search(r"\*\*摘要\*\*[：:]\s*(.+)", block, _re.DOTALL)
        if match:
            content = match.group(1).strip()
            content = _re.split(r"\n-\s*\*\*", content)[0].strip()
        if title or content:
            results.append({"title": title, "url": url, "content": content})
    return results


class AlibabaIQSSearchClient(AbstractSearchClientInterface):
    """Alibaba Cloud IQS via HTTP MCP — ``common_search`` + ``readpage_scrape``."""

    # Domains whose readpage_scrape results are almost always noise or irrelevant.
    # Why: Q20 test showed IQS scraping kuwo music, moegirl wiki, etc. for a
    # history question — each scrape takes 5-22s and returns garbage.
    _LOW_VALUE_DOMAINS = frozenset({
        "kuwo.cn", "kugou.com", "bilibili.com", "toutiao.com",
        "moegirl.org.cn", "douyin.com", "ixigua.com", "youku.com",
        "iqiyi.com", "qq.com",
    })

    def __init__(self, search_client: ModelContextProtocolHttpTransportClient,
                 readpage_client: Optional[ModelContextProtocolHttpTransportClient] = None,
                 readpage_top_n: int = 2):
        """Initialize IQS client.

        Why readpage_top_n=2 (down from 5): Most IQS search results beyond
        the top 2 are low-quality for non-Chinese queries, and each scrape
        takes 5-22s.  Reducing to 2 saves 15-66s per search call.
        """
        self.search_client = search_client
        self.readpage_client = readpage_client
        self.readpage_top_n = readpage_top_n

    @classmethod
    def from_mcp_config(cls, mcp_config: dict) -> Optional["AlibabaIQSSearchClient"]:
        """Factory: build from ``mcpServers`` in mcp_config.json."""
        servers = mcp_config.get("mcpServers", {})
        search_cfg = servers.get("iqs-search")
        if not search_cfg or search_cfg.get("type") != "http":
            return None
        search_mc = ModelContextProtocolHttpTransportClient(
            url=search_cfg["url"], api_key=search_cfg.get("api_key", ""))

        readpage_mc = None
        readpage_cfg = servers.get("iqs-readpage")
        if readpage_cfg and readpage_cfg.get("type") == "http":
            readpage_mc = ModelContextProtocolHttpTransportClient(
                url=readpage_cfg["url"], api_key=readpage_cfg.get("api_key", ""))

        return cls(search_client=search_mc, readpage_client=readpage_mc)

    @classmethod
    def _is_low_value_domain(cls, url: str) -> bool:
        """Check if URL belongs to a domain known to return noisy/irrelevant content.

        Why: IQS search results often include music sites (kuwo), video sites
        (bilibili), and niche wikis (moegirl) whose scraped content is 99%
        unrelated to factual questions.
        """
        try:
            from urllib.parse import urlparse
            host = urlparse(url).hostname or ""
            for bad_domain in cls._LOW_VALUE_DOMAINS:
                if host == bad_domain or host.endswith("." + bad_domain):
                    return True
        except Exception:
            pass
        return False

    def execute_search_query(self, query: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        start_time = time.time()
        logger.info("IQS search: query=%s", query[:80])

        raw_markdown = self.search_client.invoke_mcp_tool_via_http("common_search", {"query": query})
        if raw_markdown and (
            "call failed, status:" in raw_markdown
            or "Retrieval.Arrears" in raw_markdown
            or "Please recharge first" in raw_markdown
        ):
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_result = {
                "service": "iqs-mcp",
                "query": query,
                "results": [],
                "count": 0,
                "error": raw_markdown[:300],
                "error_code": IQS_UNAVAILABLE_ERROR_CODE,
            }
            logger.warning("IQS search backend unavailable: %s", error_result["error"])
            if self.trace_logger:
                self.trace_logger.record_search_api_call(
                    service_name="iqs-mcp", query=query,
                    request_params={"tool": "common_search", "query": query},
                    response_data=error_result, result_count=0,
                    elapsed_ms=elapsed_ms, status="error", error=error_result["error"],
                )
            return error_result
        if not raw_markdown or raw_markdown.strip() == "# 搜索结果":
            logger.warning("IQS common_search empty for: %s", query[:60])
            elapsed_ms = int((time.time() - start_time) * 1000)
            empty_result = {"service": "iqs-mcp", "query": query, "results": [], "count": 0}
            if self.trace_logger:
                self.trace_logger.record_search_api_call(
                    service_name="iqs-mcp", query=query,
                    request_params={"tool": "common_search", "query": query},
                    response_data=empty_result, result_count=0,
                    elapsed_ms=elapsed_ms, status="success", error="empty result set",
                )
            return empty_result

        results = parse_iqs_search_result_markdown_into_structured_list(raw_markdown)

        # Enrich top results with scraped full-page content for better evidence
        # P0-b: apply Layer 1 cleaning to readpage_scrape results to remove noise
        # Skip readpage entirely when language_priority is english_first — IQS
        # readpage_scrape is slow (5-22s per page) and returns poor content for
        # English pages.  The direct HTTP fetcher handles those better.
        meta = meta or {}
        skip_readpage = meta.get("language_priority") == "english_first"
        if skip_readpage:
            logger.info("IQS readpage SKIPPED: language_priority=english_first, "
                        "English pages use direct HTTP fetcher instead")
        if self.readpage_client and results and not skip_readpage:
            enrichment_count = min(self.readpage_top_n, len(results))
            for idx in range(enrichment_count):
                page_url = results[idx].get("url", "")
                if not page_url:
                    continue
                if self._is_low_value_domain(page_url):
                    logger.info("IQS readpage SKIPPED low-value domain: %s", page_url[:80])
                    continue
                try:
                    full_text = self.readpage_client.invoke_mcp_tool_via_http(
                        "readpage_scrape", {"url": page_url})
                    if full_text:
                        # Apply Layer 1 cleaning to remove HTML residue and noise
                        cleaned_text = apply_layer1_only(full_text)
                        if cleaned_text and len(cleaned_text) > len(results[idx].get("content", "")):
                            results[idx]["content"] = cleaned_text[:3000]
                except Exception as exc:
                    logger.warning("readpage_scrape failed for %s: %s", page_url[:60], exc)

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info("IQS search ok: elapsed_ms=%d query=%s results=%d",
                     elapsed_ms, query[:80], len(results))
        result_dict = {"service": "iqs-mcp", "query": query, "results": results, "count": len(results)}
        if self.trace_logger:
            self.trace_logger.record_search_api_call(
                service_name="iqs-mcp", query=query,
                request_params={"tool": "common_search", "query": query},
                response_data=result_dict, result_count=len(results),
                elapsed_ms=elapsed_ms, status="success",
            )
        return result_dict
