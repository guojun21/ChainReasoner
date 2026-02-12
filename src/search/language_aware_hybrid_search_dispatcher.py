"""Language-aware search dispatcher — routes queries through multiple engines.

Priority (based on 2026-02-11 benchmark with real competition queries):
  - EN: BrightData (Google-quality, 5s) -> Brave (fast, 2s) -> DDG -> IQS
  - ZH: BrightData (Google-quality, 5s) -> IQS (best for ZH but 30s slow)
        -> Brave -> DDG
  - Google disabled by default (quota permanently exhausted on free tier)

Why: BrightData returns real Google results via SERP scraping at ~5s with
10 results, outperforming IQS (30s, 5 results) on speed and matching on
quality after UTF-8 encoding fix.  IQS remains valuable for Chinese as
a fallback due to its Baidu-sourced index.
"""

import logging
from typing import Any, Dict, Optional

from src.search.abstract_search_client_interface import AbstractSearchClientInterface
from src.search.brave_web_search_client import BraveWebSearchClient
from src.search.alibaba_iqs_search_client import AlibabaIQSSearchClient
from src.search.google_custom_search_api_client import GoogleCustomSearchApiClient
from src.search.bright_data_serp_api_client import BrightDataSerpApiClient
from src.search.duckduckgo_web_search_client import DuckDuckGoWebSearchClient
from src.search.direct_http_web_page_content_fetcher import enrich_search_results_with_full_page_content

logger = logging.getLogger(__name__)


def check_if_text_is_primarily_chinese_characters(text: str, threshold: float = 0.3) -> bool:
    """True if CJK characters exceed *threshold* fraction of non-space chars."""
    if not text:
        return False
    cjk_count = 0
    total_count = 0
    for char in text:
        if char.isspace():
            continue
        total_count += 1
        if "\u4e00" <= char <= "\u9fff" or "\u3400" <= char <= "\u4dbf":
            cjk_count += 1
    return (cjk_count / max(total_count, 1)) >= threshold


class LanguageAwareHybridSearchDispatcher(AbstractSearchClientInterface):
    """Routes queries by language with multi-engine fallback.

    Fallback chain (based on 2026-02-11 benchmark):
      - Chinese: BrightData -> IQS -> Brave -> DDG -> Google (last resort)
      - English: BrightData -> Brave -> DDG -> IQS -> Google (last resort)
    """

    def __init__(self, iqs: Optional[AlibabaIQSSearchClient],
                 brave: Optional[BraveWebSearchClient] = None,
                 google: Optional[GoogleCustomSearchApiClient] = None,
                 brightdata: Optional[BrightDataSerpApiClient] = None,
                 duckduckgo: Optional[DuckDuckGoWebSearchClient] = None):
        self.iqs = iqs
        self.brave = brave
        self.google = google
        self.brightdata = brightdata
        self.duckduckgo = duckduckgo
        self.trace_logger = None
        # Engine-level switches used by preflight checks.
        self.engine_enabled = {
            "iqs": bool(iqs),
            "google": bool(google),
            "brightdata": bool(brightdata),
            "brave": bool(brave),
            "duckduckgo": bool(duckduckgo),
        }
        self.engine_disable_reasons: Dict[str, str] = {}

    def set_engine_enabled(self, engine_name: str, enabled: bool, reason: str = "") -> None:
        """Enable/disable one backend dynamically at runtime."""
        if engine_name not in self.engine_enabled:
            return
        self.engine_enabled[engine_name] = bool(enabled)
        if enabled:
            self.engine_disable_reasons.pop(engine_name, None)
        else:
            self.engine_disable_reasons[engine_name] = reason or "disabled by preflight"
            logger.warning("Search engine disabled: %s (%s)", engine_name, self.engine_disable_reasons[engine_name])

    def get_engine_status_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return current configured/enabled status for all engines."""
        return {
            "iqs": {
                "configured": bool(self.iqs),
                "enabled": bool(self.engine_enabled.get("iqs", False)),
                "reason": self.engine_disable_reasons.get("iqs", ""),
            },
            "google": {
                "configured": bool(self.google),
                "enabled": bool(self.engine_enabled.get("google", False)),
                "reason": self.engine_disable_reasons.get("google", ""),
            },
            "brightdata": {
                "configured": bool(self.brightdata),
                "enabled": bool(self.engine_enabled.get("brightdata", False)),
                "reason": self.engine_disable_reasons.get("brightdata", ""),
            },
            "brave": {
                "configured": bool(self.brave),
                "enabled": bool(self.engine_enabled.get("brave", False)),
                "reason": self.engine_disable_reasons.get("brave", ""),
            },
            "duckduckgo": {
                "configured": bool(self.duckduckgo),
                "enabled": bool(self.engine_enabled.get("duckduckgo", False)),
                "reason": self.engine_disable_reasons.get("duckduckgo", ""),
            },
        }

    def _is_engine_usable(self, engine_name: str) -> bool:
        if not self.engine_enabled.get(engine_name, False):
            return False
        if engine_name == "iqs":
            return self.iqs is not None
        if engine_name == "google":
            return self.google is not None
        if engine_name == "brightdata":
            return self.brightdata is not None
        if engine_name == "brave":
            return self.brave is not None
        if engine_name == "duckduckgo":
            return self.duckduckgo is not None
        return False

    def _enrich_non_iqs_results_with_page_content(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch full page text for top search results from non-IQS engines.

        Why: Brave and DuckDuckGo only return short snippets (200-500 chars).
        IQS already does readpage_scrape internally so it doesn't need this.
        Fetching the actual page content gives the LLM 10-50x more evidence.

        P0-b: Now passes query for query-aware cleaning and supports LLM refinement.
        """
        service = result.get("service", "")
        if service == "iqs-mcp":
            return result  # IQS already has full page content
        results_list = result.get("results", [])
        search_query = result.get("query", "")
        if results_list:
            enrich_search_results_with_full_page_content(
                results_list,
                top_n=3,
                max_content_length=5000,
                query=search_query,
                llm_refine_fn=getattr(self, "_llm_refine_fn", None),
                enable_llm_refinement=getattr(self, "_enable_llm_refinement", False),
                trace_logger=self.trace_logger,
            )
        return result

    def execute_search_query(self, query: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        is_chinese = check_if_text_is_primarily_chinese_characters(query)
        fallback_result = {
            "service": "hybrid-dispatcher",
            "query": query,
            "results": [],
            "count": 0,
            "error": "no_search_backend_available",
        }

        if is_chinese:
            # Chinese: BrightData -> IQS -> Brave -> DDG -> Google (last resort)
            # BrightData first: Google-quality results at 5s (vs IQS 30s)
            # IQS second: best Chinese index via Baidu, but very slow (20-35s)
            zh_chain = [
                ("brightdata", "BrightData"),
                ("iqs", "IQS"),
                ("brave", "Brave"),
                ("duckduckgo", "DDG"),
                ("google", "Google"),
            ]
            for engine_name, display_name in zh_chain:
                if not self._is_engine_usable(engine_name):
                    logger.info("%s disabled/unavailable for ZH query: %s", display_name, query[:60])
                    continue
                client = getattr(self, engine_name, None)
                if client is None:
                    continue
                result = client.execute_search_query(query, meta)
                if not result.get("error") and result.get("count", 0) > 0:
                    # IQS already enriches via readpage; others need page fetching
                    if engine_name == "iqs":
                        return result
                    return self._enrich_non_iqs_results_with_page_content(result)
                logger.info("%s failed/empty for ZH query, trying next: %s", display_name, query[:60])
                fallback_result = result
            return fallback_result

        # English: BrightData -> Brave -> DDG -> IQS -> Google (last resort)
        # BrightData first: real Google results, 8-10 results, ~5s
        # Brave second: fast (2s), 10 results, good EN quality
        en_chain = [
            ("brightdata", "BrightData"),
            ("brave", "Brave"),
            ("duckduckgo", "DDG"),
            ("iqs", "IQS"),
            ("google", "Google"),
        ]
        for engine_name, display_name in en_chain:
            if not self._is_engine_usable(engine_name):
                logger.info("%s disabled/unavailable for EN query: %s", display_name, query[:60])
                continue
            client = getattr(self, engine_name, None)
            if client is None:
                continue
            result = client.execute_search_query(query, meta)
            if not result.get("error") and result.get("count", 0) > 0:
                if engine_name == "iqs":
                    return result
                return self._enrich_non_iqs_results_with_page_content(result)
            logger.info("%s failed/empty for EN query, trying next: %s", display_name, query[:60])
            fallback_result = result
        logger.error("No usable search backend for query: %s", query[:80])
        return fallback_result
