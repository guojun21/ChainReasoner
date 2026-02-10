"""Language-aware search dispatcher — Chinese to IQS, English to Google/Brave/DDG.

Why: Chinese queries get better results from IQS (Alibaba's index);
English queries get better results from Google (broadest index), Brave,
or DuckDuckGo. This multi-engine hybrid approach maximises accuracy
across both languages in the competition dataset.
"""

import logging
from typing import Any, Dict, Optional

from src.search.abstract_search_client_interface import AbstractSearchClientInterface
from src.search.brave_web_search_client import BraveWebSearchClient
from src.search.alibaba_iqs_search_client import AlibabaIQSSearchClient
from src.search.google_custom_search_api_client import GoogleCustomSearchApiClient
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

    Fallback chain:
      - Chinese: IQS (primary) -> DuckDuckGo (fallback)
      - English: Google (primary) -> Brave (fallback) -> DuckDuckGo -> IQS (last resort)
    """

    def __init__(self, iqs: Optional[AlibabaIQSSearchClient],
                 brave: Optional[BraveWebSearchClient] = None,
                 google: Optional[GoogleCustomSearchApiClient] = None,
                 duckduckgo: Optional[DuckDuckGoWebSearchClient] = None):
        self.iqs = iqs
        self.brave = brave
        self.google = google
        self.duckduckgo = duckduckgo
        self.trace_logger = None
        # Engine-level switches used by preflight checks.
        self.engine_enabled = {
            "iqs": bool(iqs),
            "google": bool(google),
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
            # Chinese: IQS first (best Chinese index), DuckDuckGo fallback
            if self._is_engine_usable("iqs"):
                result = self.iqs.execute_search_query(query, meta)
                if not result.get("error") and result.get("count", 0) > 0:
                    return result
                fallback_result = result
            else:
                logger.info("IQS disabled/unavailable, skipping for Chinese query: %s", query[:60])
            if self._is_engine_usable("google"):
                logger.info("IQS empty/failed for Chinese query, falling back to Google: %s", query[:60])
                google_result = self.google.execute_search_query(query, meta)
                if not google_result.get("error") and google_result.get("count", 0) > 0:
                    return self._enrich_non_iqs_results_with_page_content(google_result)
                fallback_result = google_result
            else:
                logger.info("Google disabled/unavailable, skipping for Chinese query: %s", query[:60])
            if self._is_engine_usable("duckduckgo"):
                logger.info("IQS/Google empty/failed for Chinese query, falling back to DuckDuckGo: %s", query[:60])
                ddg_result = self.duckduckgo.execute_search_query(query, meta)
                if not ddg_result.get("error") and ddg_result.get("count", 0) > 0:
                    return self._enrich_non_iqs_results_with_page_content(ddg_result)
                fallback_result = ddg_result
            else:
                logger.info("DuckDuckGo disabled/unavailable, skipping for Chinese query: %s", query[:60])
            return fallback_result

        # English: Google first (broadest index), Brave fallback, DuckDuckGo, IQS last resort
        if self._is_engine_usable("google"):
            result = self.google.execute_search_query(query, meta)
            if not result.get("error") and result.get("count", 0) > 0:
                return self._enrich_non_iqs_results_with_page_content(result)
            logger.info("Google failed/empty, trying Brave: %s", query[:60])
            fallback_result = result
        else:
            logger.info("Google disabled/unavailable, skipping and trying Brave: %s", query[:60])

        if self._is_engine_usable("brave"):
            result = self.brave.execute_search_query(query, meta)
            if not result.get("error") and result.get("count", 0) > 0:
                return self._enrich_non_iqs_results_with_page_content(result)
            logger.info("Brave failed/empty, trying DuckDuckGo: %s", query[:60])
            fallback_result = result
        else:
            logger.info("Brave disabled/unavailable, skipping and trying DuckDuckGo: %s", query[:60])

        if self._is_engine_usable("duckduckgo"):
            result = self.duckduckgo.execute_search_query(query, meta)
            if not result.get("error") and result.get("count", 0) > 0:
                return self._enrich_non_iqs_results_with_page_content(result)
            logger.info("DuckDuckGo failed/empty, falling back to IQS: %s", query[:60])
            fallback_result = result
        else:
            logger.info("DuckDuckGo disabled/unavailable, skipping and trying IQS: %s", query[:60])

        if self._is_engine_usable("iqs"):
            return self.iqs.execute_search_query(query, meta)
        logger.error("No usable search backend for query: %s", query[:80])
        return fallback_result
