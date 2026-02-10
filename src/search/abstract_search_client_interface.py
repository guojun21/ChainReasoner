"""Abstract search interface — all search backends implement this contract.

Why: Decouples reasoning logic from any specific search provider.  Swap
IQS / Brave / LangStudio without touching the agent code.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class AbstractSearchClientInterface(ABC):
    """Contract every search backend must fulfil."""

    # Optional trace logger — set by the API server to record faithful API call traces
    trace_logger = None

    @abstractmethod
    def execute_search_query(self, query: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run a web search.

        Returns ``{"service", "query", "results": [{title, url, content}], "count"}``.
        """
        ...
