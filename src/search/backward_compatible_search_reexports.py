"""Backward-compatibility shim — all classes moved to new verbose-named submodules.

Why: Existing code does ``from src.search.client import X``.
This file re-exports everything so those imports keep working.
"""

from src.search.abstract_search_client_interface import AbstractSearchClientInterface as SearchClient  # noqa: F401
from src.search.model_context_protocol_transport_clients import (  # noqa: F401
    ModelContextProtocolHttpTransportClient as MCPHttpClient,
    ModelContextProtocolStdioTransportClient as MCPStdioClient,
)
from src.search.alibaba_iqs_search_client import (  # noqa: F401
    AlibabaIQSSearchClient as IQSSearchClient,
    parse_iqs_search_result_markdown_into_structured_list as _parse_search_markdown,
)
from src.search.brave_web_search_client import BraveWebSearchClient as BraveSearchClient  # noqa: F401
from src.search.google_custom_search_api_client import GoogleCustomSearchApiClient  # noqa: F401
from src.search.language_aware_hybrid_search_dispatcher import (  # noqa: F401
    LanguageAwareHybridSearchDispatcher as HybridSearchClient,
    check_if_text_is_primarily_chinese_characters as _is_mainly_chinese,
)
