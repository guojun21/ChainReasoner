"""Search package — re-exports all public classes for backward compatibility."""

from src.search.abstract_search_client_interface import AbstractSearchClientInterface
from src.search.model_context_protocol_transport_clients import (
    ModelContextProtocolHttpTransportClient,
    ModelContextProtocolStdioTransportClient,
)
from src.search.alibaba_iqs_search_client import (
    AlibabaIQSSearchClient,
    parse_iqs_search_result_markdown_into_structured_list,
)
from src.search.brave_web_search_client import BraveWebSearchClient
from src.search.google_custom_search_api_client import (
    GoogleCustomSearchApiClient,
    _parse_google_organic_results_into_structured_list,
)
from src.search.language_aware_hybrid_search_dispatcher import (
    LanguageAwareHybridSearchDispatcher,
    check_if_text_is_primarily_chinese_characters,
)

# Backward-compatible aliases for old names
SearchClient = AbstractSearchClientInterface
MCPHttpClient = ModelContextProtocolHttpTransportClient
MCPStdioClient = ModelContextProtocolStdioTransportClient
IQSSearchClient = AlibabaIQSSearchClient
BraveSearchClient = BraveWebSearchClient
HybridSearchClient = LanguageAwareHybridSearchDispatcher
is_mainly_chinese = check_if_text_is_primarily_chinese_characters
parse_iqs_markdown = parse_iqs_search_result_markdown_into_structured_list

__all__ = [
    "AbstractSearchClientInterface",
    "ModelContextProtocolHttpTransportClient",
    "ModelContextProtocolStdioTransportClient",
    "AlibabaIQSSearchClient",
    "BraveWebSearchClient",
    "GoogleCustomSearchApiClient",
    "LanguageAwareHybridSearchDispatcher",
    "check_if_text_is_primarily_chinese_characters",
    "parse_iqs_search_result_markdown_into_structured_list",
    "_parse_google_organic_results_into_structured_list",
    # Backward-compatible aliases
    "SearchClient", "MCPHttpClient", "MCPStdioClient",
    "IQSSearchClient", "BraveSearchClient", "HybridSearchClient",
    "is_mainly_chinese", "parse_iqs_markdown",
]
