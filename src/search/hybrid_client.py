"""Backward-compatibility shim — moved to language_aware_hybrid_search_dispatcher.py."""
from src.search.language_aware_hybrid_search_dispatcher import (  # noqa: F401
    LanguageAwareHybridSearchDispatcher as HybridSearchClient,
    check_if_text_is_primarily_chinese_characters as is_mainly_chinese,
)
