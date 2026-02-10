"""Backward-compatibility shim — moved to alibaba_iqs_search_client.py."""
from src.search.alibaba_iqs_search_client import (  # noqa: F401
    AlibabaIQSSearchClient as IQSSearchClient,
    parse_iqs_search_result_markdown_into_structured_list as parse_iqs_markdown,
)
