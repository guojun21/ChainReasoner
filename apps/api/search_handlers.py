"""Backward-compatibility shim — moved to legacy_search_api_handlers_and_logging.py."""
from apps.api.legacy_search_api_handlers_and_logging import (  # noqa: F401
    resolve_brave_api_key_from_env_or_config as get_brave_api_key,
    redact_api_key_middle_portion_for_logging as mask_api_key,
    append_search_trace_record_to_jsonl_audit_log as log_search_trace,
    call_iqs_common_search_via_legacy_mcp_path as call_iqs_search,
    call_brave_search_api_with_rate_limiting as call_brave_search,
)
