"""Backward-compatibility shim — moved to large_language_model_call_handlers.py."""
from apps.api.large_language_model_call_handlers import (  # noqa: F401
    send_chat_completion_request_with_retry as call_llm_generic,
    send_structured_reasoning_request_to_llm as call_llm_with_reasoning,
    get_knowledge_only_answer_from_llm as knowledge_answer_llm,
    decompose_multi_hop_question_into_search_queries as decompose_question_llm,
    extract_concise_answer_from_evidence_using_llm as extract_answer_llm,
    verify_answer_against_evidence_via_llm as verify_answer_llm,
)
