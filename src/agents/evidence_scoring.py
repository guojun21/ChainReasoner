"""Backward-compatibility shim — moved to search_result_relevance_scoring_and_ranking.py."""
from src.agents.search_result_relevance_scoring_and_ranking import (  # noqa: F401
    split_query_into_lowercase_search_terms as tokenize_query,
    calculate_single_search_result_relevance_score as score_result,
    rank_search_results_by_relevance_and_truncate as rank_and_filter,
    deduplicate_and_select_top_evidence_items as select_evidence,
    format_evidence_items_into_llm_readable_text as build_evidence_text,
)
