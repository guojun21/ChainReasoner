"""Backward-compatibility shim — moved to search_query_parsing_and_generation.py."""
from src.agents.search_query_parsing_and_generation import (  # noqa: F401
    extract_structured_clues_from_question_text as parse_question,
    generate_search_queries_from_question as generate_queries,
    classify_question_into_answer_type_category as detect_question_type,
)
