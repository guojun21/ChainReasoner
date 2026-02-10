"""Backward-compatibility shim — moved to search_agent_shared_constants_and_stopwords.py."""
from src.agents.search_agent_shared_constants_and_stopwords import (  # noqa: F401
    CHINESE_QUESTION_NOISE_STOPWORDS as CN_STOPWORDS,
    GENERIC_ENTITY_ROLE_WORDS_TO_REJECT as ENTITY_ROLE_STOPWORDS,
    LLM_REFUSAL_AND_INSUFFICIENT_EVIDENCE_PHRASES as REFUSAL_PHRASES,
)
