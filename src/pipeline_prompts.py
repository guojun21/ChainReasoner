"""Backward-compatibility shim — moved to pipeline_llm_prompt_factory_functions.py."""
from src.pipeline_llm_prompt_factory_functions import (  # noqa: F401
    create_knowledge_only_answer_callback as make_knowledge_fn,
    create_question_decomposition_callback as make_decompose_fn,
    create_evidence_based_answer_extraction_callback as make_extract_fn,
    create_answer_verification_callback as make_verify_fn,
)
