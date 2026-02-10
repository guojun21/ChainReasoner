"""Backward-compatibility shim — moved to question_answering_pipeline_builder.py."""
from src.question_answering_pipeline_builder import (  # noqa: F401
    QuestionAnsweringPipeline as Pipeline,
    build_question_answering_pipeline as build_pipeline,
    answer_single_question as answer_question,
)
