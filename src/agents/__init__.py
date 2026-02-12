"""Agent components — multi-hop reasoning, answer extraction, and search orchestration.

Module classification (P0-e directory cleanup):

Core modules (actively used by ConstrainedMultiHopSearchAgent pipeline):
  - constrained_multi_hop_search_agent.py  — main orchestration agent
  - structured_multi_hop_reasoning_planner.py — hop plan generation
  - per_hop_result_validator_and_corrector.py — unified hop evaluation + refinement
  - search_query_parsing_and_generation.py — clue extraction + query generation
  - search_result_relevance_scoring_and_ranking.py — result ranking + dedup + formatting
  - llm_answer_cleaning_and_candidate_extraction.py — answer cleaning + candidate extraction
  - multi_candidate_answer_consistency_voter.py — consistency voting + LLM arbitration
  - question_answer_format_hint_parsing_and_alignment.py — format hint parsing + post-processing
  - search_agent_shared_constants_and_stopwords.py — shared constants (stopwords, refusal phrases)

Legacy pipeline modules (used by src/core/main_agent.py, older pipeline path):
  - answer_generator.py — final answer formatting for competition output
  - executor.py — web interaction / tool execution stub
  - planner_agent.py — legacy planning agent
  - reasoner.py — legacy reasoning agent
  - validator.py — legacy validation agent

Backward-compatibility shims (re-export only, kept per 代码命名与重构规范.md):
  - answer_processing.py → llm_answer_cleaning_and_candidate_extraction.py
  - evidence_scoring.py → search_result_relevance_scoring_and_ranking.py
  - query_generation.py → search_query_parsing_and_generation.py
  - search_constants.py → search_agent_shared_constants_and_stopwords.py
  - constrained_search.py → constrained_multi_hop_search_agent.py
"""
