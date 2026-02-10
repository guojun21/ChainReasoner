"""Multi-hop search agent — orchestrates the full question-answering pipeline.

Why: Competition questions require chained reasoning (A -> B -> C).
This agent coordinates: LLM knowledge -> search hop-1 -> search hop-2
-> answer selection -> reverse verification, producing the best
possible answer from multiple evidence sources.
"""

import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agents.search_query_parsing_and_generation import (
    extract_structured_clues_from_question_text,
    generate_search_queries_from_question,
    classify_question_into_answer_type_category,
)
from src.agents.search_result_relevance_scoring_and_ranking import (
    rank_search_results_by_relevance_and_truncate,
    deduplicate_and_select_top_evidence_items,
    format_evidence_items_into_llm_readable_text,
)
from src.agents.llm_answer_cleaning_and_candidate_extraction import (
    clean_and_validate_raw_llm_answer_text,
    extract_structured_answer_candidates_from_evidence,
    select_highest_frequency_candidate_as_answer,
    check_if_answer_is_refusal_or_unknown_placeholder,
    verify_answer_against_evidence_using_llm,
)
from src.agents.multi_candidate_answer_consistency_voter import (
    select_final_answer_with_consistency_voting,
)


class ConstrainedMultiHopSearchAgent:
    """Iterative multi-hop search agent with LLM-augmented reasoning."""

    def __init__(
        self,
        search_fn: Callable[..., Dict[str, Any]],
        max_queries: int = 4,
        per_query_delay: float = 0.2,
        max_results_per_query: int = 8,
        max_evidence: int = 6,
        adaptive_threshold_n: float = 0.5,
        rewrite_fn: Optional[Callable[[str], str]] = None,
        llm_answer_fn: Optional[Callable[[str, str], str]] = None,
        llm_verify_fn: Optional[Callable[[str, str, str], Tuple[str, float]]] = None,
        llm_summarize_fn: Optional[Callable[[str], str]] = None,
        llm_decompose_fn: Optional[Callable[[str], str]] = None,
        llm_knowledge_fn: Optional[Callable[[str], str]] = None,
    ):
        self.search_fn = search_fn
        self.max_queries = max_queries
        self.per_query_delay = per_query_delay
        self.max_results_per_query = max_results_per_query
        self.max_evidence = max_evidence
        self.adaptive_threshold_n = adaptive_threshold_n
        self.rewrite_fn = rewrite_fn
        self.llm_answer_fn = llm_answer_fn
        self.llm_verify_fn = llm_verify_fn
        self.llm_summarize_fn = llm_summarize_fn
        self.llm_decompose_fn = llm_decompose_fn
        self.llm_knowledge_fn = llm_knowledge_fn
        self.llm_arbitrate_fn: Optional[Callable[[str, str], str]] = None
        self.trace_logger = None
        self._cache: Dict[str, List[Dict[str, Any]]] = {}

    # ── Search execution ────────────────────────────────────────────────

    def _perform_search_with_result_caching(self, query: str) -> Tuple[List[Dict[str, Any]], str]:
        """Execute search with per-query cache to avoid duplicate API calls."""
        if query in self._cache:
            return self._cache[query], query
        try:
            result = self.search_fn(query, {"original_query": query, "rewritten_query": query})
        except TypeError:
            result = self.search_fn(query)
        results = result.get("results", []) if isinstance(result, dict) else []
        self._cache[query] = results
        time.sleep(self.per_query_delay)
        return results, query

    def _execute_single_round_of_parallel_searches(self, queries: List[str]) -> Tuple[List[str], List[Dict], List[Dict]]:
        """Run one round of searches across all queries.

        Returns (all_text_snippets, all_ranked_results, search_traces).
        """
        all_texts, all_ranked, traces = [], [], []
        for query in queries:
            results, rewritten = self._perform_search_with_result_caching(query)
            ranked = rank_search_results_by_relevance_and_truncate(rewritten, results, self.max_results_per_query)
            traces.append({"query": query, "rewritten": rewritten, "count": len(ranked)})
            all_ranked.extend(ranked)
            for item in ranked:
                all_texts.append(f"{item.get('title', '')} {item.get('content', '')}")
        return all_texts, all_ranked, traces

    def _generate_second_hop_follow_up_queries(self, question: str, hop1_answer: str) -> List[str]:
        """Build follow-up queries using the hop-1 intermediate answer.

        Why: Multi-hop questions require chained search — the answer from
        the first round becomes a search term for the second round.
        """
        if not self.llm_decompose_fn or not hop1_answer:
            return []
        hop2_prompt = (
            f"Based on a first round of research, we found this intermediate answer: \"{hop1_answer}\"\n\n"
            f"Original question: {question}\n\n"
            f"Generate 1-2 SPECIFIC follow-up search queries that use the intermediate answer "
            f"to find the FINAL answer to the original question. "
            f"Include the key entity/name from the intermediate answer in each query.\n"
            f"Output ONLY the search queries, one per line."
        )
        try:
            raw_response = self.llm_decompose_fn(hop2_prompt)
            if not raw_response:
                return []
            queries = []
            for line in raw_response.strip().split("\n"):
                line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
                line = re.sub(r"^[-\u2022]\s*", "", line).strip().strip('"').strip("'")
                if line and len(line) >= 4:
                    queries.append(line)
            return queries[:2]
        except Exception:
            return []

    # ── Main orchestration ──────────────────────────────────────────────

    def answer(self, question: str) -> Dict[str, Any]:
        """Full pipeline: knowledge -> hop-1 search -> hop-2 search -> verify."""
        parsed = extract_structured_clues_from_question_text(question)
        queries = generate_search_queries_from_question(question, parsed, self.max_queries, self.llm_decompose_fn)
        reasoning_steps = [f"Queries: {queries}", f"Query count: {len(queries)}"]

        # Phase 0: LLM knowledge-only answer
        knowledge_answer = self._run_knowledge_only_answer_phase(question, reasoning_steps)

        # Phase 1: First search round (hop-1)
        all_texts, all_ranked, search_traces = self._execute_single_round_of_parallel_searches(queries)
        evidence = deduplicate_and_select_top_evidence_items(all_ranked, self.max_evidence)
        evidence_text = format_evidence_items_into_llm_readable_text(evidence)
        question_type = classify_question_into_answer_type_category(question)
        candidates = extract_structured_answer_candidates_from_evidence(all_texts)
        heuristic_answer = select_highest_frequency_candidate_as_answer(question, candidates, question_type)

        # Phase 2: Search-augmented LLM answer (hop-1)
        search_answer = self._run_search_augmented_answer_phase(question, knowledge_answer,
                                                     evidence_text, all_ranked)

        # Phase 2.5: Hop-2 iterative search
        hop2_answer = self._run_second_hop_deep_search_phase(question, search_answer or knowledge_answer,
                                       evidence, evidence_text, search_traces, reasoning_steps)

        # Phase 3: Select best answer via consistency voting (P1-a upgrade)
        candidate_dict = {}
        if hop2_answer:
            candidate_dict["hop2"] = hop2_answer
        if search_answer:
            candidate_dict["search"] = search_answer
        if knowledge_answer:
            candidate_dict["knowledge"] = knowledge_answer
        if heuristic_answer:
            candidate_dict["heuristic"] = heuristic_answer

        final_answer, answer_source, voting_trace = select_final_answer_with_consistency_voting(
            question=question,
            candidate_dict=candidate_dict,
            llm_arbitrate_fn=self.llm_arbitrate_fn,
            trace_logger=self.trace_logger,
        )
        reasoning_steps.append(f"Voting decision: {voting_trace.get('decision', 'unknown')}")

        # Phase 4: Reverse verification
        final_answer = self._run_answer_verification_phase(question, final_answer, answer_source,
                                          search_answer, knowledge_answer, reasoning_steps)

        reasoning_steps.extend([
            f"Evidence count: {len(evidence)}",
            f"Knowledge answer: {knowledge_answer}",
            f"Search answer: {search_answer}",
            f"Hop-2 answer: {hop2_answer}",
            f"Final answer: {final_answer}",
        ])
        return {"answer": final_answer, "reasoning_steps": reasoning_steps, "search_traces": search_traces}

    # ── Phase helpers ───────────────────────────────────────────────────

    def _run_knowledge_only_answer_phase(self, question: str, reasoning_steps: List[str]) -> str:
        """Phase 0: Ask LLM to answer from its own knowledge first."""
        if not self.llm_knowledge_fn:
            return ""
        try:
            raw = self.llm_knowledge_fn(question)
            processed = clean_and_validate_raw_llm_answer_text(raw)
            if processed:
                reasoning_steps.append(f"Knowledge answer: {processed}")
                return processed
        except Exception:
            pass
        return ""

    def _run_search_augmented_answer_phase(self, question: str, knowledge_answer: str,
                                evidence_text: str, all_ranked: List[Dict]) -> str:
        """Phase 2: Combine search evidence with LLM to extract answer."""
        if not self.llm_answer_fn or not evidence_text:
            return ""

        context = evidence_text
        if knowledge_answer:
            context = (
                f"NOTE: A preliminary answer '{knowledge_answer}' was generated from LLM knowledge alone. "
                f"This may have incorrect formatting or entity names. "
                f"Always verify and correct using the web search evidence below.\n\n"
                f"Web search evidence:\n{evidence_text}"
            )

        answer = clean_and_validate_raw_llm_answer_text(self.llm_answer_fn(question, context[:12000]))
        if answer:
            return answer

        # Extended evidence pass — include more result titles
        title_summary = "\n".join(
            f"- {item.get('title', '')}: {item.get('content', '')[:200]}"
            for item in all_ranked[:15] if item.get("title")
        )
        extended = f"{evidence_text}\n\nAdditional search results:\n{title_summary}"
        if knowledge_answer:
            extended = f"Preliminary answer from reasoning: {knowledge_answer}\n\n{extended}"
        return clean_and_validate_raw_llm_answer_text(self.llm_answer_fn(question, extended[:12000]))

    def _run_second_hop_deep_search_phase(self, question: str, hop1_answer: str,
                    evidence: List[Dict], evidence_text: str,
                    search_traces: List[Dict], reasoning_steps: List[str]) -> str:
        """Phase 2.5: Second search round using hop-1 intermediate answer."""
        if not hop1_answer or check_if_answer_is_refusal_or_unknown_placeholder(hop1_answer) or not self.llm_decompose_fn:
            return ""
        hop2_queries = self._generate_second_hop_follow_up_queries(question, hop1_answer)
        if not hop2_queries:
            return ""

        reasoning_steps.append(f"Hop-2 queries: {hop2_queries}")
        _, hop2_ranked, hop2_traces = self._execute_single_round_of_parallel_searches(hop2_queries)
        search_traces.extend(hop2_traces)

        combined = deduplicate_and_select_top_evidence_items(
            evidence + deduplicate_and_select_top_evidence_items(hop2_ranked), self.max_evidence)
        combined_text = format_evidence_items_into_llm_readable_text(combined)

        if not self.llm_answer_fn or not combined_text:
            return ""
        hop2_context = (
            f"Original question: {question}\n\n"
            f"First-round intermediate answer: {hop1_answer}\n\n"
            f"Combined evidence from two rounds of search:\n{combined_text}"
        )
        processed = clean_and_validate_raw_llm_answer_text(self.llm_answer_fn(question, hop2_context[:12000]))
        if processed:
            reasoning_steps.append(f"Hop-2 answer: {processed}")
        return processed

    @staticmethod
    def _select_best_answer_from_all_candidates(hop2: str, search: str, knowledge: str, heuristic: str) -> Tuple[str, str]:
        """Legacy Phase 3: Priority chain — hop2 > search > knowledge > heuristic.

        Why kept: Used as internal fallback when consistency voting module is unavailable.
        P1-a replaced the main call-site with select_final_answer_with_consistency_voting().
        """
        for answer, source in [(hop2, "hop2"), (search, "search"),
                               (knowledge, "knowledge"), (heuristic, "heuristic")]:
            if answer and not check_if_answer_is_refusal_or_unknown_placeholder(answer):
                return answer, source
        return "Unknown", "none"

    def _run_answer_verification_phase(self, question: str, final_answer: str, answer_source: str,
                      search_answer: str, knowledge_answer: str,
                      reasoning_steps: List[str]) -> str:
        """Phase 4: Reverse verification — search for the answer to confirm it."""
        if final_answer == "Unknown" or answer_source not in ("hop2", "search"):
            return final_answer
        if not self.search_fn or not self.llm_verify_fn:
            return final_answer
        try:
            verify_query = f"{final_answer} {question[:80]}"
            verify_result = self.search_fn(verify_query, {"original_query": verify_query})
            verify_items = verify_result.get("results", []) if isinstance(verify_result, dict) else []
            if verify_items:
                verify_text = format_evidence_items_into_llm_readable_text(verify_items[:5])
                if verify_text:
                    is_valid = verify_answer_against_evidence_using_llm(
                        question, final_answer, verify_text, self.llm_verify_fn)
                    reasoning_steps.append(f"Reverse verify: {is_valid}")
                    if not is_valid:
                        reasoning_steps.append(f"Verification failed for '{final_answer}', trying fallback")
                        if answer_source == "hop2" and search_answer and not check_if_answer_is_refusal_or_unknown_placeholder(search_answer):
                            return search_answer
                        if knowledge_answer and not check_if_answer_is_refusal_or_unknown_placeholder(knowledge_answer):
                            return knowledge_answer
        except Exception:
            pass
        return final_answer
