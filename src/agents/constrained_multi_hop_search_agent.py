"""Multi-hop search agent — orchestrates the full question-answering pipeline.

Why: Competition questions require chained reasoning (A -> B -> C).
This agent coordinates: structured hop planning -> per-hop iterative retrieval
-> evidence fusion -> answer selection -> verification, producing the best
possible answer from multiple evidence sources.

P0-d upgrade (from everything-claude-code + analysis_claude_code):
  - Iterative retrieval per hop (DISPATCH → EVALUATE → REFINE → LOOP)
  - Merged 3 LLM calls (validate + reflect + completion) into 1 unified call
  - Context isolation: each hop gets focused context (target + summary + evidence)
  - Reduced LLM calls from ~20 to ~12 per question

P0-e fix (from analysis_claude_code v3 sub-agent isolation):
  - Intermediate hops now receive hop_target as their extraction question,
    not the original user question.  This prevents the LLM from attempting
    to answer the final question prematurely in intermediate steps.
  - Last hop still receives the original question (for format hint extraction).
  - Enhanced per-cycle logging: extraction_question type + content tracked.

References:
  - Research_Agent: parse_multi_hop_plan, validate_hop_result, fuse_hop_evidence
  - open_deep_research: think_tool (inter-hop reflection)
  - Enhancing-Multi-Hop-QA: SelfVerifier
  - everything-claude-code: iterative retrieval pattern (skills/iterative-retrieval)
  - analysis_claude_code v3: sub-agent context isolation
"""

import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from src.agents.search_query_parsing_and_generation import (
    extract_structured_clues_from_question_text,
    generate_search_queries_from_question,
    classify_question_into_answer_type_category,
)
from src.mini_agent.language_router import (
    classify_question_geographic_language_priority,
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
from src.agents.structured_multi_hop_reasoning_planner import (
    generate_structured_multi_hop_plan,
    generate_queries_for_hop,
)
from src.agents.per_hop_result_validator_and_corrector import (
    evaluate_hop_result_and_decide_next_action,
    generate_refined_hop_queries_from_evaluation,
    MAX_RETRIEVAL_CYCLES_PER_HOP,
)
from src.agents.question_domain_classifier import classify_question_domain
from src.scratchpad.per_question_evidence_scratchpad import PerQuestionEvidenceScratchpad


class ConstrainedMultiHopSearchAgent:
    """Iterative multi-hop search agent with structured planning and validation."""

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
        scratchpad_base_dir: str = "",
        enable_minimal_drift_guard: bool = True,
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
        # Injected by API server after construction
        self.llm_arbitrate_fn: Optional[Callable[[str, str], str]] = None
        self.llm_generic_fn: Optional[Callable[[str, str], str]] = None
        self.llm_fuse_fn: Optional[Callable[[str, list], str]] = None
        self.llm_refute_extract_fn: Optional[Callable[[str, str], str]] = None
        self.trace_logger = None
        self._cache: Dict[str, List[Dict[str, Any]]] = {}
        # Per-question local knowledge base (scratchpad)
        self._scratchpad_base_dir = scratchpad_base_dir
        self._scratchpad: Optional[PerQuestionEvidenceScratchpad] = None
        # Lightweight drift guard: when repeated refine cycles drift to unrelated
        # entities, inject anchor terms to pull queries back to the target track.
        self._enable_minimal_drift_guard = enable_minimal_drift_guard

    # ── Search execution ────────────────────────────────────────────────

    def _perform_search_with_result_caching(
        self, query: str, domain: str = "",
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Execute search with per-query cache to avoid duplicate API calls.

        Wing-architecture: passes ``domain`` in meta so the dispatcher can
        route to the optimal search engine for the question type.
        """
        cache_key = f"{domain}:{query}" if domain else query
        if cache_key in self._cache:
            if self.trace_logger:
                self.trace_logger.record_event("search_cache_hit", f"q='{query[:60]}' domain={domain}")
            return self._cache[cache_key], query
        meta = {"original_query": query, "rewritten_query": query}
        if domain:
            meta["domain"] = domain
        # Propagate question-level language priority to search dispatcher
        # so it overrides character-level detection with smarter routing.
        lang_prio = getattr(self, "_language_priority", "bilingual_equal")
        if lang_prio != "bilingual_equal":
            meta["language_priority"] = lang_prio
            if self.trace_logger:
                self.trace_logger.record_event("language_priority_set", f"priority={lang_prio} q='{query[:60]}'")
        try:
            result = self.search_fn(query, meta)
        except TypeError:
            if self.trace_logger:
                self.trace_logger.record_event("search_fn_signature_fallback", f"q='{query[:60]}' — search_fn(query, meta) failed, falling back to search_fn(query)")
            result = self.search_fn(query)
        results = result.get("results", []) if isinstance(result, dict) else []
        self._cache[cache_key] = results
        time.sleep(self.per_query_delay)
        return results, query

    def _execute_single_round_of_parallel_searches(
        self, queries: List[str], domain: str = "",
    ) -> Tuple[List[str], List[Dict], List[Dict]]:
        """Run one round of searches across all queries.

        Wing-architecture: passes ``domain`` to the search cache so the
        dispatcher selects the best engine for the question type.

        Returns (all_text_snippets, all_ranked_results, search_traces).
        """
        all_texts, all_ranked, traces = [], [], []
        for query in queries:
            results, rewritten = self._perform_search_with_result_caching(query, domain=domain)
            ranked = rank_search_results_by_relevance_and_truncate(rewritten, results, self.max_results_per_query)
            traces.append({"query": query, "rewritten": rewritten, "count": len(ranked), "domain": domain})
            all_ranked.extend(ranked)
            for item in ranked:
                all_texts.append(f"{item.get('title', '')} {item.get('content', '')}")
        return all_texts, all_ranked, traces

    # ── Context building (P0-d: isolated per-hop context) ────────────

    def _build_isolated_hop_context(
        self,
        hop_target: str,
        hop_evidence_text: str,
        previous_hop_summaries: List[str],
    ) -> str:
        """Build focused context for a single hop's LLM answer extraction.

        Why (from analysis_claude_code v3 sub-agent isolation): Each hop should
        get ONLY its target, a brief summary of prior hops (max 200 chars each),
        and its own search evidence.  Full accumulated evidence is reserved for
        the evidence fusion stage, not individual hops.  This prevents LLM
        attention from being diluted by irrelevant prior-hop details.
        """
        parts = []
        if previous_hop_summaries:
            # Wing-architecture: summaries now contain entity-highlighted text;
            # pass more context (up to 400 chars) so precise entity names survive.
            trimmed = [s[:400] for s in previous_hop_summaries[-3:]]
            parts.append("Previous findings (entities in **bold**):\n" + "\n".join(f"- {s}" for s in trimmed))
        parts.append(f"Current search objective: {hop_target}")
        parts.append(f"Search evidence:\n{hop_evidence_text}")
        return "\n\n".join(parts)

    def _extract_anchor_terms_for_drift_guard(
        self,
        hop_target: str,
        knowledge_answer: str = "",
    ) -> List[str]:
        """Extract anchor terms used to pull refine queries back on track.

        Why: When retrieval drifts to a wrong entity, adding stable anchor terms
        (hop target constraints + knowledge answer entity) is a low-risk way to
        recover without changing the overall algorithm.
        """
        seeds: List[str] = []
        if knowledge_answer and not check_if_answer_is_refusal_or_unknown_placeholder(knowledge_answer):
            seeds.append(knowledge_answer.strip())
        seeds.append(hop_target.strip())

        terms: List[str] = []
        for seed in seeds:
            if not seed:
                continue
            # Chinese phrase anchors
            terms.extend(re.findall(r"[\u4e00-\u9fff]{2,12}", seed))
            # English token anchors
            terms.extend(re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,32}", seed))

        # Keep order, drop duplicates and overly generic terms
        blocked = {"search", "query", "project", "entity", "company", "target"}
        deduped: List[str] = []
        for t in terms:
            key = t.lower()
            if key in blocked:
                continue
            if t not in deduped:
                deduped.append(t)
        return deduped[:6]

    @staticmethod
    def _check_if_refine_result_is_drift(
        alternative_entities: List[str],
        anchor_terms: List[str],
    ) -> bool:
        """Return True if alternatives look unrelated to current anchor terms."""
        if not alternative_entities or not anchor_terms:
            return False
        alt_lower = [a.lower() for a in alternative_entities if a]
        anchor_lower = [t.lower() for t in anchor_terms if t]
        if not alt_lower or not anchor_lower:
            return False
        # Drift if none of the alternative entities contains any anchor token.
        return all(
            not any(anchor in alt for anchor in anchor_lower)
            for alt in alt_lower
        )

    def _inject_anchor_terms_into_refined_queries(
        self,
        refined_queries: List[str],
        anchor_terms: List[str],
        hop_target: str,
        max_queries: int,
    ) -> List[str]:
        """Inject anchor terms into refined queries to reduce entity drift."""
        if not refined_queries:
            refined_queries = [hop_target]
        if not anchor_terms:
            return list(dict.fromkeys(refined_queries))[:max_queries]

        anchored = list(refined_queries)
        primary_anchor = " ".join(anchor_terms[:2]).strip()
        if primary_anchor:
            anchored.append(f"{primary_anchor} {hop_target}".strip())

        # For the first two refined queries, append one anchor token if missing.
        for i, q in enumerate(refined_queries[:2]):
            if not any(a.lower() in q.lower() for a in anchor_terms):
                anchored.append(f"{q} {anchor_terms[0]}".strip())

        return list(dict.fromkeys(anchored))[:max_queries]

    # ── Hop chain backtrack: re-search when candidate is rejected ──────

    def _attempt_hop_chain_backtrack_on_candidate_rejection(
        self,
        question: str,
        rejected_entities: List[str],
        rejection_reason: str,
        original_hop_target: str,
        hop_domain: str,
        reasoning_steps: List[str],
        search_traces: List[Dict],
        all_ranked: List[Dict],
        all_texts: List[str],
    ) -> Tuple[str, str, List[Dict]]:
        """Backtrack to Hop 1 and re-search with exclusion constraints.

        Why (from Q99 log analysis): When Hop 2+ discovers the candidate from
        Hop 1 fails a hard constraint (e.g. Peter Ndlovu born 1973, outside
        1988-1995), the system has no way to go back and find a different
        candidate.  It just keeps refining within the current hop, eventually
        settling on another wrong candidate (Quinton Fortune, born 1977).

        This method re-executes a search round with:
        1. Exclusion terms (-Ndlovu -Fortune) to avoid known-wrong candidates
        2. Constraint-focused queries derived from the rejection reason
        3. The original hop target to stay on-topic

        References:
          - Research_Agent: correct_hop_result() — re-execute tool on failure
          - Weaver: knowledge gap analysis — identify missing aspects for next round
        """
        import logging
        logger = logging.getLogger(__name__)

        reasoning_steps.append(
            f"BACKTRACK: Rejected entities {rejected_entities}, "
            f"reason: {rejection_reason[:200]}"
        )

        # Build exclusion-augmented queries using LLM
        exclusion_suffix = " ".join(f"-{e}" for e in rejected_entities[:3] if e)
        backtrack_queries: List[str] = []

        if self.llm_generic_fn:
            try:
                backtrack_prompt = (
                    f"The following candidates were REJECTED for this question:\n"
                    f"Question: {question}\n"
                    f"Rejected: {', '.join(rejected_entities)}\n"
                    f"Reason: {rejection_reason[:500]}\n\n"
                    f"Generate 3-4 NEW search queries that:\n"
                    f"1. Explicitly EXCLUDE the rejected candidates\n"
                    f"2. Focus on the constraints that rejected candidates failed\n"
                    f"3. Search for ALTERNATIVE candidates\n\n"
                    f"Output ONLY search queries, one per line. No numbering or explanations."
                )
                raw = self.llm_generic_fn(
                    "You are a search query expert. Generate precise queries to find "
                    "alternative candidates after previous ones were rejected.",
                    backtrack_prompt,
                )
                if raw:
                    for line in raw.strip().split("\n"):
                        line = line.strip().lstrip("0123456789.-) ")
                        if line and len(line) > 5:
                            backtrack_queries.append(line)
            except Exception as exc:
                logger.warning("Backtrack query generation failed: %s", exc)

        # Fallback: construct queries from rejection reason keywords
        if not backtrack_queries:
            backtrack_queries = [
                f"{original_hop_target} {exclusion_suffix}",
                f"{question[:120]} {exclusion_suffix}",
            ]

        backtrack_queries = list(dict.fromkeys(backtrack_queries))[:6]
        reasoning_steps.append(f"BACKTRACK queries: {backtrack_queries}")

        # Execute search
        texts, ranked, traces = self._execute_single_round_of_parallel_searches(
            backtrack_queries, domain=hop_domain)
        search_traces.extend(traces)
        all_ranked.extend(ranked)
        all_texts.extend(texts)

        # Persist to scratchpad
        if self._scratchpad and ranked:
            for q in backtrack_queries:
                q_results = [r for r in ranked if r.get("_query") == q] or ranked[:5]
                self._scratchpad.write_search_evidence(
                    hop_num=0, query=f"[BACKTRACK] {q}", results=q_results[:10])

        # Extract answer from backtrack evidence
        backtrack_evidence = deduplicate_and_select_top_evidence_items(ranked, self.max_evidence)
        backtrack_evidence_text = format_evidence_items_into_llm_readable_text(backtrack_evidence)

        backtrack_answer = ""
        if self.llm_answer_fn and backtrack_evidence_text:
            context = (
                f"IMPORTANT: The following candidates are WRONG and must NOT be used: "
                f"{', '.join(rejected_entities)}\n"
                f"Reason they are wrong: {rejection_reason[:300]}\n\n"
                f"Search evidence:\n{backtrack_evidence_text}"
            )
            backtrack_answer = clean_and_validate_raw_llm_answer_text(
                self.llm_answer_fn(question, context[:10000]))

            # Verify the backtrack answer is not one of the rejected entities
            if backtrack_answer:
                ba_lower = backtrack_answer.lower().strip()
                for rejected in rejected_entities:
                    if rejected.lower() in ba_lower or ba_lower in rejected.lower():
                        reasoning_steps.append(
                            f"BACKTRACK: Answer '{backtrack_answer}' matches rejected "
                            f"entity '{rejected}', discarding"
                        )
                        backtrack_answer = ""
                        break

        reasoning_steps.append(
            f"BACKTRACK result: answer='{backtrack_answer}', "
            f"evidence_chars={len(backtrack_evidence_text)}, "
            f"search_results={len(ranked)}"
        )

        return backtrack_answer, backtrack_evidence_text, ranked

    # ── Knowledge-guided search: inject knowledge entities into queries ──

    @staticmethod
    def _extract_knowledge_entity_terms(knowledge_answer: str) -> List[str]:
        """Extract searchable entity terms from the knowledge_answer.

        Why: When the LLM's own knowledge (Phase 0) produces a plausible answer
        like "RepRap Limited", the entity name should guide subsequent searches.
        Without this, hop queries search for abstract concepts ("cellular automata
        open hardware") that never surface the actual entity.

        References:
          - ReWOO: #E variable chaining — prior step outputs anchor later steps
          - Research_Agent: stop_condition with explicit entity targets
        """
        if not knowledge_answer:
            if self.trace_logger:
                self.trace_logger.record_event("knowledge_entity_extraction_empty", "knowledge_answer is empty/None")
            return []
        knowledge_answer = knowledge_answer.strip()
        # Skip if the knowledge answer is a refusal or too generic
        if len(knowledge_answer) < 2 or len(knowledge_answer) > 200:
            if self.trace_logger:
                self.trace_logger.record_event("knowledge_entity_extraction_empty", f"knowledge_answer length {len(knowledge_answer)} out of range [2,200]")
            return []

        terms: List[str] = []
        # Full entity name as a single term (most valuable)
        terms.append(knowledge_answer)
        # Individual significant tokens (for partial matching)
        for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,32}", knowledge_answer):
            if token.lower() not in {
                "the", "and", "for", "ltd", "inc", "corp", "limited",
                "company", "group", "unknown", "none", "project",
            }:
                terms.append(token)
        # Chinese tokens
        for token in re.findall(r"[\u4e00-\u9fff]{2,12}", knowledge_answer):
            terms.append(token)
        # Deduplicate while preserving order
        seen: set = set()
        deduped: List[str] = []
        for t in terms:
            if t.lower() not in seen:
                seen.add(t.lower())
                deduped.append(t)
        return deduped[:5]

    @staticmethod
    def _inject_knowledge_entities_into_queries(
        queries: List[str],
        knowledge_entities: List[str],
        max_queries: int,
    ) -> List[str]:
        """Inject knowledge entity terms into hop queries to guide search.

        Why (from run_20260216_000139 log analysis): The 5-hop search spent 1325s
        searching for abstract concepts and found nothing.  Meanwhile, the
        knowledge_answer correctly identified "RepRap Limited" in 11s.  By
        injecting this entity into search queries, we guide the search engine
        to the right neighborhood immediately.

        Strategy:
        1. Keep all original queries (they may find complementary evidence)
        2. Add knowledge-anchored queries that combine entity + hop context
        3. For queries that don't mention the entity, append the primary entity
        """
        if not knowledge_entities or not queries:
            if self.trace_logger:
                self.trace_logger.record_event("knowledge_entity_injection_skipped",
                    f"entities={len(knowledge_entities) if knowledge_entities else 0} queries={len(queries) if queries else 0}")
            return queries[:max_queries]

        primary_entity = knowledge_entities[0]  # Full entity name
        augmented = list(queries)

        # Add a direct entity search query
        augmented.append(primary_entity)

        # For each original query, create an entity-augmented variant if the
        # entity is not already mentioned
        for q in queries[:3]:
            if not any(e.lower() in q.lower() for e in knowledge_entities[:2]):
                augmented.append(f"{primary_entity} {q}".strip())

        return list(dict.fromkeys(augmented))[:max_queries]

    # ── Iterative retrieval per hop (P0-d: from everything-claude-code) ──

    def _execute_iterative_retrieval_for_single_hop(
        self,
        question: str,
        hop_idx: int,
        hop: Dict[str, Any],
        hops: List[Dict[str, Any]],
        total_stop_condition: str,
        previous_hop_results: List[str],
        hop_answers: List[str],
        reasoning_steps: List[str],
        search_traces: List[Dict],
        all_ranked: List[Dict],
        all_texts: List[str],
        knowledge_answer: str = "",
    ) -> Tuple[str, str, bool, Dict[str, Any]]:
        """Execute one hop with iterative retrieval (DISPATCH → EVALUATE → REFINE → LOOP).

        Why (from everything-claude-code skills/iterative-retrieval/SKILL.md):
        Instead of 'search once, validate, maybe retry once with separate logic',
        use a unified iterative loop that evaluates after each search round and
        refines queries based on what is specifically missing.

        Returns:
            (hop_answer, hop_evidence_text, should_stop_chain, hop_meta)
        """
        hop_started_ts = time.time()
        hop_num = hop.get("hop_num", hop_idx + 1)
        hop_target = hop.get("target", "搜索相关信息")
        hop_stop_condition = hop.get("stop_condition", "获取到相关信息")
        hop_count = len(hops)
        is_last_hop = (hop_idx == len(hops) - 1)
        max_q = max(self.max_queries, 6)
        # Pass knowledge_answer to ALL hops for anchor term extraction, not just
        # hop 0.  Why: knowledge entities remain relevant throughout the chain —
        # later hops searching for "commercial entity" still benefit from knowing
        # the project is "RepRap".
        anchor_terms = self._extract_anchor_terms_for_drift_guard(
            hop_target=hop_target,
            knowledge_answer=knowledge_answer,
        )
        drift_refine_streak = 0

        # Wing-architecture: Classify hop domain for search engine routing.
        # Use the domain from hop plan if available, otherwise classify via LLM.
        hop_domain = hop.get("domain", "")
        if not hop_domain:
            hop_domain = classify_question_domain(
                question, hop_target=hop_target, llm_fn=self.llm_generic_fn)
        reasoning_steps.append(
            f"Hop {hop_num} domain: {hop_domain}"
        )

        # ── Knowledge-guided search: extract entities from knowledge_answer ──
        # Why (from run_20260216_000139 analysis): knowledge_answer correctly
        # identified "RepRap Limited" in 11s, but 5 hops spent 1325s searching
        # for abstract concepts.  Injecting the knowledge entity into queries
        # guides search to the right neighborhood immediately.
        knowledge_entities = self._extract_knowledge_entity_terms(knowledge_answer)

        # Generate initial queries for this hop (P0-f: bilingual ZH+EN queries)
        if hop_idx == 0:
            parsed = extract_structured_clues_from_question_text(question)
            base_queries = generate_search_queries_from_question(
                question, parsed, self.max_queries, self.llm_decompose_fn)
            hop_queries = generate_queries_for_hop(
                question, hop_target, previous_hop_results, self.llm_generic_fn)
            # Allow more queries since bilingual generation produces both ZH+EN
            queries = list(dict.fromkeys(base_queries + hop_queries))[:max_q]
        else:
            hop_queries = generate_queries_for_hop(
                question, hop_target, previous_hop_results, self.llm_generic_fn)
            if hop_answers and hop_answers[-1]:
                followup = self._generate_second_hop_follow_up_queries(
                    question, hop_answers[-1])
                hop_queries = list(dict.fromkeys(hop_queries + followup))
            queries = hop_queries[:max_q]

        # Inject knowledge entities into queries for ALL hops (not just hop 0)
        if knowledge_entities:
            queries = self._inject_knowledge_entities_into_queries(
                queries, knowledge_entities, max_q)
            reasoning_steps.append(
                f"Hop {hop_num} knowledge-guided: injected entities "
                f"{knowledge_entities[:3]} into {len(queries)} queries"
            )

        # Iterative retrieval loop: DISPATCH → EVALUATE → REFINE → LOOP
        hop_answer = ""
        hop_evidence_text = ""
        hop_all_ranked = []
        hop_stop_reason = "max_cycles_or_continue"
        last_evaluation: Dict[str, Any] = {}
        executed_cycles = 0

        def _record_hop_cycle_trace(cycle_index: int, details: Dict[str, Any]) -> None:
            if self.trace_logger and hasattr(self.trace_logger, "record_per_hop_retrieval_cycle_trace"):
                self.trace_logger.record_per_hop_retrieval_cycle_trace(
                    question=question,
                    hop_num=hop_num,
                    cycle=cycle_index + 1,
                    details=details,
                )

        for cycle in range(MAX_RETRIEVAL_CYCLES_PER_HOP):
            cycle_started_ts = time.time()
            executed_cycles = cycle + 1
            cycle_label = f"Hop {hop_num} cycle {cycle + 1}"
            reasoning_steps.append(f"{cycle_label} queries: {queries}")
            queries_before_refine = list(queries)
            local_retrieval_elapsed_ms = 0
            network_elapsed_ms = 0
            extraction_elapsed_ms = 0
            evaluation_elapsed_ms = 0
            refinement_triggered = False
            drift_detected = False
            queries_after_refine = list(queries_before_refine)
            anchor_terms_used: List[str] = []

            # ── LOCAL RETRIEVAL: Check scratchpad before network search ──
            # Why: Later hops often need evidence already fetched by earlier hops.
            # Querying the local BM25 index first avoids redundant API calls
            # and provides instant results from already-cleaned content.
            local_hits: list = []
            local_start_ts = time.time()
            if self._scratchpad and cycle == 0:
                for q in queries:
                    hits = self._scratchpad.search_local(q, top_k=3, min_score=1.0)
                    local_hits.extend(hits)
                if local_hits:
                    # Deduplicate by URL
                    seen_urls = set()
                    unique_local: list = []
                    for h in local_hits:
                        if h.get("url") not in seen_urls:
                            seen_urls.add(h.get("url"))
                            unique_local.append(h)
                    local_hits = unique_local[:5]
                    hop_all_ranked.extend(local_hits)
                    all_ranked.extend(local_hits)
                    reasoning_steps.append(
                        f"{cycle_label} local_kb: {len(local_hits)} hits from scratchpad "
                        f"(top_score={local_hits[0].get('_score', 0):.2f})")
            local_retrieval_elapsed_ms = int((time.time() - local_start_ts) * 1000)

            # DISPATCH: Execute search (Wing-architecture: domain-aware routing)
            network_start_ts = time.time()
            texts, ranked, traces = self._execute_single_round_of_parallel_searches(queries, domain=hop_domain)
            network_elapsed_ms = int((time.time() - network_start_ts) * 1000)
            search_traces.extend(traces)
            hop_all_ranked.extend(ranked)
            all_ranked.extend(ranked)
            all_texts.extend(texts)

            # ── PERSIST: Write search results to scratchpad ──
            if self._scratchpad and ranked:
                for q in queries:
                    # Find results that came from this query
                    q_results = [r for r in ranked if r.get("_query") == q] or ranked
                    self._scratchpad.write_search_evidence(
                        hop_num=hop_num, query=q, results=q_results[:10])

            # Build evidence for this cycle (accumulated across cycles within this hop)
            hop_evidence = deduplicate_and_select_top_evidence_items(hop_all_ranked, self.max_evidence)
            hop_evidence_text = format_evidence_items_into_llm_readable_text(hop_evidence)
            reasoning_steps.append(
                f"{cycle_label} evidence: {len(ranked)} new results, "
                f"{len(local_hits)} local, "
                f"{len(hop_all_ranked)} total, {len(hop_evidence_text)} chars"
            )

            # P0-e: Use hop-specific target for intermediate hops, original question
            # for final hop.  Why (from analysis_claude_code v3 sub-agent isolation):
            # Each hop should receive its own focused question to prevent the LLM
            # from attempting to answer the final question prematurely in
            # intermediate steps (e.g. returning "Stamp Stories" when the hop's
            # real objective is "identify the major space telescope").
            extraction_question = question if is_last_hop else hop_target
            if self.llm_answer_fn and hop_evidence_text:
                extraction_start_ts = time.time()
                isolated_context = self._build_isolated_hop_context(
                    hop_target, hop_evidence_text, previous_hop_results)
                reasoning_steps.append(
                    f"{cycle_label} extract: q={'original' if is_last_hop else 'hop_target'} "
                    f"→ '{extraction_question[:80]}'"
                )
                hop_answer = clean_and_validate_raw_llm_answer_text(
                    self.llm_answer_fn(extraction_question, isolated_context[:8000]))
                extraction_elapsed_ms = int((time.time() - extraction_start_ts) * 1000)
                reasoning_steps.append(
                    f"{cycle_label} extracted_answer: '{hop_answer[:100]}'"
                )

            # EVALUATE: Unified evaluation (validate + reflect + completion check)
            evaluation_start_ts = time.time()
            evaluation = evaluate_hop_result_and_decide_next_action(
                question=question,
                hop_num=hop_num,
                total_hops=hop_count,
                hop_target=hop_target,
                hop_stop_condition=hop_stop_condition,
                hop_result=hop_answer or hop_evidence_text[:1000],
                previous_hop_summaries="\n".join(previous_hop_results[-3:]),
                total_stop_condition=total_stop_condition,
                is_last_hop=is_last_hop,
                llm_fn=self.llm_generic_fn,
            )
            evaluation_elapsed_ms = int((time.time() - evaluation_start_ts) * 1000)
            last_evaluation = dict(evaluation)
            reasoning_steps.append(
                f"{cycle_label} eval: valid={evaluation['valid']}, "
                f"conf={evaluation['confidence']:.2f}, "
                f"action={evaluation['next_action']}, "
                f"entity={evaluation.get('extracted_entity', '')[:60]}"
            )

            # Use extracted entity from evaluation if we didn't get a hop_answer
            if evaluation.get("extracted_entity") and not hop_answer:
                hop_answer = evaluation["extracted_entity"]
                if self.trace_logger:
                    self.trace_logger.record_event("entity_extraction_fallback",
                        f"hop {hop_num}: using evaluation entity '{hop_answer[:60]}' as hop_answer was empty")

            # Check if chain is already complete (early termination)
            # P0-f: Raised threshold from 0.75 to 0.85 to reduce premature
            # termination — the old threshold let through half-confident guesses.
            if evaluation["is_chain_complete"] and evaluation["confidence"] >= 0.85:
                chain_answer = evaluation.get("chain_answer", "")
                if chain_answer and len(chain_answer) < 200:
                    hop_answer = chain_answer
                reasoning_steps.append(
                    f"Chain complete at hop {hop_num} cycle {cycle + 1} (conf={evaluation['confidence']:.2f})")
                hop_stop_reason = "chain_complete"
                _record_hop_cycle_trace(cycle, {
                    "hop_target": hop_target,
                    "hop_domain": hop_domain,
                    "queries_before_refine": queries_before_refine,
                    "queries_after_refine": queries_after_refine,
                    "local_retrieval": {
                        "enabled": bool(self._scratchpad and cycle == 0),
                        "elapsed_ms": local_retrieval_elapsed_ms,
                        "hits_count": len(local_hits),
                        "top_score": local_hits[0].get("_score", 0.0) if local_hits else 0.0,
                    },
                    "network_retrieval": {
                        "elapsed_ms": network_elapsed_ms,
                        "query_count": len(queries_before_refine),
                        "queries_used": queries_before_refine,
                        "total_snippets": len(texts),
                        "result_count": len(ranked),
                    },
                    "evidence": {
                        "hop_total_ranked_count": len(hop_all_ranked),
                        "evidence_chars": len(hop_evidence_text),
                    },
                    "extraction": {
                        "elapsed_ms": extraction_elapsed_ms,
                        "question_type": "original" if is_last_hop else "hop_target",
                        "hop_answer_preview": hop_answer[:120],
                    },
                    "evaluation": {
                        "elapsed_ms": evaluation_elapsed_ms,
                        "valid": evaluation.get("valid", False),
                        "confidence": evaluation.get("confidence", 0.0),
                        "extracted_entity": evaluation.get("extracted_entity", ""),
                        "alternative_entities": evaluation.get("alternative_entities", []),
                        "missing_info": evaluation.get("missing_info", ""),
                        "next_action": evaluation.get("next_action", ""),
                        "is_chain_complete": evaluation.get("is_chain_complete", False),
                    },
                    "refinement": {
                        "triggered": refinement_triggered,
                        "drift_detected": drift_detected,
                        "drift_refine_streak": drift_refine_streak,
                        "anchor_terms_used": anchor_terms_used,
                    },
                    "stop_reason": hop_stop_reason,
                    "cycle_elapsed_ms": int((time.time() - cycle_started_ts) * 1000),
                })
                return hop_answer, hop_evidence_text, True, {
                    "total_cycles": executed_cycles,
                    "stop_reason": hop_stop_reason,
                    "elapsed_ms": int((time.time() - hop_started_ts) * 1000),
                    "last_evaluation": last_evaluation,
                }

            # Decision: stop iterating, refine, or continue to next hop
            if evaluation["next_action"] == "sufficient":
                reasoning_steps.append(f"{cycle_label}: sufficient evidence")
                hop_stop_reason = "sufficient"
            elif evaluation["next_action"] == "refine" and cycle < MAX_RETRIEVAL_CYCLES_PER_HOP - 1:
                # REFINE: Generate better queries based on what is missing
                refinement_triggered = True
                alternatives = evaluation.get("alternative_entities", []) or []
                drift_detected = self._check_if_refine_result_is_drift(
                    alternative_entities=alternatives,
                    anchor_terms=anchor_terms,
                )
                drift_refine_streak = drift_refine_streak + 1 if drift_detected else 0
                if self.trace_logger:
                    self.trace_logger.record_event(
                        "drift_detected" if drift_detected else "drift_not_detected",
                        f"hop {hop_num} cycle {cycle+1}: streak={drift_refine_streak} alternatives={alternatives[:3]}")

                refined_queries = generate_refined_hop_queries_from_evaluation(
                    question=question,
                    hop_num=hop_num,
                    hop_target=hop_target,
                    current_result=hop_answer or hop_evidence_text[:400],
                    missing_info=evaluation.get("missing_info", ""),
                    reasoning=evaluation.get("reasoning", ""),
                    llm_fn=self.llm_generic_fn,
                )
                # P1 minimal drift guard: if repeated refine cycles keep drifting
                # away from anchor terms, inject anchors into next-round queries.
                if (self._enable_minimal_drift_guard
                        and drift_refine_streak >= 2
                        and anchor_terms):
                    anchor_terms_used = anchor_terms[:2]
                    refined_queries = self._inject_anchor_terms_into_refined_queries(
                        refined_queries=refined_queries,
                        anchor_terms=anchor_terms_used,
                        hop_target=hop_target,
                        max_queries=max_q,
                    )
                    reasoning_steps.append(
                        f"{cycle_label} drift_guard active: inject anchors {anchor_terms_used}"
                    )

                queries = refined_queries
                queries_after_refine = list(queries)
                hop_stop_reason = "refine"
                reasoning_steps.append(f"{cycle_label} refine → {queries_after_refine}")
            else:
                # Valid or last cycle — move on
                hop_stop_reason = "continue_next_hop"

            _record_hop_cycle_trace(cycle, {
                "hop_target": hop_target,
                "hop_domain": hop_domain,
                "queries_before_refine": queries_before_refine,
                "queries_after_refine": queries_after_refine,
                "local_retrieval": {
                    "enabled": bool(self._scratchpad and cycle == 0),
                    "elapsed_ms": local_retrieval_elapsed_ms,
                    "hits_count": len(local_hits),
                    "top_score": local_hits[0].get("_score", 0.0) if local_hits else 0.0,
                },
                "network_retrieval": {
                    "elapsed_ms": network_elapsed_ms,
                    "query_count": len(queries_before_refine),
                    "queries_used": queries_before_refine,
                    "total_snippets": len(texts),
                    "result_count": len(ranked),
                },
                "evidence": {
                    "hop_total_ranked_count": len(hop_all_ranked),
                    "evidence_chars": len(hop_evidence_text),
                },
                "extraction": {
                    "elapsed_ms": extraction_elapsed_ms,
                    "question_type": "original" if is_last_hop else "hop_target",
                    "hop_answer_preview": hop_answer[:120],
                },
                "evaluation": {
                    "elapsed_ms": evaluation_elapsed_ms,
                    "valid": evaluation.get("valid", False),
                    "confidence": evaluation.get("confidence", 0.0),
                    "extracted_entity": evaluation.get("extracted_entity", ""),
                    "alternative_entities": evaluation.get("alternative_entities", []),
                    "missing_info": evaluation.get("missing_info", ""),
                    "next_action": evaluation.get("next_action", ""),
                    "is_chain_complete": evaluation.get("is_chain_complete", False),
                },
                "refinement": {
                    "triggered": refinement_triggered,
                    "drift_detected": drift_detected,
                    "drift_refine_streak": drift_refine_streak,
                    "anchor_terms_used": anchor_terms_used,
                },
                "stop_reason": hop_stop_reason,
                "cycle_elapsed_ms": int((time.time() - cycle_started_ts) * 1000),
            })

            if evaluation["next_action"] == "sufficient":
                break
            if evaluation["next_action"] == "refine" and cycle < MAX_RETRIEVAL_CYCLES_PER_HOP - 1:
                continue
            break

        return hop_answer, hop_evidence_text, False, {
            "total_cycles": executed_cycles,
            "stop_reason": hop_stop_reason,
            "elapsed_ms": int((time.time() - hop_started_ts) * 1000),
            "last_evaluation": last_evaluation,
        }

    # ── Main orchestration (P0-d: iterative retrieval + merged eval) ──

    def answer(self, question: str) -> Dict[str, Any]:
        """Full pipeline: plan -> iterative hop retrieval -> fuse -> vote -> verify.

        P0-d flow (from everything-claude-code + analysis_claude_code):
          Phase 0: knowledge answer (1 LLM call)
          Phase 1: hop plan (1 LLM call)
          Phase 2: per-hop iterative retrieval (2-4 LLM calls per hop)
          Phase 3: evidence fusion (1 LLM call if multi-hop)
          Phase 3.5: final answer extraction (1 LLM call)
          Phase 4: consistency voting (0-2 LLM calls)
          Phase 5: reverse verification (0-1 LLM call)
          Total: ~10-14 LLM calls (down from ~18-22)
        """
        reasoning_steps = []
        search_traces = []

        # ── Classify question language priority (english_first / chinese_first / bilingual_equal) ──
        # Why: A purely English question about Western history should not trigger
        # IQS (Chinese search) even when hop planning generates Chinese queries.
        # This routing was previously only active in the MiniAgent loop; now it
        # also governs the hop search phase via meta["language_priority"].
        try:
            lang_result = classify_question_geographic_language_priority(
                question, llm_fn=None,
            )
            self._language_priority = lang_result.get("priority", "bilingual_equal")
            reasoning_steps.append(
                f"Language routing: {self._language_priority} "
                f"(signals={lang_result.get('detected_signals', [])})"
            )
        except Exception:
            self._language_priority = "bilingual_equal"

        # ── Create per-question scratchpad (local knowledge base) ──
        # Why: Persist all search results and cleaned pages to disk so later
        # hops can re-use evidence via BM25 local retrieval instead of
        # re-searching the web.
        if self._scratchpad_base_dir:
            # Extract question_id from question text hash (first 6 chars)
            import hashlib
            q_id = hashlib.md5(question.encode()).hexdigest()[:6]
            self._scratchpad = PerQuestionEvidenceScratchpad(
                base_dir=self._scratchpad_base_dir,
                question_id=q_id,
                question_text=question,
            )
            # Propagate scratchpad to the search dispatcher so page enrichment
            # can persist cleaned pages to the scratchpad's pages/ directory.
            if hasattr(self.search_fn, "__self__"):
                server = self.search_fn.__self__
                dispatcher = getattr(server, "_hybrid_search", None)
                if dispatcher:
                    dispatcher.scratchpad = self._scratchpad
            reasoning_steps.append(f"Scratchpad created: q{q_id}")
        else:
            self._scratchpad = None

        # Phase 0: LLM knowledge-only answer (always first for baseline)
        knowledge_answer = self._run_knowledge_only_answer_phase(question, reasoning_steps)

        # Phase 1: Generate structured hop plan
        hop_plan = generate_structured_multi_hop_plan(
            question, llm_fn=self.llm_generic_fn,
        )
        hops = hop_plan.get("hops", [])
        total_stop_condition = hop_plan.get("total_stop_condition", "获取到完整答案")
        reasoning_steps.append(
            f"Hop plan: {len(hops)} hops, targets: "
            f"{[h.get('target','')[:60] for h in hops]}"
        )
        self._record_hop_planning_trace(question, hop_plan)

        # Phase 2: Execute hops with iterative retrieval
        all_ranked = []
        all_texts = []
        hop_evidence_texts = []
        hop_answers = []
        previous_hop_results = []

        # Wing-architecture: structured_hop_results stores precise entities and
        # evidence snippets (instead of vague summary strings) for cross-hop
        # passing.  The ``previous_hop_results`` list remains string-based for
        # backward compatibility with generate_queries_for_hop(), but now
        # contains entity-highlighted summaries.
        structured_hop_results: List[Dict[str, Any]] = []
        # Collect per-hop evaluations so the MiniAgent can see pipeline
        # self-assessment (e.g. valid=False, confidence=0.3 on Hop 4).
        hop_evaluation_summaries: List[Dict[str, Any]] = []
        knowledge_chain_answer = ""  # Set by knowledge-anchored recovery if triggered
        knowledge_chain_evidence = ""
        # Backtrack guard: allow at most 1 backtrack per question to prevent
        # infinite loops.  Tracks rejected entities across hops so the backtrack
        # search can exclude all known-wrong candidates.
        backtrack_used = False
        backtrack_rejected_entities: List[str] = []

        for hop_idx, hop in enumerate(hops):
            hop_num = hop.get("hop_num", hop_idx + 1)
            hop_target = hop.get("target", "搜索相关信息")

            hop_answer, hop_evidence_text, should_stop, hop_meta = self._execute_iterative_retrieval_for_single_hop(
                question=question,
                hop_idx=hop_idx,
                hop=hop,
                hops=hops,
                total_stop_condition=total_stop_condition,
                previous_hop_results=previous_hop_results,
                hop_answers=hop_answers,
                reasoning_steps=reasoning_steps,
                search_traces=search_traces,
                all_ranked=all_ranked,
                all_texts=all_texts,
                knowledge_answer=knowledge_answer,
            )

            hop_evidence_texts.append(hop_evidence_text)
            hop_answers.append(hop_answer)

            # ── LOG: Per-hop result summary for diagnostics ──
            eval_info_log = hop_meta.get("last_evaluation", {})
            hop_entity_log = eval_info_log.get("extracted_entity", hop_answer or "")
            hop_valid_log = eval_info_log.get("valid", None)
            hop_conf_log = eval_info_log.get("confidence", None)
            logger.info(
                "[HOP %d RESULT] entity=%s valid=%s confidence=%s "
                "cycles=%d elapsed_ms=%d stop_reason=%s",
                hop_num,
                repr(hop_entity_log[:80]) if hop_entity_log else "EMPTY",
                hop_valid_log,
                f"{hop_conf_log:.2f}" if hop_conf_log is not None else "N/A",
                hop_meta.get("total_cycles", 0),
                hop_meta.get("elapsed_ms", 0),
                hop_meta.get("stop_reason", "unknown"),
            )

            # ── BACKTRACK CHECK: Log all condition variables before evaluating ──
            bt_eval = hop_meta.get("last_evaluation", {})
            bt_conf = bt_eval.get("confidence", 1.0)
            bt_valid = bt_eval.get("valid", True)
            logger.info(
                "[BACKTRACK CHECK] hop_idx=%d backtrack_used=%s "
                "eval_valid=%s eval_confidence=%.2f "
                "conditions_met=%s (need: hop_idx>=1, conf<=0.3, valid=False, not backtrack_used)",
                hop_idx, backtrack_used, bt_valid, bt_conf,
                (not backtrack_used and hop_idx >= 1 and bt_conf <= 0.3 and not bt_valid),
            )

            # ── BACKTRACK: When evaluation rejects candidate with low confidence,
            # re-search from scratch with exclusion constraints ──
            # Why (from Q99 analysis): Hop 2 discovered Peter Ndlovu (born 1973)
            # fails the 1988-1995 constraint, but could only refine within Hop 2,
            # eventually settling on Quinton Fortune (also wrong).  Backtracking
            # to Hop 1 with exclusion terms finds the correct younger candidate.
            #
            # References:
            #   - Research_Agent: correct_hop_result() — retry tool on validation failure
            #   - Weaver: knowledge gap analysis — use missing info to guide next search
            if (not backtrack_used
                    and hop_idx >= 1
                    and hop_meta.get("last_evaluation", {}).get("confidence", 1.0) <= 0.3
                    and not hop_meta.get("last_evaluation", {}).get("valid", True)):
                eval_info = hop_meta.get("last_evaluation", {})
                missing_info = eval_info.get("missing_info", "")
                rejected_entity = eval_info.get("extracted_entity", "")
                # Collect all entities that have been tried and failed
                if rejected_entity and rejected_entity not in backtrack_rejected_entities:
                    backtrack_rejected_entities.append(rejected_entity)
                # Also add previous hop answers that were invalidated
                for prev_ha in hop_answers[:-1]:
                    if (prev_ha
                            and prev_ha not in backtrack_rejected_entities
                            and not check_if_answer_is_refusal_or_unknown_placeholder(prev_ha)):
                        backtrack_rejected_entities.append(prev_ha)

                if backtrack_rejected_entities:
                    backtrack_used = True
                    reasoning_steps.append(
                        f"BACKTRACK TRIGGERED at Hop {hop_meta.get('last_evaluation', {}).get('extracted_entity', '')} "
                        f"(conf={eval_info.get('confidence', 0):.2f}): "
                        f"rejected={backtrack_rejected_entities}"
                    )
                    bt_answer, bt_evidence, bt_ranked = (
                        self._attempt_hop_chain_backtrack_on_candidate_rejection(
                            question=question,
                            rejected_entities=backtrack_rejected_entities,
                            rejection_reason=missing_info,
                            original_hop_target=hops[0].get("target", ""),
                            hop_domain=hops[0].get("domain", "news"),
                            reasoning_steps=reasoning_steps,
                            search_traces=search_traces,
                            all_ranked=all_ranked,
                            all_texts=all_texts,
                        ))
                    if bt_answer and not check_if_answer_is_refusal_or_unknown_placeholder(bt_answer):
                        # Inject backtrack answer as a high-priority candidate
                        # by overriding the current hop answer and adding to
                        # structured results
                        hop_answer = bt_answer
                        hop_answers[-1] = bt_answer
                        hop_evidence_texts[-1] = bt_evidence or hop_evidence_text
                        reasoning_steps.append(
                            f"BACKTRACK SUCCESS: new candidate '{bt_answer}'"
                        )

            # Wing-architecture: Structured entity passing — store precise entity
            # + short evidence snippet for downstream hops, instead of a vague
            # summary string.  This lets query generation use exact names/values.
            entity = hop_answer.strip() if hop_answer else ""
            evidence_snippet = hop_evidence_text[:200].strip() if hop_evidence_text else ""
            hop_result_struct = {
                "hop_num": hop_num,
                "target": hop_target,
                "entity": entity,
                "evidence_snippet": evidence_snippet,
                "confidence": 0.8 if entity else 0.2,
            }
            structured_hop_results.append(hop_result_struct)

            # Collect evaluation summary for MiniAgent injection
            _he = hop_meta.get("last_evaluation", {})
            hop_evaluation_summaries.append({
                "hop_num": hop_num,
                "entity": entity[:100] if entity else "",
                "valid": _he.get("valid", True),
                "confidence": _he.get("confidence", 0.5),
            })

            # ── PERSIST: Update scratchpad INDEX.md with hop entity ──
            if self._scratchpad and entity:
                self._scratchpad.update_index(
                    hop_num=hop_num,
                    entity=entity,
                    summary=evidence_snippet,
                    target=hop_target,
                )

            # Build an entity-highlighted summary for generate_queries_for_hop()
            if entity:
                summary = f"Hop {hop_num} ({hop_target}): **{entity}** — {evidence_snippet}"
            else:
                summary = f"Hop {hop_num} ({hop_target}): no clear answer — {evidence_snippet}"
            previous_hop_results.append(summary)

            # ── TRACE: Per-hop summary (new process-level observability) ──
            if self.trace_logger and hasattr(self.trace_logger, "record_per_hop_summary_trace"):
                self.trace_logger.record_per_hop_summary_trace(
                    question=question,
                    hop_num=hop_num,
                    hop_target=hop_target,
                    details={
                        "domain": hop.get("domain", ""),
                        "total_cycles": int(hop_meta.get("total_cycles", 0)),
                        "final_entity": entity,
                        "stop_reason": hop_meta.get("stop_reason", ""),
                        "elapsed_ms": int(hop_meta.get("elapsed_ms", 0)),
                        "evidence_chars": len(hop_evidence_text or ""),
                        "should_stop_chain": bool(should_stop),
                        "last_evaluation": hop_meta.get("last_evaluation", {}),
                    },
                )

            # ── Early short-circuit: skip remaining hops when knowledge confirmed ──
            # Why (from run_20260216_000139 analysis): knowledge_answer was correct
            # in 11s, but 5 hops spent 1325s (80% of total time) searching blindly.
            # If Hop 1's extracted answer matches knowledge_answer, we can verify
            # with a single web search and skip remaining hops — saving ~20 minutes.
            #
            # References:
            #   - Research_Agent: stop_condition per hop with explicit termination
            #   - LLMCompiler: join() barrier — only proceed when evidence is ready
            if hop_idx == 0 and knowledge_answer and hop_answer:
                knowledge_confirmed = self._check_knowledge_hop1_agreement(
                    knowledge_answer, hop_answer, question)
                if knowledge_confirmed:
                    reasoning_steps.append(
                        f"EARLY SHORT-CIRCUIT: Hop 1 answer '{hop_answer}' confirms "
                        f"knowledge '{knowledge_answer}' — skipping remaining {len(hops) - 1} hops"
                    )
                    # Run a quick verification search to strengthen evidence
                    ka_answer, ka_evidence, ka_ranked = (
                        self._run_knowledge_anchored_verification_chain(
                            question, knowledge_answer, hops,
                            reasoning_steps, search_traces))
                    if ka_answer and not check_if_answer_is_refusal_or_unknown_placeholder(ka_answer):
                        knowledge_chain_answer = ka_answer
                        knowledge_chain_evidence = ka_evidence
                        all_ranked.extend(ka_ranked)
                    break  # Skip remaining hops

            # ── Knowledge-anchored dual-chain recovery (after Hop 1) ──
            # Why: If the LLM's own knowledge and hop 1's search result point to
            # different entities, the search chain has gone off track.  We launch
            # a parallel verification chain using the knowledge entity to produce
            # an additional candidate answer for the voting phase.
            if hop_idx == 0 and knowledge_answer:
                is_divergent = self._check_knowledge_hop1_divergence(
                    knowledge_answer, hop_answer, question, reasoning_steps)
                if is_divergent:
                    ka_answer, ka_evidence, ka_ranked = (
                        self._run_knowledge_anchored_verification_chain(
                            question, knowledge_answer, hops,
                            reasoning_steps, search_traces))
                    if ka_answer and not check_if_answer_is_refusal_or_unknown_placeholder(ka_answer):
                        knowledge_chain_answer = ka_answer
                        knowledge_chain_evidence = ka_evidence
                        # Inject knowledge chain evidence into all_ranked so it
                        # can contribute to the evidence fusion phase too.
                        all_ranked.extend(ka_ranked)
                        reasoning_steps.append(
                            f"Knowledge-anchored recovery injected: '{ka_answer}'"
                        )

            if should_stop:
                break

        # Phase 3: Evidence fusion
        all_evidence = deduplicate_and_select_top_evidence_items(all_ranked, self.max_evidence)
        if self.llm_fuse_fn and len(hop_evidence_texts) > 1:
            fused_evidence = self.llm_fuse_fn(question, hop_evidence_texts)
            if fused_evidence:
                reasoning_steps.append(f"Evidence fusion: {len(fused_evidence)} chars from {len(hop_evidence_texts)} hops")
            else:
                fused_evidence = format_evidence_items_into_llm_readable_text(all_evidence)
        else:
            fused_evidence = format_evidence_items_into_llm_readable_text(all_evidence)

        # Phase 3.5: Extract final answer from fused evidence
        search_answer = ""
        if self.llm_answer_fn and fused_evidence:
            context = fused_evidence
            if knowledge_answer:
                context = (
                    f"NOTE: A preliminary answer '{knowledge_answer}' was generated from LLM knowledge alone. "
                    f"This may have incorrect formatting or entity names. "
                    f"Always verify and correct using the web search evidence below.\n\n"
                    f"Web search evidence:\n{fused_evidence}"
                )
                if self.trace_logger:
                    self.trace_logger.record_event("knowledge_context_injected",
                        f"injected knowledge_answer='{knowledge_answer[:60]}' into extraction context")
            search_answer = clean_and_validate_raw_llm_answer_text(
                self.llm_answer_fn(question, context[:12000]))

        # Heuristic answer from all evidence
        question_type = classify_question_into_answer_type_category(question)
        candidates_struct = extract_structured_answer_candidates_from_evidence(all_texts)
        heuristic_answer = select_highest_frequency_candidate_as_answer(
            question, candidates_struct, question_type)

        # Get the best hop answer (last non-empty one)
        hop2_answer = ""
        for ha in reversed(hop_answers):
            if ha and not check_if_answer_is_refusal_or_unknown_placeholder(ha):
                hop2_answer = ha
                break

        # Phase 4: Select best answer via consistency voting (P1-a)
        candidate_dict = {}
        if hop2_answer:
            candidate_dict["hop2"] = hop2_answer
        if search_answer:
            candidate_dict["search"] = search_answer
        if knowledge_answer:
            candidate_dict["knowledge"] = knowledge_answer
        if heuristic_answer:
            candidate_dict["heuristic"] = heuristic_answer
        # Knowledge-anchored dual-chain: inject the verified answer from the
        # recovery chain as a high-priority candidate.  This gives the voting
        # system two knowledge-backed votes (knowledge + knowledge_chain) to
        # outweigh the potentially wrong search chain answers.
        if knowledge_chain_answer:
            candidate_dict["knowledge_chain"] = knowledge_chain_answer

        if self.trace_logger:
            self.trace_logger.record_event("voting_candidates_assembled",
                f"candidates: {', '.join(f'{k}={v[:40]}' for k, v in candidate_dict.items())}",
                details={"candidates": {k: v[:100] for k, v in candidate_dict.items()}})

        final_answer, answer_source, voting_trace = select_final_answer_with_consistency_voting(
            question=question,
            candidate_dict=candidate_dict,
            llm_arbitrate_fn=self.llm_arbitrate_fn,
            trace_logger=self.trace_logger,
        )
        reasoning_steps.append(f"Voting decision: {voting_trace.get('decision', 'unknown')}")

        # Phase 4.5: MiniAgent evidence review
        # Why: The voting mechanism is rigid — this loop lets the LLM freely
        # browse collected evidence and make a more informed final decision.
        if (self._scratchpad
                and self.llm_generic_fn
                and self._scratchpad.get_stats().get("bm25_document_count", 0) > 0):
            from src.mini_agent import run_mini_agent_loop

            agent_result = run_mini_agent_loop(
                question=question,
                pipeline_answer=final_answer,
                pipeline_candidates=candidate_dict,
                scratchpad=self._scratchpad,
                llm_fn=self.llm_generic_fn,
                search_fn=self.search_fn,
                trace_logger=self.trace_logger,
                mcp_config=getattr(self, '_mcp_config', None),
                mcp_http_clients=getattr(self, '_mcp_http_clients', None),
                hop_evaluations=hop_evaluation_summaries,
            )
            reasoning_steps.append(
                f"MiniAgent: answer='{agent_result['answer'][:60]}' "
                f"confidence={agent_result['confidence']:.2f} "
                f"tools={agent_result['tool_calls_count']} "
                f"iters={agent_result['iterations']}"
            )
            # Accept MiniAgent answer if confidence is high enough
            if agent_result["confidence"] > 0.6:
                final_answer = agent_result["answer"]
                answer_source = "mini_agent"
                reasoning_steps.append(
                    f"MiniAgent overrode pipeline answer (conf={agent_result['confidence']:.2f})"
                )

        # Phase 5: Reverse verification
        final_answer = self._run_answer_verification_phase(question, final_answer, answer_source,
                                          search_answer, knowledge_answer, reasoning_steps)

        # Phase 6 (Wing-architecture): Confidence-driven abstention.
        # Wing scores 42 by submitting empty answers for 5 low-confidence questions,
        # avoiding wrong-answer penalties.  We replicate this: if confidence is too
        # low, return "" instead of a likely-wrong answer.
        overall_confidence = voting_trace.get("confidence", 0.5)
        final_answer, believe_score = self._apply_confidence_driven_abstention(
            final_answer, candidate_dict, voting_trace, reasoning_steps)

        reasoning_steps.extend([
            f"Evidence count: {len(all_evidence)}",
            f"Knowledge answer: {knowledge_answer}",
            f"Search answer: {search_answer}",
            f"Hop answers: {hop_answers}",
            f"Final answer: {final_answer}",
            f"Believe score: {believe_score:.2f}",
        ])
        # ── Log scratchpad stats ──
        if self._scratchpad:
            sp_stats = self._scratchpad.get_stats()
            reasoning_steps.append(
                f"Scratchpad stats: {sp_stats['bm25_document_count']} docs, "
                f"{sp_stats['evidence_files_written']} evidence files, "
                f"{sp_stats['page_files_written']} page files, "
                f"{sp_stats['local_search_queries']} local queries, "
                f"{sp_stats['local_search_hits']} hits")
            if self.trace_logger and hasattr(self.trace_logger, "record_scratchpad_operation_trace"):
                self.trace_logger.record_scratchpad_operation_trace(
                    operation="session_summary",
                    details=sp_stats,
                )

        return {
            "answer": final_answer,
            "reasoning_steps": reasoning_steps,
            "search_traces": search_traces,
            "believe": believe_score,
        }

    # ── Legacy query generation (kept for compatibility) ─────────────

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

    # ── Phase helpers ───────────────────────────────────────────────────

    def _run_knowledge_only_answer_phase(self, question: str, reasoning_steps: List[str]) -> str:
        """Phase 0: Ask LLM to answer from its own knowledge first.

        Why (enhanced after Q99 analysis): The original implementation silently
        swallowed all exceptions, including timeouts.  When the LLM API times
        out 3 times (180s wasted), knowledge_answer is empty, which disables:
        1. Knowledge-guided search (entity injection into queries)
        2. Early short-circuit (skip remaining hops when knowledge confirmed)
        3. Knowledge-anchored dual-chain recovery

        Enhancement: Two-tier fallback:
        Tier 1: Normal LLM call with full reasoning prompt
        Tier 2: If Tier 1 fails, try a degraded "direct answer only" prompt
                 with shorter max_tokens (faster response, less likely to timeout)
        Tier 3: If both LLM calls fail, extract pseudo-knowledge from question
                 text using rule-based entity extraction

        References:
          - Research_Agent: fallback to direct reasoning on tool failure
          - Weaver: budget-based early stopping with graceful degradation
        """
        import logging
        logger = logging.getLogger(__name__)

        # Tier 1: Normal knowledge call
        if self.llm_knowledge_fn:
            try:
                raw = self.llm_knowledge_fn(question)
                processed = clean_and_validate_raw_llm_answer_text(raw)
                if processed:
                    reasoning_steps.append(f"Knowledge answer: {processed}")
                    return processed
            except Exception as exc:
                logger.warning("Knowledge answer Tier 1 failed: %s", exc)

        # Tier 2: Degraded direct-answer prompt (shorter, faster)
        if self.llm_generic_fn:
            try:
                degraded_system = (
                    "Answer the question directly in 1-10 words. "
                    "No explanation, no reasoning, just the answer."
                )
                degraded_user = f"Question: {question}\nAnswer:"
                raw = self.llm_generic_fn(degraded_system, degraded_user)
                if raw:
                    # Extract just the answer part
                    answer_text = raw.strip()
                    # Remove common prefixes
                    for prefix in ["Answer:", "answer:", "A:"]:
                        if answer_text.startswith(prefix):
                            answer_text = answer_text[len(prefix):].strip()
                    processed = clean_and_validate_raw_llm_answer_text(answer_text)
                    if processed:
                        reasoning_steps.append(f"Knowledge answer (degraded): {processed}")
                        return processed
            except Exception as exc:
                logger.warning("Knowledge answer Tier 2 (degraded) failed: %s", exc)

        # Tier 3: Rule-based pseudo-knowledge extraction from question text
        # Why: Even without LLM, we can extract key entity hints from the
        # question structure to guide subsequent searches.
        pseudo = self._extract_pseudo_knowledge_from_question_text(question)
        if pseudo:
            reasoning_steps.append(f"Knowledge answer (pseudo): {pseudo}")
            return pseudo

        return ""

    @staticmethod
    def _extract_pseudo_knowledge_from_question_text(question: str) -> str:
        """Extract pseudo-knowledge entity hints from question text.

        Why: When LLM is completely unavailable (all timeouts), we can still
        extract useful search guidance from the question structure.  This is
        NOT an answer — it is a set of key terms that can guide search queries.

        Examples:
          "first African player in Premier League" -> "" (too generic)
          "the company founded by Adrian Bowyer" -> "Adrian Bowyer" (specific entity)
          "the author of 'War and Peace'" -> "War and Peace" (specific entity)
        """
        # Extract quoted entities
        quoted = re.findall(r"['\"]([^'\"]{2,60})['\"]", question)
        if quoted:
            return quoted[0]

        # Extract proper nouns (capitalized multi-word sequences)
        proper_nouns = re.findall(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", question)
        # Filter out common non-entity phrases
        noise = {"Premier League", "English Premier", "World Cup", "European Union",
                 "United States", "United Kingdom", "New York", "Los Angeles"}
        proper_nouns = [p for p in proper_nouns if p not in noise]
        if proper_nouns:
            return proper_nouns[0]

        # Extract Chinese proper nouns (《》 brackets)
        zh_titles = re.findall(r"[《「]([^》」]{2,30})[》」]", question)
        if zh_titles:
            return zh_titles[0]

        return ""

    # ── Knowledge-anchored dual-chain recovery ─────────────────────

    def _check_knowledge_hop1_agreement(
        self,
        knowledge_answer: str,
        hop1_answer: str,
        question: str,
    ) -> bool:
        """Return True if knowledge_answer and hop1_answer agree on the same entity.

        Why: When both the LLM's own knowledge (Phase 0) and the first hop's
        search result point to the same entity, we have high confidence that the
        answer is correct.  This enables the early short-circuit optimization
        that skips remaining hops, saving 80%+ of total processing time.

        The check is conservative — we only confirm agreement when the two
        answers clearly refer to the *same* entity, not when they merely share
        a few words.
        """
        if not knowledge_answer or not hop1_answer:
            return False
        # If hop1 failed (Unknown/refusal), no agreement
        if check_if_answer_is_refusal_or_unknown_placeholder(hop1_answer):
            return False
        if check_if_answer_is_refusal_or_unknown_placeholder(knowledge_answer):
            return False

        ka_lower = knowledge_answer.lower().strip()
        h1_lower = hop1_answer.lower().strip()

        # Direct substring match: one contains the other
        if ka_lower in h1_lower or h1_lower in ka_lower:
            return True

        # High word overlap (>= 60%) indicates same entity
        ka_words = set(ka_lower.split())
        h1_words = set(h1_lower.split())
        if ka_words and h1_words:
            overlap = len(ka_words & h1_words) / min(len(ka_words), len(h1_words))
            if overlap >= 0.6:
                return True

        # For short answers (< 5 words), check edit distance heuristic
        if len(ka_lower) < 50 and len(h1_lower) < 50:
            # Simple character-level containment check
            common_chars = sum(1 for c in ka_lower if c in h1_lower)
            similarity = common_chars / max(len(ka_lower), 1)
            if similarity >= 0.7:
                return True

        return False

    def _check_knowledge_hop1_divergence(
        self,
        knowledge_answer: str,
        hop1_answer: str,
        question: str,
        reasoning_steps: List[str],
    ) -> bool:
        """Return True if knowledge_answer and hop1_answer diverge significantly.

        Why: When the LLM's own knowledge (Phase 0) and the first hop's search
        result point to different entities, the search chain has likely gone off
        track.  Detecting this early lets us launch a knowledge-anchored
        verification chain to recover the correct answer.

        The check is conservative — we only flag divergence when the two answers
        clearly refer to *different* entities, not when they merely differ in
        formatting or specificity.
        """
        if not knowledge_answer or not hop1_answer:
            return False
        # If hop1 failed (Unknown/refusal), knowledge is likely better
        if check_if_answer_is_refusal_or_unknown_placeholder(hop1_answer):
            reasoning_steps.append(
                f"Knowledge-Hop1 divergence: hop1 is Unknown/refusal, "
                f"knowledge='{knowledge_answer}' — divergent"
            )
            return True

        ka_lower = knowledge_answer.lower().strip()
        h1_lower = hop1_answer.lower().strip()

        # Quick substring check: if one contains the other, they likely agree
        if ka_lower in h1_lower or h1_lower in ka_lower:
            return False

        # Check shared key words (more than 50% overlap → likely same entity)
        ka_words = set(ka_lower.split())
        h1_words = set(h1_lower.split())
        if ka_words and h1_words:
            overlap = len(ka_words & h1_words) / min(len(ka_words), len(h1_words))
            if overlap >= 0.5:
                return False

        # Use LLM for a definitive judgment
        if self.llm_generic_fn:
            system_prompt = (
                "You compare two answers to the same question. "
                "Reply ONLY 'same' or 'different'. "
                "'same' = they refer to the same entity/concept (even if names differ slightly). "
                "'different' = they refer to clearly different entities."
            )
            user_prompt = (
                f"Question: {question[:200]}\n"
                f"Answer A (LLM knowledge): {knowledge_answer}\n"
                f"Answer B (web search hop 1): {hop1_answer}\n\n"
                f"Are A and B the same entity or different?"
            )
            try:
                result = self.llm_generic_fn(system_prompt, user_prompt)
                if result and "different" in result.lower():
                    reasoning_steps.append(
                        f"Knowledge-Hop1 DIVERGENCE (LLM confirmed): "
                        f"knowledge='{knowledge_answer}' vs hop1='{hop1_answer}'"
                    )
                    return True
                if result and "same" in result.lower():
                    return False
            except Exception as exc:
                if self.trace_logger:
                    self.trace_logger.record_event("exception_swallowed", f"divergence_check LLM call failed: {exc}")

        # Default: assume no divergence (conservative)
        return False

    def _run_knowledge_anchored_verification_chain(
        self,
        question: str,
        knowledge_answer: str,
        hops: List[Dict[str, Any]],
        reasoning_steps: List[str],
        search_traces: List[Dict],
    ) -> Tuple[str, str, List[Dict]]:
        """Run a verification chain using knowledge_answer as the anchor entity.

        Why: When hop 1 diverges from the LLM's knowledge, the search chain is
        likely chasing the wrong entity.  This method searches for the *knowledge*
        entity instead, gathers evidence, and extracts a verified answer.  The
        result becomes an additional candidate in the voting phase.

        Returns:
            (verified_answer, evidence_text, ranked_results)
        """
        reasoning_steps.append(
            f"Knowledge-anchored chain: starting verification for '{knowledge_answer}'"
        )

        # Step 1: Generate verification queries from the knowledge entity
        verify_queries = [
            f"{knowledge_answer} {question[:80]}",
            knowledge_answer,
        ]
        # Add bilingual queries if LLM is available
        if self.llm_generic_fn:
            try:
                gen_prompt = (
                    f"Generate 2 Chinese and 2 English search queries to verify "
                    f"whether '{knowledge_answer}' is the correct answer to:\n"
                    f"{question[:200]}\n\n"
                    f"Output ONLY search queries, one per line, no numbering."
                )
                raw = self.llm_generic_fn(
                    "Generate verification search queries.", gen_prompt
                )
                if raw:
                    for line in raw.strip().split("\n"):
                        line = line.strip().strip('"').strip("'")
                        if line and len(line) >= 4:
                            verify_queries.append(line)
            except Exception as exc:
                if self.trace_logger:
                    self.trace_logger.record_event("exception_swallowed", f"verify_query_generation failed: {exc}")
        verify_queries = list(dict.fromkeys(verify_queries))[:6]
        reasoning_steps.append(
            f"Knowledge-anchored chain: queries={verify_queries}"
        )

        # Step 2: Execute searches
        from src.agents.question_domain_classifier import classify_question_domain
        domain = classify_question_domain(
            question, hop_target=knowledge_answer, llm_fn=self.llm_generic_fn
        )
        ka_all_ranked: List[Dict] = []
        ka_all_texts: List[str] = []
        for query in verify_queries:
            results, rewritten = self._perform_search_with_result_caching(
                query, domain=domain
            )
            ranked = rank_search_results_by_relevance_and_truncate(
                rewritten, results, self.max_results_per_query
            )
            search_traces.append({
                "query": query, "rewritten": rewritten,
                "count": len(ranked), "domain": domain,
                "source": "knowledge_anchored_chain",
            })
            ka_all_ranked.extend(ranked)
            for item in ranked:
                ka_all_texts.append(
                    f"{item.get('title', '')} {item.get('content', '')}"
                )

        if not ka_all_ranked:
            reasoning_steps.append(
                "Knowledge-anchored chain: no search results found"
            )
            return "", "", []

        # Step 3: Build evidence and extract answer
        ka_evidence = deduplicate_and_select_top_evidence_items(
            ka_all_ranked, self.max_evidence
        )
        ka_evidence_text = format_evidence_items_into_llm_readable_text(ka_evidence)

        verified_answer = ""
        if self.llm_answer_fn and ka_evidence_text:
            context = (
                f"IMPORTANT: A preliminary answer '{knowledge_answer}' was "
                f"generated from LLM knowledge. Verify it against the web "
                f"search evidence below. If the evidence supports it, return "
                f"the exact correct name/answer. If not, extract the correct "
                f"answer from the evidence.\n\n"
                f"Web search evidence:\n{ka_evidence_text}"
            )
            verified_answer = clean_and_validate_raw_llm_answer_text(
                self.llm_answer_fn(question, context[:10000])
            )

        reasoning_steps.append(
            f"Knowledge-anchored chain: verified_answer='{verified_answer}', "
            f"evidence_chars={len(ka_evidence_text)}, results={len(ka_all_ranked)}"
        )
        return verified_answer, ka_evidence_text, ka_all_ranked

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
        """Phase 5: Reverse verification — search for the answer to confirm it.

        P0-f enhancement (from Enhancing-Multi-Hop-QA self_verification recovery):
        When verification returns REFUTES, its reasoning often names the correct
        entity.  We now extract that entity via LLM and use it as the corrected
        answer — the highest-ROI recovery path.

        Phase 7 enhancements:
        - 7.1: Adversarial query — also search excluding the current answer
        - 7.2: If confidence < 0.8, search for the second-best candidate
        - 7.3: Verification prompt now asks for Better_entity
        """
        if final_answer == "Unknown" or answer_source not in ("hop2", "search", "mini_agent", "agentic_loop", "arbitrated", "consensus"):
            if self.trace_logger:
                self.trace_logger.record_event("verification_skipped",
                    f"answer='{final_answer[:40]}' source={answer_source} — not eligible for verification")
            return final_answer
        if not self.search_fn or not self.llm_verify_fn:
            if self.trace_logger:
                self.trace_logger.record_event("verification_skipped",
                    f"search_fn={bool(self.search_fn)} verify_fn={bool(self.llm_verify_fn)} — missing functions")
            return final_answer
        try:
            # Phase 7.1: Two verification queries — confirmatory + adversarial
            verify_query = f"{final_answer} {question[:80]}"
            verify_result = self.search_fn(verify_query, {"original_query": verify_query})
            verify_items = verify_result.get("results", []) if isinstance(verify_result, dict) else []

            # Adversarial query: search for alternatives excluding current answer
            adversarial_items = []
            try:
                adversarial_query = f"{question[:100]} -\"{final_answer}\""
                adversarial_result = self.search_fn(adversarial_query, {"original_query": adversarial_query})
                adversarial_items = adversarial_result.get("results", []) if isinstance(adversarial_result, dict) else []
            except Exception as exc:
                if self.trace_logger:
                    self.trace_logger.record_event("adversarial_query_failed", f"error={exc}")

            # Combine evidence from both queries
            all_verify_items = verify_items[:3] + adversarial_items[:2]

            if all_verify_items:
                verify_text = format_evidence_items_into_llm_readable_text(all_verify_items[:5])
                if verify_text:
                    is_valid, verdict, reasoning, confidence = verify_answer_against_evidence_using_llm(
                        question, final_answer, verify_text, self.llm_verify_fn)
                    reasoning_steps.append(
                        f"Reverse verify: verdict={verdict}, conf={confidence:.2f}, "
                        f"reasoning={reasoning[:150]}"
                    )
                    if not is_valid:
                        # P0-f: When REFUTES, try to extract the correct entity
                        # from the verification reasoning before falling back.
                        if verdict == "REFUTES" and reasoning and self.llm_refute_extract_fn:
                            corrected = self.llm_refute_extract_fn(question, reasoning)
                            if corrected and not check_if_answer_is_refusal_or_unknown_placeholder(corrected):
                                reasoning_steps.append(
                                    f"REFUTES recovery: extracted '{corrected}' from reasoning"
                                )
                                return corrected

                        # Only fallback to alternative answers when REFUTES, not INSUFFICIENT.
                        # INSUFFICIENT means evidence is ambiguous — keep the current answer.
                        if verdict == "REFUTES":
                            reasoning_steps.append(f"Verification REFUTED '{final_answer}', trying fallback")
                            if answer_source == "hop2" and search_answer and not check_if_answer_is_refusal_or_unknown_placeholder(search_answer):
                                return search_answer
                            if knowledge_answer and not check_if_answer_is_refusal_or_unknown_placeholder(knowledge_answer):
                                return knowledge_answer
                        else:
                            # Phase 7.2: INSUFFICIENT with low confidence — search for second candidate
                            if confidence < 0.8 and search_answer and search_answer != final_answer:
                                reasoning_steps.append(
                                    f"Verification INSUFFICIENT (conf={confidence:.2f}), "
                                    f"checking alternative: '{search_answer}'"
                                )
                                try:
                                    alt_query = f"{search_answer} {question[:80]}"
                                    alt_result = self.search_fn(alt_query, {"original_query": alt_query})
                                    alt_items = alt_result.get("results", []) if isinstance(alt_result, dict) else []
                                    if alt_items:
                                        alt_text = format_evidence_items_into_llm_readable_text(alt_items[:3])
                                        if alt_text:
                                            alt_valid, alt_verdict, alt_reasoning, alt_conf = (
                                                verify_answer_against_evidence_using_llm(
                                                    question, search_answer, alt_text, self.llm_verify_fn))
                                            reasoning_steps.append(
                                                f"Alternative verify: '{search_answer}' verdict={alt_verdict}, "
                                                f"conf={alt_conf:.2f}"
                                            )
                                            if alt_valid and alt_conf > confidence:
                                                reasoning_steps.append(
                                                    f"Alternative '{search_answer}' has higher confidence, switching"
                                                )
                                                return search_answer
                                except Exception as exc:
                                    if self.trace_logger:
                                        self.trace_logger.record_event("exception_swallowed", f"alternative_verification failed: {exc}")
                            reasoning_steps.append(
                                f"Verification INSUFFICIENT for '{final_answer}', keeping answer"
                            )
        except Exception as exc:
            if self.trace_logger:
                self.trace_logger.record_event("exception_swallowed", f"verification_phase failed: {exc}")
        return final_answer

    # ── Phase 6: Confidence-driven abstention (Wing architecture) ──────

    @staticmethod
    def _apply_confidence_driven_abstention(
        final_answer: str,
        candidate_dict: Dict[str, str],
        voting_trace: Dict[str, Any],
        reasoning_steps: List[str],
    ) -> Tuple[str, float]:
        """Return (possibly-emptied answer, believe_score).

        Wing-architecture: Wing achieves 42 points partly by abstaining on 5
        questions where confidence is low, avoiding wrong-answer penalties.
        We replicate this with three abstention rules:

        1. All candidates are Unknown/refusal → return "" (abstain)
        2. Voting confidence < 0.4 and no consensus → return "" (abstain)
        3. Final answer is itself Unknown → return "" (abstain)

        The ``believe`` score (0.0-1.0) is also returned for logging, mirroring
        Wing's ``believe`` field.
        """
        # Compute believe score from voting confidence
        believe = float(voting_trace.get("confidence", 0.5))
        decision = voting_trace.get("decision", "")

        # Rule 1: If the final answer is Unknown/refusal, abstain
        if check_if_answer_is_refusal_or_unknown_placeholder(final_answer):
            reasoning_steps.append(
                f"Abstention rule 1: final_answer is Unknown/refusal → abstain (believe={believe:.2f})"
            )
            return "", believe

        # Rule 2: All candidates are Unknown/refusal → abstain
        non_unknown_candidates = [
            v for v in candidate_dict.values()
            if v and not check_if_answer_is_refusal_or_unknown_placeholder(v)
        ]
        if not non_unknown_candidates:
            reasoning_steps.append(
                f"Abstention rule 2: all {len(candidate_dict)} candidates are Unknown → abstain"
            )
            return "", 0.0

        # Rule 3: Low confidence + no consensus → abstain
        if believe < 0.4 and decision in ("no_consensus", "single_candidate", ""):
            reasoning_steps.append(
                f"Abstention rule 3: believe={believe:.2f} < 0.4 and decision='{decision}' → abstain"
            )
            return "", believe

        return final_answer, believe

    # ── Trace logging ───────────────────────────────────────────────────

    def _record_hop_planning_trace(self, question: str, hop_plan: Dict[str, Any]) -> None:
        """Record the structured hop plan for post-mortem analysis."""
        if not self.trace_logger:
            return
        try:
            if hasattr(self.trace_logger, 'record_hop_planning_trace'):
                self.trace_logger.record_hop_planning_trace(
                    question=question,
                    hop_plan=hop_plan,
                )
        except Exception:
            pass
