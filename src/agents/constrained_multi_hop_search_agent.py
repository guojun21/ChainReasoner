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
            return self._cache[cache_key], query
        meta = {"original_query": query, "rewritten_query": query}
        if domain:
            meta["domain"] = domain
        try:
            result = self.search_fn(query, meta)
        except TypeError:
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
    ) -> Tuple[str, str, bool]:
        """Execute one hop with iterative retrieval (DISPATCH → EVALUATE → REFINE → LOOP).

        Why (from everything-claude-code skills/iterative-retrieval/SKILL.md):
        Instead of 'search once, validate, maybe retry once with separate logic',
        use a unified iterative loop that evaluates after each search round and
        refines queries based on what is specifically missing.

        Returns:
            (hop_answer, hop_evidence_text, should_stop_chain)
        """
        hop_num = hop.get("hop_num", hop_idx + 1)
        hop_target = hop.get("target", "搜索相关信息")
        hop_stop_condition = hop.get("stop_condition", "获取到相关信息")
        hop_count = len(hops)
        is_last_hop = (hop_idx == len(hops) - 1)

        # Wing-architecture: Classify hop domain for search engine routing.
        # Use the domain from hop plan if available, otherwise classify via LLM.
        hop_domain = hop.get("domain", "")
        if not hop_domain:
            hop_domain = classify_question_domain(
                question, hop_target=hop_target, llm_fn=self.llm_generic_fn)
        reasoning_steps.append(
            f"Hop {hop_num} domain: {hop_domain}"
        )

        # Generate initial queries for this hop (P0-f: bilingual ZH+EN queries)
        if hop_idx == 0:
            parsed = extract_structured_clues_from_question_text(question)
            base_queries = generate_search_queries_from_question(
                question, parsed, self.max_queries, self.llm_decompose_fn)
            hop_queries = generate_queries_for_hop(
                question, hop_target, previous_hop_results, self.llm_generic_fn)
            # Allow more queries since bilingual generation produces both ZH+EN
            max_q = max(self.max_queries, 6)
            queries = list(dict.fromkeys(base_queries + hop_queries))[:max_q]
        else:
            hop_queries = generate_queries_for_hop(
                question, hop_target, previous_hop_results, self.llm_generic_fn)
            if hop_answers and hop_answers[-1]:
                followup = self._generate_second_hop_follow_up_queries(
                    question, hop_answers[-1])
                hop_queries = list(dict.fromkeys(hop_queries + followup))
            max_q = max(self.max_queries, 6)
            queries = hop_queries[:max_q]

        # Iterative retrieval loop: DISPATCH → EVALUATE → REFINE → LOOP
        hop_answer = ""
        hop_evidence_text = ""
        hop_all_ranked = []

        for cycle in range(MAX_RETRIEVAL_CYCLES_PER_HOP):
            cycle_label = f"Hop {hop_num} cycle {cycle + 1}"
            reasoning_steps.append(f"{cycle_label} queries: {queries}")

            # DISPATCH: Execute search (Wing-architecture: domain-aware routing)
            texts, ranked, traces = self._execute_single_round_of_parallel_searches(queries, domain=hop_domain)
            search_traces.extend(traces)
            hop_all_ranked.extend(ranked)
            all_ranked.extend(ranked)
            all_texts.extend(texts)

            # Build evidence for this cycle (accumulated across cycles within this hop)
            hop_evidence = deduplicate_and_select_top_evidence_items(hop_all_ranked, self.max_evidence)
            hop_evidence_text = format_evidence_items_into_llm_readable_text(hop_evidence)
            reasoning_steps.append(
                f"{cycle_label} evidence: {len(ranked)} new results, "
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
                isolated_context = self._build_isolated_hop_context(
                    hop_target, hop_evidence_text, previous_hop_results)
                reasoning_steps.append(
                    f"{cycle_label} extract: q={'original' if is_last_hop else 'hop_target'} "
                    f"→ '{extraction_question[:80]}'"
                )
                hop_answer = clean_and_validate_raw_llm_answer_text(
                    self.llm_answer_fn(extraction_question, isolated_context[:8000]))
                reasoning_steps.append(
                    f"{cycle_label} extracted_answer: '{hop_answer[:100]}'"
                )

            # EVALUATE: Unified evaluation (validate + reflect + completion check)
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
            reasoning_steps.append(
                f"{cycle_label} eval: valid={evaluation['valid']}, "
                f"conf={evaluation['confidence']:.2f}, "
                f"action={evaluation['next_action']}, "
                f"entity={evaluation.get('extracted_entity', '')[:60]}"
            )

            # Use extracted entity from evaluation if we didn't get a hop_answer
            if evaluation.get("extracted_entity") and not hop_answer:
                hop_answer = evaluation["extracted_entity"]

            # Check if chain is already complete (early termination)
            # P0-f: Raised threshold from 0.75 to 0.85 to reduce premature
            # termination — the old threshold let through half-confident guesses.
            if evaluation["is_chain_complete"] and evaluation["confidence"] >= 0.85:
                chain_answer = evaluation.get("chain_answer", "")
                if chain_answer and len(chain_answer) < 200:
                    hop_answer = chain_answer
                reasoning_steps.append(
                    f"Chain complete at hop {hop_num} cycle {cycle + 1} (conf={evaluation['confidence']:.2f})")
                return hop_answer, hop_evidence_text, True  # should_stop_chain=True

            # Decision: stop iterating, refine, or continue to next hop
            if evaluation["next_action"] == "sufficient":
                reasoning_steps.append(f"{cycle_label}: sufficient evidence")
                break
            elif evaluation["next_action"] == "refine" and cycle < MAX_RETRIEVAL_CYCLES_PER_HOP - 1:
                # REFINE: Generate better queries based on what is missing
                queries = generate_refined_hop_queries_from_evaluation(
                    question=question,
                    hop_num=hop_num,
                    hop_target=hop_target,
                    current_result=hop_answer or hop_evidence_text[:400],
                    missing_info=evaluation.get("missing_info", ""),
                    reasoning=evaluation.get("reasoning", ""),
                    llm_fn=self.llm_generic_fn,
                )
                reasoning_steps.append(f"{cycle_label} refine → {queries}")
            else:
                # Valid or last cycle — move on
                break

        return hop_answer, hop_evidence_text, False  # should_stop_chain=False

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
        knowledge_chain_answer = ""  # Set by knowledge-anchored recovery if triggered
        knowledge_chain_evidence = ""

        for hop_idx, hop in enumerate(hops):
            hop_num = hop.get("hop_num", hop_idx + 1)
            hop_target = hop.get("target", "搜索相关信息")

            hop_answer, hop_evidence_text, should_stop = self._execute_iterative_retrieval_for_single_hop(
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
            )

            hop_evidence_texts.append(hop_evidence_text)
            hop_answers.append(hop_answer)

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

            # Build an entity-highlighted summary for generate_queries_for_hop()
            if entity:
                summary = f"Hop {hop_num} ({hop_target}): **{entity}** — {evidence_snippet}"
            else:
                summary = f"Hop {hop_num} ({hop_target}): no clear answer — {evidence_snippet}"
            previous_hop_results.append(summary)

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

        final_answer, answer_source, voting_trace = select_final_answer_with_consistency_voting(
            question=question,
            candidate_dict=candidate_dict,
            llm_arbitrate_fn=self.llm_arbitrate_fn,
            trace_logger=self.trace_logger,
        )
        reasoning_steps.append(f"Voting decision: {voting_trace.get('decision', 'unknown')}")

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

    # ── Knowledge-anchored dual-chain recovery ─────────────────────

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
            except Exception:
                pass

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
            except Exception:
                pass
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
        """
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

                        reasoning_steps.append(f"Verification failed for '{final_answer}', trying fallback")
                        if answer_source == "hop2" and search_answer and not check_if_answer_is_refusal_or_unknown_placeholder(search_answer):
                            return search_answer
                        if knowledge_answer and not check_if_answer_is_refusal_or_unknown_placeholder(knowledge_answer):
                            return knowledge_answer
        except Exception:
            pass
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
