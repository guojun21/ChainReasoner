"""Multi-candidate answer consistency voter — replaces priority-chain selection.

Why: The old Phase 3 used a fixed priority chain (hop2 > search > knowledge > heuristic)
which ignores agreement among candidates.  When multiple reasoning paths converge on the
same answer, that answer is far more likely to be correct.  This module implements:

  1. Jaccard-similarity-based consensus detection (zero LLM cost)
  2. LLM arbitration as a fallback when no consensus exists (+1 cheap LLM call)

References:
  - MetaGPT ScEnsemble: LLM picks the most consistent solution from candidates
  - MetaGPT MdEnsemble: Majority vote via Counter.most_common()
  - Research_Agent _check_consistency: Jaccard similarity for quick agreement check
"""

import logging
import re as _re
import string
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Similarity threshold: above this, two answers are considered "agreeing"
CONSENSUS_SIMILARITY_THRESHOLD = 0.70

# Priority order for tie-breaking when multiple answers agree equally
ANSWER_SOURCE_PRIORITY_ORDER = ["hop2", "search", "knowledge", "heuristic"]

# Articles and stopwords to strip for comparison
_COMPARISON_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "at", "on", "for", "to", "and", "is", "was",
    "are", "were", "be", "been", "being", "that", "this", "with", "from", "by",
})


# ---------------------------------------------------------------------------
# Text normalisation for comparison
# ---------------------------------------------------------------------------

def normalize_answer_text_for_comparison(text: str) -> str:
    """Normalise answer text for fair similarity comparison.

    Why: Small formatting differences ("RepRapPro Ltd." vs "RepRapPro Ltd")
    should not prevent consensus detection.  We lowercase, strip punctuation,
    remove articles/stopwords, and collapse whitespace.
    """
    if not text:
        return ""
    text = text.lower().strip()
    # Remove all punctuation except hyphens (important for entity names)
    text = _re.sub(r"[^\w\s\-]", " ", text)
    # Remove stopwords
    words = [w for w in text.split() if w not in _COMPARISON_STOPWORDS]
    # Collapse whitespace
    return " ".join(words).strip()


# ---------------------------------------------------------------------------
# Jaccard similarity
# ---------------------------------------------------------------------------

def calculate_answer_pair_jaccard_similarity(answer_a: str, answer_b: str) -> float:
    """Compute Jaccard similarity between two normalised answer strings.

    Why: Jaccard on word sets is a fast, language-agnostic proxy for semantic
    overlap.  Borrowed from Research_Agent's _string_similarity.
    """
    norm_a = normalize_answer_text_for_comparison(answer_a)
    norm_b = normalize_answer_text_for_comparison(answer_b)
    if not norm_a or not norm_b:
        return 0.0
    # Exact match after normalisation
    if norm_a == norm_b:
        return 1.0
    set_a = set(norm_a.split())
    set_b = set(norm_b.split())
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Consensus detection
# ---------------------------------------------------------------------------

def detect_consensus_among_answer_candidates(
    candidates: Dict[str, str],
) -> Tuple[Optional[str], float, str]:
    """Detect whether any pair of candidates agrees (Jaccard >= threshold).

    Args:
        candidates: {source_name: answer_text} — only non-empty, non-refusal entries.

    Returns:
        (consensus_answer, confidence, decision_reason)
        consensus_answer is None if no agreement found.
    """
    sources = list(candidates.keys())
    if len(sources) < 2:
        if sources:
            sole_source = sources[0]
            return candidates[sole_source], 0.5, f"single_candidate:{sole_source}"
        return None, 0.0, "no_candidates"

    # Build pairwise similarity matrix
    pairs: List[Tuple[str, str, float]] = []
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            sim = calculate_answer_pair_jaccard_similarity(
                candidates[sources[i]], candidates[sources[j]])
            pairs.append((sources[i], sources[j], sim))

    # Find the best agreeing pair
    pairs.sort(key=lambda p: p[2], reverse=True)
    best_pair = pairs[0]

    if best_pair[2] >= CONSENSUS_SIMILARITY_THRESHOLD:
        # Pick the answer from the higher-priority source in the agreeing pair
        src_a, src_b, sim = best_pair
        priority_a = ANSWER_SOURCE_PRIORITY_ORDER.index(src_a) if src_a in ANSWER_SOURCE_PRIORITY_ORDER else 99
        priority_b = ANSWER_SOURCE_PRIORITY_ORDER.index(src_b) if src_b in ANSWER_SOURCE_PRIORITY_ORDER else 99
        winner_source = src_a if priority_a <= priority_b else src_b
        return (
            candidates[winner_source],
            min(1.0, sim + 0.1),  # slight confidence boost for agreement
            f"consensus:{src_a}+{src_b}(sim={sim:.2f})->prefer:{winner_source}",
        )

    # Check for majority (3+ candidates with same normalised text)
    norm_to_sources: Dict[str, List[str]] = {}
    for src, answer in candidates.items():
        norm = normalize_answer_text_for_comparison(answer)
        norm_to_sources.setdefault(norm, []).append(src)
    for norm, srcs in sorted(norm_to_sources.items(), key=lambda x: -len(x[1])):
        if len(srcs) >= 2:
            # Pick highest-priority source
            best_src = min(srcs, key=lambda s: ANSWER_SOURCE_PRIORITY_ORDER.index(s) if s in ANSWER_SOURCE_PRIORITY_ORDER else 99)
            return (
                candidates[best_src],
                0.8,
                f"majority:{'+'.join(srcs)}->prefer:{best_src}",
            )

    return None, 0.0, f"no_consensus(best_sim={best_pair[2]:.2f}:{best_pair[0]}vs{best_pair[1]})"


# ---------------------------------------------------------------------------
# LLM arbitration (ScEnsemble-inspired)
# ---------------------------------------------------------------------------

def select_best_answer_via_llm_arbitration(
    question: str,
    candidates: Dict[str, str],
    llm_arbitrate_fn: Callable[[str, str], str],
) -> Tuple[str, str]:
    """Use LLM to pick the best answer from multiple candidates.

    Why: When no statistical consensus exists among candidates, an LLM can
    evaluate specificity, completeness, and plausibility — inspired by
    MetaGPT ScEnsemble's approach.

    Returns:
        (selected_answer, decision_reason)
    """
    sources = list(candidates.keys())
    if not sources:
        return "Unknown", "llm_arbitration:no_candidates"

    # Build labelled candidate list (A, B, C, ...)
    labels = []
    label_to_source = {}
    candidate_text_parts = []
    for idx, source in enumerate(sources):
        label = chr(65 + idx)  # A, B, C, ...
        labels.append(label)
        label_to_source[label] = source
        candidate_text_parts.append(f"{label} (from {source}): {candidates[source]}")

    candidates_block = "\n".join(candidate_text_parts)
    valid_labels = ", ".join(labels)

    system_prompt = (
        "You are an answer quality evaluator. Given a question and multiple candidate answers "
        "from different reasoning paths, select the BEST answer.\n\n"
        "Evaluate based on:\n"
        "1. Specificity and completeness (more precise answers are better)\n"
        "2. Consistency with the question's requirements\n"
        "3. Whether it directly answers what is asked\n\n"
        f"Output ONLY a single letter ({valid_labels}). No explanation."
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Candidate answers:\n{candidates_block}\n\n"
        f"Best answer (output ONLY one letter from {valid_labels}):"
    )

    try:
        raw_response = llm_arbitrate_fn(system_prompt, user_prompt)
        if raw_response:
            # Extract the first valid letter from the response
            for char in raw_response.strip().upper():
                if char in label_to_source:
                    winner_source = label_to_source[char]
                    return (
                        candidates[winner_source],
                        f"llm_arbitration:selected={char}({winner_source})",
                    )
    except Exception as exc:
        logger.warning("LLM arbitration failed: %s", exc)

    # Fallback: return the first candidate by priority
    best_source = min(sources, key=lambda s: ANSWER_SOURCE_PRIORITY_ORDER.index(s) if s in ANSWER_SOURCE_PRIORITY_ORDER else 99)
    return candidates[best_source], f"llm_arbitration_fallback:priority={best_source}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def select_final_answer_with_consistency_voting(
    question: str,
    candidate_dict: Dict[str, str],
    llm_arbitrate_fn: Optional[Callable[[str, str], str]] = None,
    trace_logger: Optional[Any] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Select the best answer using consistency voting + optional LLM arbitration.

    This replaces the old priority-chain selection (hop2 > search > knowledge > heuristic)
    with a two-stage process:
      Stage 1: Detect consensus via Jaccard similarity (zero LLM cost)
      Stage 2: If no consensus, use LLM arbitration (cheap: max_tokens=64)

    Args:
        question: The original question text.
        candidate_dict: {source_name: answer_text} for all non-empty/non-refusal candidates.
        llm_arbitrate_fn: Optional callable(system_prompt, user_prompt) -> response.
        trace_logger: Optional trace logger for recording voting decisions.

    Returns:
        (final_answer, answer_source, voting_trace_dict)
    """
    from src.agents.llm_answer_cleaning_and_candidate_extraction import (
        check_if_answer_is_refusal_or_unknown_placeholder,
    )

    # Filter out empty, None, and refusal answers
    valid_candidates: Dict[str, str] = {}
    for source, answer in candidate_dict.items():
        if answer and answer.strip() and not check_if_answer_is_refusal_or_unknown_placeholder(answer):
            valid_candidates[source] = answer.strip()

    voting_trace: Dict[str, Any] = {
        "all_candidates": {k: v for k, v in candidate_dict.items()},
        "valid_candidates": {k: v for k, v in valid_candidates.items()},
        "valid_count": len(valid_candidates),
    }

    # No valid candidates
    if not valid_candidates:
        voting_trace["decision"] = "no_valid_candidates"
        _record_voting_trace(question, "Unknown", "none", voting_trace, trace_logger)
        return "Unknown", "none", voting_trace

    # Single candidate — return directly
    if len(valid_candidates) == 1:
        source = list(valid_candidates.keys())[0]
        answer = valid_candidates[source]
        voting_trace["decision"] = f"single_candidate:{source}"
        _record_voting_trace(question, answer, source, voting_trace, trace_logger)
        return answer, source, voting_trace

    # Stage 1: Consensus detection via similarity
    consensus_answer, confidence, reason = detect_consensus_among_answer_candidates(valid_candidates)

    # Compute all pairwise similarities for the trace
    sources = list(valid_candidates.keys())
    similarity_matrix: Dict[str, float] = {}
    for i in range(len(sources)):
        for j in range(i + 1, len(sources)):
            sim = calculate_answer_pair_jaccard_similarity(
                valid_candidates[sources[i]], valid_candidates[sources[j]])
            similarity_matrix[f"{sources[i]}vs{sources[j]}"] = round(sim, 3)
    voting_trace["similarity_matrix"] = similarity_matrix

    if consensus_answer is not None:
        # Consensus found — extract the source from the reason string
        answer_source = "consensus"
        for src in ANSWER_SOURCE_PRIORITY_ORDER:
            if src in reason and valid_candidates.get(src) == consensus_answer:
                answer_source = src
                break
        voting_trace["decision"] = reason
        voting_trace["consensus_found"] = True
        voting_trace["confidence"] = confidence
        voting_trace["llm_arbitration_used"] = False
        _record_voting_trace(question, consensus_answer, answer_source, voting_trace, trace_logger)
        return consensus_answer, answer_source, voting_trace

    # Stage 2: LLM arbitration
    voting_trace["consensus_found"] = False

    if llm_arbitrate_fn:
        arb_answer, arb_reason = select_best_answer_via_llm_arbitration(
            question, valid_candidates, llm_arbitrate_fn)
        # Extract source from reason
        answer_source = "arbitrated"
        for src in ANSWER_SOURCE_PRIORITY_ORDER:
            if src in arb_reason and valid_candidates.get(src) == arb_answer:
                answer_source = src
                break
        voting_trace["decision"] = arb_reason
        voting_trace["llm_arbitration_used"] = True
        _record_voting_trace(question, arb_answer, answer_source, voting_trace, trace_logger)
        return arb_answer, answer_source, voting_trace

    # Fallback: old priority chain
    for source in ANSWER_SOURCE_PRIORITY_ORDER:
        if source in valid_candidates:
            voting_trace["decision"] = f"priority_fallback:{source}"
            voting_trace["llm_arbitration_used"] = False
            _record_voting_trace(question, valid_candidates[source], source, voting_trace, trace_logger)
            return valid_candidates[source], source, voting_trace

    # Should not reach here, but just in case
    first_source = list(valid_candidates.keys())[0]
    voting_trace["decision"] = f"last_resort:{first_source}"
    _record_voting_trace(question, valid_candidates[first_source], first_source, voting_trace, trace_logger)
    return valid_candidates[first_source], first_source, voting_trace


def _record_voting_trace(
    question: str,
    final_answer: str,
    answer_source: str,
    voting_trace: Dict[str, Any],
    trace_logger: Optional[Any],
) -> None:
    """Record voting decision to the trace logger if available."""
    if trace_logger and hasattr(trace_logger, "record_answer_consistency_voting_trace"):
        try:
            trace_logger.record_answer_consistency_voting_trace(
                question=question,
                candidates=voting_trace.get("valid_candidates", {}),
                similarity_matrix=voting_trace.get("similarity_matrix", {}),
                consensus_found=voting_trace.get("consensus_found", False),
                llm_arbitration_used=voting_trace.get("llm_arbitration_used", False),
                final_answer=final_answer,
                answer_source=answer_source,
                decision_reason=voting_trace.get("decision", ""),
            )
        except Exception as exc:
            logger.debug("Failed to record voting trace: %s", exc)
