"""Answer cleaning, candidate extraction, and verification.

Why: LLM answers are noisy — they contain preambles, refusals,
and formatting artefacts.  This module strips all that down to
the minimal answer string the competition scoring expects.
"""

import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agents.search_agent_shared_constants_and_stopwords import (
    CHINESE_QUESTION_NOISE_STOPWORDS,
    GENERIC_ENTITY_ROLE_WORDS_TO_REJECT,
    LLM_REFUSAL_AND_INSUFFICIENT_EVIDENCE_PHRASES,
)


def strip_llm_preambles_and_extract_core_answer(answer: str) -> str:
    """Strip common LLM preambles/suffixes to extract the core answer."""
    prefixes = [
        r"^(?:the\s+)?answer\s*(?:is|:)\s*",
        r"^based\s+on\s+the\s+(?:evidence|information|search\s+results)[,:]?\s*",
        r"^according\s+to\s+the\s+(?:evidence|sources?)[,:]?\s*",
        r"^from\s+the\s+evidence[,:]?\s*",
    ]
    cleaned = answer
    for prefix in prefixes:
        cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE).strip()
    if len(cleaned) < 80 and cleaned.endswith("."):
        cleaned = cleaned[:-1].strip()
    if len(cleaned) >= 2 and cleaned[0] in '"\'«\u201c\u300c' and cleaned[-1] in '"\'»\u201d\u300d':
        cleaned = cleaned[1:-1].strip()
    return cleaned


def check_if_answer_is_refusal_or_unknown_placeholder(text: str) -> bool:
    """True if the text is a refusal, unknown marker, or generic role word.

    Why: The competition requires specific factual answers; generic
    responses like "Unknown" or "The author" score zero.
    """
    if not text or not text.strip():
        return True
    cleaned = text.strip().lower()
    if cleaned in ("unknown", "n/a", "none", "not found", "no answer"):
        return True
    if cleaned in GENERIC_ENTITY_ROLE_WORDS_TO_REJECT:
        return True
    if cleaned in {"the author", "the director", "the writer", "the founder",
                    "the leader", "the president", "the company",
                    "the organization", "the person", "the individual"}:
        return True
    if len(cleaned) > 40 and any(phrase in cleaned for phrase in LLM_REFUSAL_AND_INSUFFICIENT_EVIDENCE_PHRASES):
        return True
    return False


def clean_and_validate_raw_llm_answer_text(raw_answer: str) -> str:
    """Clean + validate a raw LLM answer; returns empty string on refusal."""
    if not raw_answer:
        return ""
    candidate = strip_llm_preambles_and_extract_core_answer(raw_answer.strip())
    if check_if_answer_is_refusal_or_unknown_placeholder(candidate):
        return ""
    if len(candidate) < 200:
        return candidate
    # Long response — extract first meaningful short line
    for line in candidate.split("\n"):
        line = line.strip()
        if not line:
            continue
        cleaned_line = strip_llm_preambles_and_extract_core_answer(line)
        if cleaned_line and not check_if_answer_is_refusal_or_unknown_placeholder(cleaned_line) and len(cleaned_line) < 200:
            return cleaned_line
    short = candidate[:200].strip()
    if "\n" in short:
        short = short.split("\n")[0].strip()
    if short and not check_if_answer_is_refusal_or_unknown_placeholder(short):
        return short
    return ""


def extract_structured_answer_candidates_from_evidence(texts: List[str]) -> Dict[str, List[str]]:
    """Extract structured candidates (years, numbers, names) from evidence texts."""
    years, numbers, quoted, english_titles, cn_terms, page_numbers = [], [], [], [], [], []
    for text in texts:
        years.extend(re.findall(r"\b(1[5-9]\d{2}|20\d{2})\b", text))
        numbers.extend(re.findall(r"\b\d+\b", text))
        quoted.extend(re.findall(r'[\""\u201c](.+?)[\""\u201d]', text))
        english_titles.extend(re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text))
        cn_terms.extend(re.findall(r"[\u4e00-\u9fff]{2,6}", text))
        page_numbers.extend(re.findall(r"\b(\d+)\s*(?:pages?|\u9875)\b", text, re.I))
        page_numbers.extend(re.findall(r"(?:total|pages?)\s*(?:of|:)?\s*(\d+)\b", text, re.I))
    cn_terms = [term for term in cn_terms if term not in CHINESE_QUESTION_NOISE_STOPWORDS]
    return {
        "years": years, "numbers": numbers, "page_numbers": page_numbers,
        "quoted": quoted, "english_titles": english_titles, "cn_terms": cn_terms,
    }


def select_highest_frequency_candidate_as_answer(question: str, candidates: Dict[str, List[str]],
                          question_type: str) -> str:
    """Pick the single best heuristic answer from extracted candidates.

    Why: When LLM extraction fails, frequency-based candidate selection
    from search snippets provides a reasonable fallback answer.
    """
    candidate_pool: List[str] = []
    question_lower = question.lower()

    if question_type == "year":
        candidate_pool = candidates.get("years", [])
    elif question_type == "number":
        if "page" in question_lower or "\u9875" in question:
            page_nums = candidates.get("page_numbers", [])
            if page_nums:
                return Counter(page_nums).most_common(1)[0][0]
        candidate_pool = candidates.get("numbers", [])
    elif question_type in ("person", "place", "entity"):
        candidate_pool = candidates.get("quoted", []) + candidates.get("english_titles", []) + candidates.get("cn_terms", [])
        candidate_pool = [
            c for c in candidate_pool
            if c.strip()
            and c.strip().lower() not in GENERIC_ENTITY_ROLE_WORDS_TO_REJECT
            and not (len(c.strip().split()) <= 1 and c.strip().lower() in question_lower)
        ]

    if not candidate_pool:
        return "Unknown"

    counter = Counter(candidate_pool)
    best, best_score = None, -1.0
    for candidate, freq in counter.most_common():
        score = float(freq)
        if candidate in question:
            score -= 1.0
        if len(candidate) < 2:
            score -= 0.5
        if question_type == "number" and candidate.isdigit():
            score += 0.2
        if score > best_score:
            best_score = score
            best = candidate
    return best or "Unknown"


def verify_answer_against_evidence_using_llm(question: str, answer: str, evidence_text: str,
                  llm_verify_fn: Optional[Callable] = None) -> Tuple[bool, str, str, float]:
    """Ask the LLM to confirm or refute an answer against evidence.

    Returns:
        (is_valid, verdict_label, reasoning, confidence).
        is_valid is True only when verdict is SUPPORTS with confidence >= 0.5.
        verdict_label is one of SUPPORTS/REFUTES/INSUFFICIENT.
        reasoning contains the verification reasoning text (useful for
        extracting correct entities when REFUTES).
    """
    if not llm_verify_fn:
        return True, "SUPPORTS", "", 1.0
    label, confidence, reasoning = llm_verify_fn(question, answer, evidence_text)
    is_valid = label == "SUPPORTS" and confidence >= 0.5
    return is_valid, label, reasoning, confidence
