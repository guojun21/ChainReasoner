"""Query parsing, generation, and question-type detection.

Why: Converts a natural-language question into a set of targeted search
queries — the critical first step that determines search quality.
Supports both LLM-based decomposition and heuristic fallback.
"""

import re
from typing import Any, Callable, Dict, List, Optional

from src.agents.search_agent_shared_constants_and_stopwords import CHINESE_QUESTION_NOISE_STOPWORDS


def extract_structured_clues_from_question_text(question: str) -> Dict[str, Any]:
    """Extract structured clues (names, years, terms) from a question."""
    quoted = re.findall(r'[\""\u201c](.+?)[\""\u201d]', question)
    years = re.findall(r"\b(1[5-9]\d{2}|20\d{2})\b", question)
    numbers = re.findall(r"\b\d+\b", question)
    english_terms = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", question)
    cn_terms = [t for t in re.findall(r"[\u4e00-\u9fff]{2,6}", question) if t not in CHINESE_QUESTION_NOISE_STOPWORDS]
    english_names = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", question)
    return {
        "quoted": quoted, "years": years, "numbers": numbers,
        "english_terms": english_terms, "cn_terms": cn_terms, "english_names": english_names,
    }


def generate_search_queries_from_question(question: str, parsed: Dict[str, Any],
                     max_queries: int = 4,
                     llm_decompose_fn: Optional[Callable[[str], str]] = None) -> List[str]:
    """Produce search queries via LLM decomposition, with heuristic fallback."""
    queries: List[str] = []

    if llm_decompose_fn:
        try:
            raw_response = llm_decompose_fn(question)
            if raw_response:
                for line in raw_response.strip().split("\n"):
                    line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
                    line = re.sub(r"^[-\u2022]\s*", "", line).strip().strip('"').strip("'")
                    if line and len(line) >= 4:
                        queries.append(line)
        except Exception:
            pass

    if not queries:
        queries = _build_heuristic_search_queries_from_parsed_clues(question, parsed)

    return _remove_duplicate_queries_preserving_order(queries)[:max_queries]


def classify_question_into_answer_type_category(question: str) -> str:
    """Classify question to guide candidate extraction (year/number/person/place/entity)."""
    question_lower = question.lower()
    if any(k in question for k in ["哪一年", "哪年", "年份", "年代", "直接回答数字"]) or "year" in question_lower or "when" in question_lower:
        return "year"
    if any(k in question for k in ["多少", "数量", "页", "英里", "公里", "年龄", "总页数"]) or "how many" in question_lower or "number of" in question_lower:
        return "number"
    if any(k in question for k in ["谁", "哪位", "作者", "作家", "导演", "主持", "创立", "创办", "主演", "发明"]) or "who" in question_lower or "author" in question_lower:
        return "person"
    if any(k in question for k in ["哪里", "在哪", "哪国", "哪座", "城市"]) or "where" in question_lower or "location" in question_lower:
        return "place"
    return "entity"


def _build_heuristic_search_queries_from_parsed_clues(question: str, parsed: Dict[str, Any]) -> List[str]:
    """Build queries from parsed clues when LLM decomposition is unavailable."""
    queries: List[str] = []
    words = question.split()
    queries.append(question if len(words) <= 12 else " ".join(words[:12]))

    quoted = parsed.get("quoted", [])
    years = parsed.get("years", [])
    en_names = parsed.get("english_names", [])
    cn_terms = parsed.get("cn_terms", [])

    if quoted:
        queries.append(quoted[0])
        if years:
            queries.append(f"{quoted[0]} {years[0]}")
    if en_names:
        name_query = " ".join(en_names[:3])
        if years:
            name_query += f" {years[0]}"
        queries.append(name_query)
    if cn_terms and len(cn_terms) >= 2:
        queries.append(" ".join(cn_terms[:4]))
    return queries


def _remove_duplicate_queries_preserving_order(queries: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
    seen = set()
    unique = []
    for query in queries:
        query = query.strip()
        key = query.lower()
        if query and key not in seen:
            seen.add(key)
            unique.append(query)
    return unique
