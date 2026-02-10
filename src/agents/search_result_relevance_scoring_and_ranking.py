"""Evidence scoring, ranking, and selection.

Why: Search returns many results but only the most relevant ones
should be fed to the LLM.  This module scores each result by
query-term overlap and source authority, then deduplicates.
"""

import re
from typing import Any, Dict, List


def split_query_into_lowercase_search_terms(query: str) -> List[str]:
    """Split query into lowercase terms for scoring."""
    terms = re.split(r"[\s,;\uff0c\u3002/\\|]+", query.lower())
    return [term for term in terms if term and len(term) > 1]


def calculate_single_search_result_relevance_score(query: str, result: Dict[str, Any]) -> float:
    """Score a single search result by keyword overlap and source authority.

    Why: Title matches are weighted 2x because titles are more precise;
    Wikipedia/Britannica get authority boosts because they contain
    verified factual content critical for competition accuracy.
    """
    title = (result.get("title") or "").lower()
    content = (result.get("content") or "").lower()
    url = (result.get("url") or "").lower()
    terms = split_query_into_lowercase_search_terms(query)
    if not terms:
        return 0.0

    title_hits = sum(1 for term in terms if term in title)
    content_hits = sum(1 for term in terms if term in content)
    score = title_hits * 2.0 + content_hits * 1.0

    if len(content) < 50:
        score -= 0.5
    if "wikipedia.org" in url:
        score += 3.0
    elif any(domain in url for domain in ["britannica.com", "baidu.com/baike", "zhihu.com", "ncbi.nlm.nih.gov"]):
        score += 1.5
    return score


def rank_search_results_by_relevance_and_truncate(query: str, results: List[Dict[str, Any]],
                    max_results: int = 10) -> List[Dict[str, Any]]:
    """Rank results by relevance score and truncate."""
    scored = []
    for result in results:
        item = dict(result)
        item["_score"] = calculate_single_search_result_relevance_score(query, result)
        scored.append(item)
    scored.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
    return scored[:max_results]


def deduplicate_and_select_top_evidence_items(results: List[Dict[str, Any]], max_evidence: int = 12) -> List[Dict[str, Any]]:
    """Deduplicate by (url, title) and keep top-scored evidence items."""
    seen = set()
    unique = []
    for result in results:
        key = (result.get("url"), result.get("title"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
    unique.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
    return unique[:max_evidence]


def format_evidence_items_into_llm_readable_text(evidence: List[Dict[str, Any]]) -> str:
    """Format evidence items into a single text block for LLM consumption."""
    parts = []
    for item in evidence:
        title = item.get("title", "")
        url = item.get("url", "")
        content = item.get("content", "")
        if title or content:
            parts.append(f"[{title}]({url})\n{content}")
    return "\n\n".join(parts)[:10000]
