#!/usr/bin/env python3
"""
Constrained Search Agent
Implements a search-only pipeline with heuristic reasoning.
Uses LLM-based query decomposition for multi-hop questions.
"""

import re
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple


_CN_STOPWORDS = {
    "问题", "答案", "哪些", "哪位", "哪年", "哪一年", "哪个", "哪个人", "哪里", "在哪",
    "是谁", "是什么", "如何", "为什么",
    "一个", "一种", "这个", "那个", "相关", "资料", "信息", "报告"
}

# Question-role words to exclude from person/entity answers (avoid "Author" etc.)
_ENTITY_ROLE_STOPWORDS = {
    "author", "director", "writer", "teacher", "guru", "composer", "person",
    "individual", "figure", "leader", "founder", "host", "presenter", "editor",
    "publisher", "singer", "artist", "scientist", "scholar", "researcher",
}
# Phrases that indicate LLM refusal; treat as no-answer and use heuristic
_REFUSAL_PHRASES = (
    "cannot be determined", "does not contain", "cannot be determined",
    "total number of pages cannot", "based on the given evidence",
    "no specific information", "cannot be determined from",
    "not enough information", "insufficient evidence",
)


class ConstrainedSearchAgent:
    """Constrained search agent using simple search results."""

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
        llm_decompose_fn: Optional[Callable[[str], str]] = None
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
        self._cache: Dict[str, List[Dict[str, Any]]] = {}

    def parse_question(self, question: str) -> Dict[str, Any]:
        """Parse question into basic clues."""
        quoted = re.findall(r'[\""\u201c](.+?)[\""\u201d]', question)
        years = re.findall(r"\b(1[5-9]\d{2}|20\d{2})\b", question)
        numbers = re.findall(r"\b\d+\b", question)
        english_terms = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", question)
        cn_terms = re.findall(r"[\u4e00-\u9fff]{2,6}", question)
        cn_terms = [t for t in cn_terms if t not in _CN_STOPWORDS]
        english_names = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", question)

        return {
            "quoted": quoted,
            "years": years,
            "numbers": numbers,
            "english_terms": english_terms,
            "cn_terms": cn_terms,
            "english_names": english_names
        }

    def generate_queries(self, question: str, parsed: Dict[str, Any]) -> List[str]:
        """Generate search queries using LLM decomposition, with heuristic fallback."""
        queries: List[str] = []

        # Strategy 1: Use LLM to decompose into smart search queries
        if self.llm_decompose_fn:
            try:
                raw = self.llm_decompose_fn(question)
                if raw:
                    for line in raw.strip().split("\n"):
                        line = line.strip()
                        # Remove numbered prefixes like "1.", "1)", "- "
                        line = re.sub(r"^\d+[\.\)]\s*", "", line)
                        line = re.sub(r"^[-\u2022]\s*", "", line)
                        line = line.strip().strip('"').strip("'")
                        if line and len(line) >= 4:
                            queries.append(line)
            except Exception:
                pass

        # Strategy 2: Heuristic fallback using parsed clues
        if not queries:
            quoted = parsed.get("quoted", [])
            years = parsed.get("years", [])
            cn_terms = parsed.get("cn_terms", [])
            en_names = parsed.get("english_names", [])

            # Use a substantial portion of the question as a query
            q_words = question.split()
            if len(q_words) <= 12:
                queries.append(question)
            else:
                queries.append(" ".join(q_words[:12]))

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

        # Deduplicate while preserving order
        deduped = []
        seen_lower = set()
        for q in queries:
            q = q.strip()
            ql = q.lower()
            if q and ql not in seen_lower:
                seen_lower.add(ql)
                deduped.append(q)
        return deduped[: self.max_queries]

    def _search(self, query: str) -> Tuple[List[Dict[str, Any]], str]:
        if query in self._cache:
            return self._cache[query], query
        # Skip rewrite for LLM-decomposed queries - they are already optimized
        # Rewriting was stripping context and making queries too generic
        rewritten = query
        try:
            result = self.search_fn(rewritten, {"original_query": query, "rewritten_query": rewritten})
        except TypeError:
            result = self.search_fn(rewritten)
        results = result.get("results", []) if isinstance(result, dict) else []
        self._cache[query] = results
        time.sleep(self.per_query_delay)
        return results, rewritten

    def _detect_question_type(self, question: str) -> str:
        q = question.lower()
        if any(k in question for k in ["哪一年", "哪年", "年份", "年代", "直接回答数字"]) or "year" in q or "when" in q:
            return "year"
        if any(k in question for k in ["多少", "数量", "页", "英里", "公里", "年龄", "总页数"]) or "how many" in q or "number of" in q or "total number" in q:
            return "number"
        if any(k in question for k in ["谁", "哪位", "作者", "作家", "导演", "主持", "创立", "创办", "主演", "发明"]) or "who" in q or "author" in q:
            return "person"
        if any(k in question for k in ["哪里", "在哪", "哪国", "哪座", "城市"]) or "where" in q or "location" in q:
            return "place"
        return "entity"

    def _extract_candidates(self, texts: List[str]) -> Dict[str, List[str]]:
        years = []
        numbers = []
        quoted = []
        english_titles = []
        cn_terms = []
        page_numbers = []

        for text in texts:
            years.extend(re.findall(r"\b(1[5-9]\d{2}|20\d{2})\b", text))
            numbers.extend(re.findall(r"\b\d+\b", text))
            quoted.extend(re.findall(r'[\""\u201c](.+?)[\""\u201d]', text))
            english_titles.extend(re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text))
            cn_terms.extend(re.findall(r"[\u4e00-\u9fff]{2,6}", text))
            page_numbers.extend(re.findall(r"\b(\d+)\s*(?:pages?|\u9875)\b", text, re.I))
            page_numbers.extend(re.findall(r"(?:total|pages?)\s*(?:of|:)?\s*(\d+)\b", text, re.I))

        cn_terms = [t for t in cn_terms if t not in _CN_STOPWORDS]
        return {
            "years": years,
            "numbers": numbers,
            "page_numbers": page_numbers,
            "quoted": quoted,
            "english_titles": english_titles,
            "cn_terms": cn_terms
        }

    def _select_candidate(self, question: str, candidates: Dict[str, List[str]]) -> str:
        qtype = self._detect_question_type(question)
        candidate_pool: List[str] = []
        q_lower = question.lower()

        if qtype == "year":
            candidate_pool = candidates.get("years", [])
        elif qtype == "number":
            if "page" in q_lower or "\u9875" in question:
                page_nums = candidates.get("page_numbers", [])
                if page_nums:
                    counter = Counter(page_nums)
                    best_num = counter.most_common(1)[0][0]
                    return str(best_num)
            candidate_pool = candidates.get("numbers", [])
        elif qtype in ("person", "place", "entity"):
            candidate_pool = candidates.get("quoted", [])
            candidate_pool += candidates.get("english_titles", [])
            candidate_pool += candidates.get("cn_terms", [])
            candidate_pool = [
                c for c in candidate_pool
                if c.strip()
                and c.strip().lower() not in _ENTITY_ROLE_STOPWORDS
                and not (len(c.strip().split()) <= 1 and c.strip().lower() in q_lower)
            ]

        if not candidate_pool:
            return "Unknown"

        counter = Counter(candidate_pool)
        best = None
        best_score = -1.0
        for cand, freq in counter.most_common():
            score = float(freq)
            if cand in question:
                score -= 1.0
            if len(cand) < 2:
                score -= 0.5
            if qtype == "number" and cand.isdigit():
                score += 0.2
            if score > best_score:
                best_score = score
                best = cand

        return best or "Unknown"

    def _clean_llm_answer(self, answer: str) -> str:
        """Clean LLM answer by removing common preamble/suffix patterns."""
        # Remove common prefixes like "The answer is: ", "Answer: ", "Based on..."
        prefixes = [
            r"^(?:the\s+)?answer\s*(?:is|:)\s*",
            r"^based\s+on\s+the\s+(?:evidence|information|search\s+results)[,:]?\s*",
            r"^according\s+to\s+the\s+(?:evidence|sources?)[,:]?\s*",
            r"^from\s+the\s+evidence[,:]?\s*",
        ]
        cleaned = answer
        for prefix in prefixes:
            cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE).strip()
        # Remove trailing period if the answer is short (likely a name/number)
        if len(cleaned) < 80 and cleaned.endswith("."):
            cleaned = cleaned[:-1].strip()
        # Remove surrounding quotes
        if len(cleaned) >= 2:
            if (cleaned[0] in '"\'«\u201c\u300c' and cleaned[-1] in '"\'»\u201d\u300d'):
                cleaned = cleaned[1:-1].strip()
        return cleaned

    def _tokenize_terms(self, query: str) -> List[str]:
        terms = re.split(r"[\s,;\uff0c\u3002/\\|]+", query.lower())
        return [t for t in terms if t and len(t) > 1]

    def _score_result(self, query: str, result: Dict[str, Any]) -> float:
        title = (result.get("title") or "").lower()
        content = (result.get("content") or "").lower()
        terms = self._tokenize_terms(query)
        if not terms:
            return 0.0
        title_hits = sum(1 for t in terms if t in title)
        content_hits = sum(1 for t in terms if t in content)
        score = title_hits * 2.0 + content_hits * 1.0
        if len(content) < 50:
            score -= 0.5
        return score

    def _rank_and_filter(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scored = []
        for r in results:
            score = self._score_result(query, r)
            item = dict(r)
            item["_score"] = score
            scored.append(item)

        scored.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
        # Keep all results to maximize evidence
        return scored[: self.max_results_per_query]

    def _select_evidence(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for r in results:
            key = (r.get("url"), r.get("title"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        deduped.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
        return deduped[: self.max_evidence]

    def _build_evidence_text(self, evidence: List[Dict[str, Any]]) -> str:
        parts = []
        for item in evidence:
            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "")
            if title or content:
                parts.append(f"[{title}]({url})\n{content}")
        text = "\n\n".join(parts)
        return text[:6000]

    def _verify_answer(self, question: str, answer: str, evidence_text: str) -> bool:
        if not self.llm_verify_fn:
            return True
        label, confidence = self.llm_verify_fn(question, answer, evidence_text)
        if label == "SUPPORTS" and confidence >= 0.5:
            return True
        return False

    def answer(self, question: str) -> Dict[str, Any]:
        """Generate answer using constrained search."""
        parsed = self.parse_question(question)
        queries = self.generate_queries(question, parsed)

        reasoning_steps = []
        reasoning_steps.append(f"Queries: {queries}")
        reasoning_steps.append(f"Query count: {len(queries)}")

        all_texts = []
        all_ranked_results = []
        search_traces = []
        for q in queries:
            results, rewritten = self._search(q)
            ranked = self._rank_and_filter(rewritten, results)
            search_traces.append({"query": q, "rewritten": rewritten, "count": len(ranked)})
            all_ranked_results.extend(ranked)
            for item in ranked:
                title = item.get("title", "")
                content = item.get("content", "")
                all_texts.append(f"{title} {content}")

        evidence = self._select_evidence(all_ranked_results)
        evidence_text = self._build_evidence_text(evidence)
        candidates = self._extract_candidates(all_texts)
        heuristic_answer = self._select_candidate(question, candidates)
        final_answer = heuristic_answer

        if self.llm_answer_fn and evidence_text:
            llm_answer = self.llm_answer_fn(question, evidence_text)
            if llm_answer:
                candidate = llm_answer.strip()
                # Remove common LLM preamble patterns
                candidate = self._clean_llm_answer(candidate)
                # Treat long refusals as no-answer and use heuristic
                is_refusal = (
                    len(candidate) > 60
                    and any(p in candidate.lower() for p in _REFUSAL_PHRASES)
                )
                # Also reject generic role words as answers
                is_role_word = (
                    candidate.strip().lower() in _ENTITY_ROLE_STOPWORDS
                    or candidate.strip().lower() in {"the author", "the director", "the writer",
                        "the founder", "the leader", "the president", "the company",
                        "the organization", "the person", "the individual"}
                )
                if is_refusal or is_role_word:
                    candidate = ""
                if candidate and len(candidate) < 500:
                    final_answer = candidate
                elif candidate:
                    # If LLM gave a long response, take first meaningful line
                    lines = [l.strip() for l in candidate.split("\n") if l.strip()]
                    if lines:
                        final_answer = lines[0]

        reasoning_steps.append(f"Evidence count: {len(evidence)}")
        reasoning_steps.append(f"Final answer: {final_answer}")

        return {
            "answer": final_answer,
            "reasoning_steps": reasoning_steps,
            "search_traces": search_traces
        }
