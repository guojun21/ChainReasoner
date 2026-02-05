#!/usr/bin/env python3
"""
Constrained Search Agent
Implements a search-only pipeline with heuristic reasoning.
"""

import re
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple


_CN_STOPWORDS = {
    "问题", "答案", "哪些", "哪位", "哪年", "哪一年", "哪个", "哪个人", "哪里", "在哪",
    "作者", "作家", "导演", "主演", "主持", "是谁", "是什么", "如何", "为什么",
    "一个", "一种", "这个", "那个", "相关", "资料", "信息", "研究", "报告"
}


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
        llm_summarize_fn: Optional[Callable[[str], str]] = None
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
        self._cache: Dict[str, List[Dict[str, Any]]] = {}

    def parse_question(self, question: str) -> Dict[str, Any]:
        """Parse question into basic clues."""
        quoted = re.findall(r"[\"“](.+?)[\"”]", question)
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

    def score_breakthrough(self, clue: str, parsed: Dict[str, Any]) -> float:
        """Score a clue for breakthrough potential."""
        uniqueness = 1.0
        if any(ch.isdigit() for ch in clue):
            uniqueness += 0.8
        if len(clue) >= 6:
            uniqueness += 0.5
        if clue in parsed.get("quoted", []):
            uniqueness += 0.7

        searchability = 0.5
        if re.search(r"[A-Za-z]", clue):
            searchability += 0.6
        if re.search(r"[\u4e00-\u9fff]", clue):
            searchability += 0.4
        if any(ch.isdigit() for ch in clue):
            searchability += 0.3

        other_clues = (
            len(parsed.get("years", [])) +
            len(parsed.get("numbers", [])) +
            len(parsed.get("english_terms", [])) +
            len(parsed.get("cn_terms", []))
        )
        verifiability = 1.0 if other_clues >= 2 else 0.7

        return uniqueness * searchability * verifiability

    def identify_breakthrough(self, parsed: Dict[str, Any]) -> List[str]:
        """Identify breakthrough clues."""
        clues = []
        clues.extend(parsed.get("quoted", []))
        clues.extend(parsed.get("years", []))
        clues.extend(parsed.get("numbers", []))
        clues.extend(parsed.get("english_terms", []))
        clues.extend(parsed.get("cn_terms", []))

        scored = [(c, self.score_breakthrough(c, parsed)) for c in clues]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored]

    def _decompose_question(self, question: str) -> List[str]:
        """Heuristic question decomposition for multi-clause prompts."""
        splitters = ["并且", "以及", "同时", "此外", "而且", "另外"]
        segments = [question]
        for splitter in splitters:
            new_segments = []
            for seg in segments:
                parts = seg.split(splitter)
                new_segments.extend(parts)
            segments = new_segments
        rough = []
        for seg in segments:
            seg = seg.strip()
            if len(seg) >= 8:
                rough.append(seg)
        return rough[:2]

    def generate_queries(self, question: str, parsed: Dict[str, Any]) -> List[str]:
        """Generate search queries based on breakthrough clues."""
        breakthroughs = self.identify_breakthrough(parsed)
        queries: List[str] = []

        quoted = parsed.get("quoted", [])
        years = parsed.get("years", [])
        cn_terms = parsed.get("cn_terms", [])
        en_terms = parsed.get("english_terms", [])

        if quoted:
            queries.append(f"\"{quoted[0]}\"")
            if years:
                queries.append(f"\"{quoted[0]}\" {years[0]}")
        if breakthroughs:
            queries.append(breakthroughs[0])
        if len(breakthroughs) > 1:
            queries.append(f"{breakthroughs[0]} {breakthroughs[1]}")
        if cn_terms:
            queries.append(" ".join(cn_terms[:2]))
        if en_terms:
            queries.append(" ".join(en_terms[:3]))
        if years and cn_terms:
            queries.append(f"{years[0]} {cn_terms[0]}")

        sub_questions = self._decompose_question(question)
        for sub in sub_questions:
            sub_parsed = self.parse_question(sub)
            sub_breakthroughs = self.identify_breakthrough(sub_parsed)
            if sub_breakthroughs:
                queries.append(sub_breakthroughs[0])

        # Fallback to a trimmed question if no clue found
        if not queries:
            queries.append(" ".join(question.split()[:8]))

        # Deduplicate while preserving order
        deduped = []
        for q in queries:
            q = q.strip()
            if q and q not in deduped:
                deduped.append(q)
        return deduped[: self.max_queries]

    def _search(self, query: str) -> Tuple[List[Dict[str, Any]], str]:
        if query in self._cache:
            return self._cache[query], query
        rewritten = self.rewrite_fn(query) if self.rewrite_fn else query
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
        if any(k in question for k in ["哪一年", "哪年", "年份", "年代"]) or "year" in q or "when" in q:
            return "year"
        if any(k in question for k in ["多少", "几", "数量", "页", "英里", "公里", "年龄"]) or "how many" in q or "number" in q:
            return "number"
        if any(k in question for k in ["谁", "哪位", "作者", "作家", "导演", "主持", "创立", "创办", "主演", "发明"]) or "who" in q or "author" in q:
            return "person"
        if any(k in question for k in ["哪里", "在哪", "哪国", "哪座"]) or "where" in q or "location" in q:
            return "place"
        return "entity"

    def _extract_candidates(self, texts: List[str]) -> Dict[str, List[str]]:
        years = []
        numbers = []
        quoted = []
        english_titles = []
        cn_terms = []

        for text in texts:
            years.extend(re.findall(r"\b(1[5-9]\d{2}|20\d{2})\b", text))
            numbers.extend(re.findall(r"\b\d+\b", text))
            quoted.extend(re.findall(r"[\"“](.+?)[\"”]", text))
            english_titles.extend(re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text))
            cn_terms.extend(re.findall(r"[\u4e00-\u9fff]{2,6}", text))

        cn_terms = [t for t in cn_terms if t not in _CN_STOPWORDS]
        return {
            "years": years,
            "numbers": numbers,
            "quoted": quoted,
            "english_titles": english_titles,
            "cn_terms": cn_terms
        }

    def _select_candidate(self, question: str, candidates: Dict[str, List[str]]) -> str:
        qtype = self._detect_question_type(question)
        candidate_pool: List[str] = []

        if qtype == "year":
            candidate_pool = candidates.get("years", [])
        elif qtype == "number":
            candidate_pool = candidates.get("numbers", [])
        elif qtype in ("person", "place", "entity"):
            candidate_pool = candidates.get("quoted", [])
            candidate_pool += candidates.get("english_titles", [])
            candidate_pool += candidates.get("cn_terms", [])

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

    def _tokenize_terms(self, query: str) -> List[str]:
        terms = re.split(r"[\s,;，。/\\|]+", query.lower())
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
        scores = [r.get("_score", 0.0) for r in scored]
        if not scores:
            return []
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = variance ** 0.5
        threshold = mean - self.adaptive_threshold_n * std
        filtered = [r for r in scored if r.get("_score", 0.0) >= threshold]
        if not filtered:
            filtered = scored[:3]
        return filtered[: self.max_results_per_query]

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
            content = item.get("content", "")
            if title or content:
                parts.append(f"{title} - {content}")
        text = "\n".join(parts)
        return text[:4000]

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
        reasoning_steps.append(f"解析线索: quoted={len(parsed['quoted'])}, years={len(parsed['years'])}, numbers={len(parsed['numbers'])}")
        if queries:
            reasoning_steps.append(f"突破口: {queries[0]}")
        reasoning_steps.append(f"查询数量: {len(queries)}")

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
        if self.llm_summarize_fn and evidence_text:
            summary = self.llm_summarize_fn(evidence_text)
            if summary:
                reasoning_steps.append("证据摘要: 已压缩")
                evidence_text = summary
        candidates = self._extract_candidates(all_texts)
        heuristic_answer = self._select_candidate(question, candidates)
        final_answer = heuristic_answer

        if self.llm_answer_fn and evidence_text:
            llm_answer = self.llm_answer_fn(question, evidence_text)
            if llm_answer:
                candidate = llm_answer.strip()
                if self._verify_answer(question, candidate, evidence_text):
                    final_answer = candidate
                else:
                    final_answer = heuristic_answer

        reasoning_steps.append(f"证据数: {len(evidence)}")
        reasoning_steps.append(f"候选答案数: {sum(len(v) for v in candidates.values())}")
        reasoning_steps.append(f"最终答案: {final_answer}")

        return {
            "answer": final_answer,
            "reasoning_steps": reasoning_steps,
            "search_traces": search_traces
        }
