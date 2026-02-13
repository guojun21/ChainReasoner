"""Question domain classifier — routes each hop to the best search strategy.

Why (Wing architecture): Wing uses 5 domain categories to decide which search
engine and query style to use for each hop.  Academic questions need Google
Scholar-style queries, government questions need IQS for Chinese official data,
business questions need precise company name search, etc.

Categories:
  1. 学术与科学贡献 (academic)   — papers, discoveries, awards, inventions
  2. 企业与商业信息 (business)   — companies, products, financials, founders
  3. 政府与公开数据 (government) — policies, statistics, official records
  4. 文化与体育数据 (culture)    — sports, arts, media, entertainment, history
  5. 新闻与媒体档案 (news)       — current events, journalism, press releases
"""

import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Valid domain labels — used for validation and search routing
VALID_DOMAINS = ("academic", "business", "government", "culture", "news")

_CLASSIFY_SYSTEM_PROMPT = """\
You are a question domain classifier.  Given a question or a search hop target,
classify it into exactly ONE of these 5 domains:

1. academic   — scientific research, papers, discoveries, awards, inventions, Nobel prizes, university-related
2. business   — companies, products, financials, founders, CEOs, corporate events, industry
3. government — government policies, legislation, official statistics, public records, census
4. culture    — sports, arts, music, film, TV, literature, entertainment, history, geography, cultural heritage
5. news       — current events, breaking news, journalism, press releases, media coverage

Reply with ONLY the single domain keyword (e.g. "academic").  No explanation.
"""


def classify_question_domain(
    question: str,
    hop_target: str = "",
    llm_fn: Optional[Callable[[str, str], str]] = None,
) -> str:
    """Classify a question/hop into one of 5 domains.

    Falls back to keyword heuristics when LLM is unavailable.

    Returns one of: "academic", "business", "government", "culture", "news".
    """
    # Try LLM classification first (most accurate)
    if llm_fn:
        try:
            user_prompt = f"Question: {question}"
            if hop_target:
                user_prompt += f"\nCurrent hop target: {hop_target}"
            raw = llm_fn(_CLASSIFY_SYSTEM_PROMPT, user_prompt)
            if raw:
                label = raw.strip().lower().strip('"').strip("'").strip(".")
                if label in VALID_DOMAINS:
                    return label
                # Fuzzy match for Chinese labels
                _zh_map = {
                    "学术": "academic", "科学": "academic", "科研": "academic",
                    "企业": "business", "商业": "business", "公司": "business",
                    "政府": "government", "公开数据": "government", "政策": "government",
                    "文化": "culture", "体育": "culture", "历史": "culture",
                    "新闻": "news", "媒体": "news",
                }
                for zh_key, en_val in _zh_map.items():
                    if zh_key in label:
                        return en_val
        except Exception as exc:
            logger.debug("LLM domain classification failed: %s", exc)

    # Keyword heuristic fallback
    text = f"{question} {hop_target}".lower()
    return _classify_by_keywords(text)


def _classify_by_keywords(text: str) -> str:
    """Rule-based domain classification using keyword matching.

    Why: When LLM is unavailable (or for speed), keyword matching provides
    a reasonable approximation of the domain.
    """
    academic_kw = [
        "研究", "论文", "发表", "发现", "诺贝尔", "教授", "大学", "学院",
        "scientist", "professor", "university", "research", "discovery",
        "nobel", "award", "published", "journal", "phd", "thesis",
        "institute", "experiment", "theory",
    ]
    business_kw = [
        "公司", "企业", "CEO", "创始人", "股票", "市值", "收入",
        "company", "corporation", "founder", "revenue", "stock",
        "business", "startup", "acquired", "merger", "brand",
        "headquarters", "subsidiary", "IPO",
    ]
    government_kw = [
        "政府", "政策", "法律", "法规", "统计", "人口",
        "government", "policy", "legislation", "census", "official",
        "ministry", "regulation", "federal", "municipal", "statute",
    ]
    news_kw = [
        "新闻", "报道", "记者", "事件", "突发",
        "news", "reported", "journalist", "breaking", "press",
        "coverage", "headline", "media",
    ]

    # Score each domain
    scores = {
        "academic": sum(1 for kw in academic_kw if kw in text),
        "business": sum(1 for kw in business_kw if kw in text),
        "government": sum(1 for kw in government_kw if kw in text),
        "news": sum(1 for kw in news_kw if kw in text),
    }

    best_domain = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best_domain] > 0:
        return best_domain
    # Default: culture is the broadest catch-all (history, geography, sports, etc.)
    return "culture"
