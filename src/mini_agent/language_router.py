"""Question geographic-language router — determines search language priority.

Why: A Chinese-language question about a British company (e.g. RepRapPro Ltd)
needs English-first search because English sources have far more coverage.
The current system treats all Chinese questions equally, missing critical
English-language evidence.  This module adds a three-layer geographic-language
routing system that decides whether to prioritise English or Chinese searches.

Architecture (three layers, fast to slow):
  Layer 1: Signal detection — extract geographic/language signals from text (zero cost)
  Layer 2: Rule engine — deterministic rules from signal combinations (zero cost)
  Layer 3: LLM judgment — only when signals are ambiguous (+1 LLM call)
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language priority constants
# ---------------------------------------------------------------------------

LANGUAGE_PRIORITY_ENGLISH_FIRST = "english_first"
LANGUAGE_PRIORITY_CHINESE_FIRST = "chinese_first"
LANGUAGE_PRIORITY_BILINGUAL_EQUAL = "bilingual_equal"

# Query ratio suggestions for each priority
_QUERY_RATIO_MAP = {
    LANGUAGE_PRIORITY_ENGLISH_FIRST: {"en": 0.7, "zh": 0.3},
    LANGUAGE_PRIORITY_CHINESE_FIRST: {"en": 0.3, "zh": 0.7},
    LANGUAGE_PRIORITY_BILINGUAL_EQUAL: {"en": 0.5, "zh": 0.5},
}

# ---------------------------------------------------------------------------
# Signal keyword lists
# ---------------------------------------------------------------------------

_WESTERN_GEOGRAPHY_KEYWORDS = frozenset({
    # Countries / regions (Chinese)
    "欧洲", "美国", "英国", "法国", "德国", "意大利", "西班牙", "荷兰",
    "瑞士", "瑞典", "挪威", "丹麦", "芬兰", "比利时", "奥地利", "波兰",
    "俄罗斯", "加拿大", "澳大利亚", "新西兰", "巴西", "墨西哥", "印度",
    "日本", "韩国", "以色列", "土耳其", "南非", "阿根廷",
    # Countries / regions (English)
    "Europe", "America", "USA", "UK", "Britain", "France", "Germany",
    "Italy", "Spain", "Netherlands", "Switzerland", "Sweden", "Norway",
    "Denmark", "Finland", "Belgium", "Austria", "Poland", "Russia",
    "Canada", "Australia", "Brazil", "Mexico", "India", "Japan", "Korea",
    "Israel", "Turkey",
    # International organisations
    "联合国", "世界银行", "国际货币基金", "世贸组织", "北约",
    "United Nations", "World Bank", "NATO", "WHO", "WTO", "IMF",
    # Famous western institutions
    "Harvard", "MIT", "Stanford", "Oxford", "Cambridge", "Princeton",
    "Yale", "Columbia", "Berkeley", "Caltech",
    "哈佛", "麻省理工", "斯坦福", "牛津", "剑桥",
})

_CHINESE_GEOGRAPHY_KEYWORDS = frozenset({
    # China-specific
    "中国", "北京", "上海", "广州", "深圳", "天津", "重庆", "成都",
    "武汉", "杭州", "南京", "西安", "台湾", "香港", "澳门",
    # Chinese dynasties
    "清朝", "明朝", "唐朝", "宋朝", "汉朝", "秦朝", "元朝", "隋朝",
    "三国", "春秋", "战国", "魏晋", "南北朝",
    # Chinese politics
    "国务院", "人大", "政协", "中共", "共产党", "中央", "省委",
    "全国人大", "全国政协",
    # Chinese landmarks
    "故宫", "长城", "天安门", "颐和园", "兵马俑",
    # Chinese culture
    "儒家", "道家", "佛教", "四书五经", "科举",
})

_INTERNATIONAL_ABBREVIATIONS = frozenset({
    "NASA", "FBI", "CIA", "NATO", "WHO", "UN", "EU", "IMF", "WTO",
    "GDP", "CEO", "IPO", "IEEE", "ACM", "AAAI", "MIT", "PhD", "MBA",
    "FIFA", "NBA", "NFL", "IOC", "OPEC", "ASEAN", "BRICS",
    "CERN", "ESA", "DARPA", "NIH", "CDC", "FDA",
})


# ---------------------------------------------------------------------------
# Layer 1: Signal detection (zero LLM cost)
# ---------------------------------------------------------------------------

def _detect_language_priority_signals_from_question_text(
    question: str,
) -> Dict[str, Any]:
    """Extract geographic/language signals from question text.

    Why: Fast heuristic signals (regex + keyword matching) can resolve
    most cases without any LLM call.  Only ambiguous cases need Layer 3.
    """
    signals: List[str] = []

    # Signal 1: English entity names (capitalised multi-word sequences)
    english_names = re.findall(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b", question
    )
    if english_names:
        signals.append("english_entity_names")

    # Signal 2: Western geography keywords
    if any(kw in question for kw in _WESTERN_GEOGRAPHY_KEYWORDS):
        signals.append("western_geography")

    # Signal 3: Chinese geography keywords
    if any(kw in question for kw in _CHINESE_GEOGRAPHY_KEYWORDS):
        signals.append("chinese_geography")

    # Signal 4: International abbreviations (all-caps 2-6 letters)
    found_abbrevs = set(re.findall(r"\b[A-Z]{2,6}\b", question))
    if found_abbrevs & _INTERNATIONAL_ABBREVIATIONS:
        signals.append("international_abbreviations")

    # Signal 5: Question language detection
    cn_chars = re.findall(r"[\u4e00-\u9fff]", question)
    cn_char_ratio = len(cn_chars) / max(len(question), 1)
    if cn_char_ratio > 0.3:
        signals.append("question_in_chinese")
    else:
        signals.append("question_in_english")

    # Signal 6: English words in a Chinese question (mixed-language indicator)
    if "question_in_chinese" in signals:
        en_words = re.findall(r"\b[a-zA-Z]{3,}\b", question)
        if len(en_words) >= 2:
            signals.append("english_terms_in_chinese_question")

    return {
        "signals": signals,
        "english_names": english_names,
        "cn_char_ratio": cn_char_ratio,
    }


# ---------------------------------------------------------------------------
# Layer 2: Rule engine (zero LLM cost)
# ---------------------------------------------------------------------------

def _apply_language_routing_rules(signals: List[str]) -> str:
    """Apply deterministic rules to decide language priority from signals.

    Why: Most questions can be routed correctly with simple rules.
    LLM judgment (Layer 3) is only needed for truly ambiguous cases.
    """
    has_en_entities = "english_entity_names" in signals
    has_western_geo = "western_geography" in signals
    has_intl_abbrev = "international_abbreviations" in signals
    has_cn_geo = "chinese_geography" in signals
    has_en_terms_in_cn = "english_terms_in_chinese_question" in signals
    is_cn_question = "question_in_chinese" in signals

    # Rule 1: Chinese question + English entities/western geography -> English first
    if is_cn_question and (has_en_entities or has_western_geo or has_intl_abbrev):
        if has_cn_geo:
            return LANGUAGE_PRIORITY_BILINGUAL_EQUAL  # Mixed topic
        return LANGUAGE_PRIORITY_ENGLISH_FIRST

    # Rule 2: Chinese question + English terms embedded -> likely English topic
    if is_cn_question and has_en_terms_in_cn and not has_cn_geo:
        return LANGUAGE_PRIORITY_ENGLISH_FIRST

    # Rule 3: Chinese question + Chinese geography + no English signals -> Chinese first
    if is_cn_question and has_cn_geo and not has_en_entities and not has_western_geo:
        return LANGUAGE_PRIORITY_CHINESE_FIRST

    # Rule 4: English question -> English first
    if not is_cn_question:
        return LANGUAGE_PRIORITY_ENGLISH_FIRST

    # Default: bilingual equal
    return LANGUAGE_PRIORITY_BILINGUAL_EQUAL


# ---------------------------------------------------------------------------
# Layer 3: LLM judgment (only when signals are ambiguous)
# ---------------------------------------------------------------------------

_LANGUAGE_ROUTING_LLM_SYSTEM_PROMPT = """\
Given a question, determine which language world the question's subject belongs to.

- "english_first": The event/entity is primarily from the English-speaking world \
(Western countries, international organizations, English-named entities). \
English sources will have far better coverage.
- "chinese_first": The event/entity is primarily from the Chinese-speaking world \
(China, Chinese history, Chinese politics, Chinese culture). \
Chinese sources will have far better coverage.
- "bilingual_equal": The topic spans both worlds equally.

Reply with ONLY one of: english_first, chinese_first, bilingual_equal"""


def _classify_with_llm_judgment(
    question: str,
    llm_fn: Callable[[str, str], str],
) -> Optional[str]:
    """Use LLM to determine language priority when rules are ambiguous.

    Why: Some questions have conflicting signals (e.g. Chinese question about
    a topic that could be either Chinese or Western).  The LLM can understand
    the semantic content to make a better judgment.
    """
    try:
        raw = llm_fn(_LANGUAGE_ROUTING_LLM_SYSTEM_PROMPT, f"Question: {question}")
        if raw:
            label = raw.strip().lower().strip('"').strip("'").strip(".")
            if label in (LANGUAGE_PRIORITY_ENGLISH_FIRST,
                         LANGUAGE_PRIORITY_CHINESE_FIRST,
                         LANGUAGE_PRIORITY_BILINGUAL_EQUAL):
                return label
    except Exception as exc:
        logger.debug("LLM language routing failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def classify_question_geographic_language_priority(
    question: str,
    hop_target: str = "",
    llm_fn: Optional[Callable[[str, str], str]] = None,
) -> Dict[str, Any]:
    """Determine search language priority based on where the event/entity belongs.

    Why: A Chinese-language question about a British company needs English-first
    search.  This three-layer system (signals -> rules -> LLM) makes that
    determination efficiently, using LLM only when heuristics are insufficient.

    Args:
        question: The original question text.
        hop_target: Optional current hop target for more context.
        llm_fn: Optional LLM function for Layer 3 judgment.

    Returns:
        Dict with keys:
            priority: "english_first" | "chinese_first" | "bilingual_equal"
            confidence: 0.0-1.0
            reasoning: Brief explanation
            suggested_query_ratio: {"en": float, "zh": float}
            detected_signals: List of signal names
    """
    # Combine question and hop target for signal detection
    full_text = question
    if hop_target:
        full_text = f"{question} {hop_target}"

    # Layer 1: Signal detection
    signal_result = _detect_language_priority_signals_from_question_text(full_text)
    signals = signal_result["signals"]

    # Layer 2: Rule engine
    priority = _apply_language_routing_rules(signals)
    confidence = 0.8  # Rules are generally reliable

    # Layer 3: LLM judgment (only for bilingual_equal — the ambiguous case)
    used_llm = False
    if priority == LANGUAGE_PRIORITY_BILINGUAL_EQUAL and llm_fn:
        llm_result = _classify_with_llm_judgment(question, llm_fn)
        if llm_result:
            priority = llm_result
            confidence = 0.9
            used_llm = True

    # Build reasoning
    reasoning_parts = [f"signals={signals}"]
    if used_llm:
        reasoning_parts.append("refined_by_llm")
    reasoning = ", ".join(reasoning_parts)

    result = {
        "priority": priority,
        "confidence": confidence,
        "reasoning": reasoning,
        "suggested_query_ratio": _QUERY_RATIO_MAP.get(
            priority, _QUERY_RATIO_MAP[LANGUAGE_PRIORITY_BILINGUAL_EQUAL]
        ),
        "detected_signals": signals,
    }

    logger.info(
        "Language routing: priority=%s confidence=%.2f signals=%s",
        priority, confidence, signals,
    )

    return result
