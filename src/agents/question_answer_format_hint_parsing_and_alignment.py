"""Question-level answer format hint parsing and conservative alignment helpers.

Why: Competition scoring is exact-match. Many answers are semantically correct
but lose points due to tiny format mismatches. This module extracts format hints
from question text (e.g. "要求格式形如：Alibaba Group Limited") and provides
shared utilities for prompt constraints and lightweight post-processing.
"""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ParsedQuestionAnswerFormatHint:
    """Structured format constraints parsed from a question."""

    has_format_hint: bool
    format_example: str
    expected_language: str
    expected_answer_style: str


def _detect_if_text_is_primarily_chinese_characters(text: str) -> bool:
    """Return True when CJK characters dominate visible text.

    Why: We use this to choose language-specific connector normalization rules.
    """
    if not text:
        return False
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_count = len(re.findall(r"[A-Za-z]", text))
    return cjk_count > latin_count


def _infer_expected_answer_style_from_format_example(format_example: str) -> str:
    """Infer expected answer style from a format example snippet."""
    if not format_example:
        return "generic_short_phrase"
    if re.search(r"\b(?:ltd|limited|inc|corp|company|group)\b", format_example, re.I):
        return "english_company_name"
    if re.search(r"[\u4e00-\u9fff]+\s*和\s*[\u4e00-\u9fff]+", format_example):
        return "chinese_dual_entity_with_he_connector"
    if re.fullmatch(r"\d{4}", format_example.strip()):
        return "year_only"
    if re.fullmatch(r"\d+", format_example.strip()):
        return "number_only"
    return "generic_short_phrase"


def extract_structured_answer_format_hint_from_question_text(
    question_text: str,
) -> ParsedQuestionAnswerFormatHint:
    """Parse answer format hint from question text.

    Recognised examples:
    - 要求格式形如：Alibaba Group Limited
    - 回答格式形如：张三和李四
    """
    if not question_text:
        return ParsedQuestionAnswerFormatHint(
            has_format_hint=False,
            format_example="",
            expected_language="unknown",
            expected_answer_style="generic_short_phrase",
        )

    # Capture everything after "格式形如：" up to sentence boundary.
    match = re.search(
        r"(?:要求|回答)?\s*格式(?:形如|为)?\s*[：:]\s*([^\n。；;!?！？]+)",
        question_text,
        re.IGNORECASE,
    )
    format_example = match.group(1).strip() if match else ""
    format_example = format_example.strip("\"'“”‘’` ")

    if format_example:
        expected_language = "zh" if _detect_if_text_is_primarily_chinese_characters(format_example) else "en"
    else:
        expected_language = "zh" if _detect_if_text_is_primarily_chinese_characters(question_text) else "en"

    return ParsedQuestionAnswerFormatHint(
        has_format_hint=bool(format_example),
        format_example=format_example,
        expected_language=expected_language,
        expected_answer_style=_infer_expected_answer_style_from_format_example(format_example),
    )


def build_format_sensitive_answer_constraints_for_prompt(question_text: str) -> str:
    """Build an answer-format instruction block for LLM extraction prompts."""
    parsed_hint = extract_structured_answer_format_hint_from_question_text(question_text)
    if not parsed_hint.has_format_hint:
        return (
            "FORMAT CONSTRAINTS:\n"
            "- Return only the final answer text, with no explanation.\n"
            "- Keep the answer as short as possible while preserving correctness.\n"
        )
    return (
        "FORMAT CONSTRAINTS:\n"
        f"- The question gives a format example: {parsed_hint.format_example}\n"
        "- This shows the GENERAL TYPE of answer expected (e.g. a company name with legal suffix).\n"
        "- CRITICAL: Use the EXACT entity name and suffix as found in the web search evidence.\n"
        "- If the preliminary answer says 'Limited' but evidence says 'Ltd', use 'Ltd'.\n"
        "- Do NOT change abbreviations: 'Ltd' stays 'Ltd', 'Limited' stays 'Limited'.\n"
        "- Output only the answer itself; do not include prefixes like 'Answer:'.\n"
    )


def _strip_outer_quotes_and_terminal_punctuation(answer_text: str) -> str:
    """Remove outer quotes and trailing punctuation safely."""
    if not answer_text:
        return ""
    cleaned_text = answer_text.strip().strip("\"'“”‘’` ")
    cleaned_text = re.sub(r"[。．\.!?！？;；:,，\s]+$", "", cleaned_text)
    return cleaned_text.strip()


def align_answer_text_to_format_hint_with_conservative_rules(
    answer_text: str, question_text: str
) -> str:
    """Apply conservative format alignment rules based on question hint.

    Why: We only do low-risk formatting transformations to avoid semantic drift.
    """
    cleaned_text = _strip_outer_quotes_and_terminal_punctuation(answer_text)
    if not cleaned_text:
        return ""

    parsed_hint = extract_structured_answer_format_hint_from_question_text(question_text)
    if not parsed_hint.has_format_hint:
        return cleaned_text

    if parsed_hint.expected_answer_style == "chinese_dual_entity_with_he_connector":
        split_parts = [
            split_item.strip()
            for split_item in re.split(r"\s*(?:、|,|，|/|\\|\band\b|与|及|和)\s*", cleaned_text, flags=re.I)
            if split_item.strip()
        ]
        if len(split_parts) >= 2:
            return f"{split_parts[0]}和{split_parts[1]}"
    if parsed_hint.expected_answer_style in {"year_only", "number_only"}:
        number_match = re.search(r"\d{4}" if parsed_hint.expected_answer_style == "year_only" else r"\d+", cleaned_text)
        if number_match:
            return number_match.group(0)
    return cleaned_text


# ── Format-aware answer post-processing pipeline ───────────────────────
#    Inspired by Research_Agent normalize.py and ReAct wrappers.py.
#    Fixes common LLM output issues that break exact-match scoring.

_LLM_ANSWER_PREFIX_PATTERNS_EN = [
    r"^(?:the\s+)?answer\s*(?:is|:)\s*",
    r"^based\s+on\s+(?:the\s+)?(?:evidence|information|search\s+results|provided\s+information)[,:]?\s*(?:the\s+answer\s*(?:is|:)\s*)?",
    r"^according\s+to\s+(?:the\s+)?(?:evidence|sources?|search\s+results)[,:]?\s*(?:the\s+answer\s*(?:is|:)\s*)?",
    r"^from\s+(?:the\s+)?evidence[,:]?\s*",
    r"^(?:the\s+)?(?:english\s+)?name\s+(?:of\s+.+?\s+)?is\s+",
    r"^(?:the\s+)?(?:company|organization|entity|person|author|podcast|film|book|paper)\s*(?:'s\s+)?(?:name\s+)?is\s+",
    r"^it\s+is\s+(?:called\s+)?",
]

_LLM_ANSWER_PREFIX_PATTERNS_ZH = [
    r"^答案[：:]\s*",
    r"^(?:该|这个|这)\s*(?:商业实体|公司|组织|机构|人物|作品|纪念碑|设备)\s*的?\s*(?:英文|中文)?(?:名称?|名字)\s*(?:是|为|叫)[：:]*\s*",
    r"^根据(?:证据|资料|搜索结果|信息)[，,：:]*\s*(?:答案\s*(?:是|为)[：:]*\s*)?",
]


def _strip_all_known_llm_answer_prefixes(answer: str) -> str:
    """Remove all known LLM answer prefix patterns.

    Why: LLMs often prepend 'The answer is', 'Based on evidence,' etc.
    before the actual answer, which breaks exact-match scoring.
    """
    cleaned = answer
    for pattern in _LLM_ANSWER_PREFIX_PATTERNS_EN + _LLM_ANSWER_PREFIX_PATTERNS_ZH:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def _extract_core_answer_from_sentence_wrapper(answer: str) -> str:
    """Extract core answer from sentence wrappers.

    Why: LLMs sometimes wrap the answer in a sentence like
    'The company is called RepRapPro Ltd.' — we need just 'RepRapPro Ltd'.
    """
    if len(answer) > 200:
        return answer

    sentence_patterns = [
        r"^(?:the\s+)?(?:company|entity|organization|institution|person)\s+(?:is\s+(?:called\s+)?|named\s+)(.+?)\.?$",
        r"^(?:it\s+is\s+(?:called\s+)?)(.+?)\.?$",
        r"^(?:this\s+is\s+)(.+?)\.?$",
    ]
    for pattern in sentence_patterns:
        match = re.match(pattern, answer, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            if extracted:
                return extracted
    return answer


def _normalize_numeric_answer_if_purely_numeric(answer: str) -> str:
    """Normalize pure numeric answers: remove commas, handle decimals.

    Why: Research_Agent normalize.py does this for numeric answers.
    Only activates when the entire answer looks like a number.
    """
    if re.fullmatch(r"\s*[\d,.\s]+\s*", answer):
        num_match = re.search(r"(\d[\d,]*\.?\d*)", answer)
        if num_match:
            num_str = num_match.group(1).replace(",", "")
            try:
                if "." in num_str:
                    num_val = float(num_str)
                    if num_val == int(num_val):
                        return str(int(num_val))
                    return num_str
                return num_str
            except (ValueError, OverflowError):
                pass
    return answer


def apply_format_aware_answer_postprocessing_pipeline(
    raw_answer: str,
    question_text: str,
) -> tuple:
    """Apply comprehensive format-aware post-processing to a raw answer.

    Why: Competition scoring is exact-match. This pipeline fixes common
    formatting issues: LLM preambles, stray quotes, numeric formatting,
    and format-hint alignment. Inspired by Research_Agent normalize.py
    and ReAct wrappers.py, adapted for our scoring requirements.

    Returns (cleaned_answer, trace_dict) for debugging.
    """
    trace = {"raw_answer": raw_answer}

    if not raw_answer or not raw_answer.strip():
        trace["final_answer"] = ""
        return "", trace

    answer = raw_answer.strip()

    # Step 1: Strip LLM answer prefixes
    answer = _strip_all_known_llm_answer_prefixes(answer)
    trace["after_prefix_strip"] = answer

    # Step 2: Remove outer quotes and terminal punctuation
    answer = _strip_outer_quotes_and_terminal_punctuation(answer)
    trace["after_quote_strip"] = answer

    # Step 3: Normalize whitespace
    answer = " ".join(answer.split())
    trace["after_whitespace"] = answer

    # Step 4: Extract core from sentence wrappers
    answer = _extract_core_answer_from_sentence_wrapper(answer)
    trace["after_sentence_unwrap"] = answer

    # Step 5: Numeric normalization (only for pure numeric answers)
    answer = _normalize_numeric_answer_if_purely_numeric(answer)
    trace["after_number_normalize"] = answer

    # Step 6: Format-hint-aware alignment (conservative rules)
    answer = align_answer_text_to_format_hint_with_conservative_rules(
        answer, question_text
    )
    trace["after_format_alignment"] = answer

    # Step 7: Final cleanup
    answer = answer.strip()
    trace["final_answer"] = answer

    # Record format hint info in trace
    parsed = extract_structured_answer_format_hint_from_question_text(
        question_text
    )
    trace["format_hint"] = {
        "has_format_hint": parsed.has_format_hint,
        "format_example": parsed.format_example,
        "expected_language": parsed.expected_language,
        "expected_answer_style": parsed.expected_answer_style,
    }

    return answer, trace
