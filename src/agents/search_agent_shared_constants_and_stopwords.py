"""Shared constants for the search agent pipeline.

Why: Centralises stopwords and refusal phrases so they can be
reused across query generation, answer processing, and candidate
extraction without circular imports.
"""

# Chinese stopwords filtered from search queries to avoid noise
CHINESE_QUESTION_NOISE_STOPWORDS = {
    "问题", "答案", "哪些", "哪位", "哪年", "哪一年", "哪个", "哪个人", "哪里", "在哪",
    "是谁", "是什么", "如何", "为什么",
    "一个", "一种", "这个", "那个", "相关", "资料", "信息", "报告"
}

# Generic role words to reject as answers — we need actual names, not roles
GENERIC_ENTITY_ROLE_WORDS_TO_REJECT = {
    "author", "director", "writer", "teacher", "guru", "composer", "person",
    "individual", "figure", "leader", "founder", "host", "presenter", "editor",
    "publisher", "singer", "artist", "scientist", "scholar", "researcher",
}

# LLM refusal phrases — if the answer contains these, treat as no-answer
LLM_REFUSAL_AND_INSUFFICIENT_EVIDENCE_PHRASES = (
    "cannot be determined", "does not contain", "cannot be determined",
    "total number of pages cannot", "based on the given evidence",
    "no specific information", "cannot be determined from",
    "not enough information", "insufficient evidence",
    "i cannot", "i'm unable", "i am unable",
    "does not provide", "not mentioned", "no evidence",
    "cannot find", "unable to determine", "not possible to determine",
)
