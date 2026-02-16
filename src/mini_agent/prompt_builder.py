"""MiniAgent system prompt builder — dynamically generates prompts from tool registry.

Why: The old system hardcoded tool definitions in a giant string constant.
This module (inspired by nanobot context.py build_system_prompt) generates
the system prompt dynamically from the tool registry, ensuring docs stay
in sync with actual tool implementations.

Architecture: "Technical Co-Founder" phased framework (Phase 1-4) replaces
the old flat "Role + Strategy + Rules" layout.  Each phase has a clear
objective, decision criteria, and escalation trigger so the LLM follows a
structured workflow instead of ad-hoc instruction hopping.

References:
  - nanobot: agent/context.py build_system_prompt() — composable prompt sections
  - OpenManus: planning.py — SYSTEM + NEXT_STEP phased prompts
  - Research_Agent: tool_routing.yaml — three-layer routing decision
  - "Technical Co-Founder" framework — phased project execution with rules
"""

from typing import Any, Dict, List, Optional, Tuple

from src.mini_agent.tool_base_and_registry import MiniAgentToolRegistry

# ---------------------------------------------------------------------------
# Prompt sections — composable building blocks (Phase 1-4 framework)
# ---------------------------------------------------------------------------

_ROLE_SECTION = """\
# Role

You are an Evidence Verification & Answer Refinement Specialist.

A multi-hop search pipeline has already collected evidence for a complex \
question and proposed candidate answers. Your job is to:
1. Review the collected evidence with a critical eye
2. Verify or correct the proposed answer using evidence
3. Submit a precise, factual final answer

You are the LAST line of defense before the answer is submitted. \
Treat every candidate answer with healthy skepticism — verify it, \
don't just rubber-stamp it."""

_PHASE_1_DISCOVERY = """\

# Phase 1: Discovery (Evidence Reconnaissance)

**Objective**: Understand what evidence exists and what is missing.

Actions:
- Scan the INDEX.md to see all collected evidence files and their hop summaries
- Identify which question constraints are covered by evidence and which are NOT
- Assess the pipeline's candidate answers: do they look plausible given the evidence?

**Decision Gate** (move to Phase 2 when):
- You have a mental map of available evidence
- You know which candidate answer(s) to verify first

**Escalation** (skip to Phase 3 if):
- INDEX.md shows < 3 evidence files total
- All evidence files are < 200 chars combined"""

_PHASE_2_VERIFICATION = """\

# Phase 2: Verification (Cross-Check Candidates)

**Objective**: Verify the top candidate answer against collected evidence.

Actions:
- grep_evidence for the key entity name from the proposed answer
- read_file on the most relevant evidence file(s)
- Cross-check: does the evidence DIRECTLY support the proposed answer?
- Look for CONTRADICTIONS: does any evidence point to a DIFFERENT entity?

**Quantitative Triggers**:
- grep_evidence returns < 200 chars -> evidence is WEAK -> escalate to Phase 3
- read_file returns < 100 chars -> evidence is WEAK -> escalate to Phase 3
- First 2 local tool calls return < 300 chars total -> evidence is INSUFFICIENT -> escalate to Phase 3
- Evidence mentions multiple candidate entities -> you MUST verify EACH before proceeding
- Evidence shows ALL candidates FAIL a constraint (e.g. wrong birth year, wrong nationality) -> \
ALL candidates are REJECTED -> you MUST escalate to Phase 3 to find NEW candidates via web_search

**Candidate Rejection Detection**:
If evidence contains phrases like "does not meet", "fails", "outside the range", \
"born in [year] which is outside", "not between", or similar negations for ALL \
proposed candidates, then ALL candidates are WRONG. Do NOT submit "Unknown" or \
"None of the candidates satisfy all constraints". Instead, IMMEDIATELY escalate \
to Phase 3 and search for ALTERNATIVE candidates using the rejection constraints.

**Decision Gate** (move to Phase 4 when):
- Evidence clearly and directly confirms one answer
- No contradictions remain

**Escalation** (move to Phase 3 when):
- Local evidence is insufficient, contradictory, or ambiguous
- ALL proposed candidates are rejected by evidence (MANDATORY escalation)"""

_PHASE_3_INVESTIGATION = """\

# Phase 3: Investigation (External Search — ONLY when needed)

**Objective**: Fill evidence gaps that local files cannot resolve.

Actions:
- Use web_search with SPECIFIC entity names (NOT abstract concepts)
  GOOD: "RepRapPro Limited company ceased trading"
  BAD: "cellular automata open hardware project"
- Use fetch_page when you found a promising URL in search results
- After EACH search, immediately evaluate: is this enough to decide?

**Candidate Rejection Recovery** (CRITICAL):
If Phase 2 revealed that ALL pipeline candidates are WRONG (e.g. all fail a birth year, \
nationality, or tenure constraint), you MUST:
1. Identify the specific constraints that rejected candidates failed
2. Construct search queries that INCLUDE those constraints and EXCLUDE rejected names
   GOOD: "first African Premier League player born 1988-1995 Taurus -Ndlovu -Fortune"
   BAD: "Peter Ndlovu birth year" (searching for a known-wrong candidate)
3. Search for the CORRECT candidate, not more info about wrong ones

**Efficiency Rules**:
- Search for the ENTITY NAME, not the question topic
- If the pipeline already has a candidate answer, search for THAT entity first
- Maximum 4 web_search calls — if 4 searches don't resolve it, decide with what you have
- Do NOT search for background information you already know — focus on the SPECIFIC answer
- NEVER submit "Unknown" or "None of the candidates" if you still have web_search calls remaining

**Decision Gate** (move to Phase 4 when):
- New evidence confirms or refutes the candidate answer
- You have exhausted reasonable search strategies"""

_PHASE_4_DECISION = """\

# Phase 4: Decision (Submit Final Answer)

**Objective**: Make a final decision and submit.

**Self-Check Checklist** (verify ALL before submitting):
1. Does your answer satisfy EVERY constraint mentioned in the question?
2. Have you considered alternative candidates? Could a different entity be correct?
3. Is there contradictory evidence you haven't addressed?
4. Is the entity name EXACT and COMPLETE?
   - Project name vs company name (e.g. "RepRap" vs "RepRapPro Ltd") are DIFFERENT
   - Legal/registered name vs informal name — always prefer the formal name from evidence
5. Is the answer format correct? (name, number, date — NOT a sentence or explanation)

**Then**: Call submit_answer with your final answer and confidence score."""

_RULES_SECTION = """\

# Rules (Hard Constraints — NEVER violate)

## Response Format
- Your ENTIRE response must be ONLY a single <tool_call> tag. No text before or after.
  WRONG: "Let me search for..." <tool_call>...</tool_call>
  CORRECT: <tool_call>{"name": "grep_evidence", "args": {"pattern": "Ford"}}</tool_call>

## Answer Requirements
- You MUST call submit_answer as your FINAL action. No exceptions.
- Answer must be a precise, short factual answer (name, number, date), NOT a sentence.
- When the question asks for a company/entity name, verify the EXACT legal name from evidence.

## Tool Priority
- ALWAYS try local evidence tools first (grep_evidence, read_file, search_local)
- ONLY use network tools (web_search, fetch_page) after local evidence proves insufficient
- Do NOT read files endlessly. After 2-3 evidence files, decide or search externally.

## How to Work with Me
- Treat the pipeline's proposed answer as a HYPOTHESIS, not a fact
- I make the decisions based on evidence; you verify them
- Be honest about uncertainty — a confident wrong answer is worse than admitting ambiguity
- Move fast: each tool call costs time. Don't repeat searches with similar queries."""


# ---------------------------------------------------------------------------
# Main builder function
# ---------------------------------------------------------------------------

def build_mini_agent_system_prompt(
    registry: MiniAgentToolRegistry,
    question: str,
    pipeline_answer: str,
    pipeline_candidates: Dict[str, str],
    index_content: str,
    max_iterations: int = 12,
    hop_evaluations: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, str]:
    """Build (system_prompt, user_prompt) from registry + question context.

    Why: Dynamic prompt generation from the registry means adding a new tool
    automatically updates the system prompt — no manual string editing needed.

    Architecture: Assembles the Phase 1-4 framework sections with dynamically
    generated tool definitions from the registry.  The phased structure guides
    the LLM through Discovery -> Verification -> Investigation -> Decision,
    matching the "Technical Co-Founder" project execution pattern.

    Args:
        registry: Tool registry with all registered tools.
        question: The original question.
        pipeline_answer: The answer from the pipeline's voting phase.
        pipeline_candidates: Dict of candidate answers (source -> answer).
        index_content: Content of INDEX.md from scratchpad.
        max_iterations: Maximum loop iterations (shown to LLM as countdown).
        hop_evaluations: Optional list of per-hop evaluation dicts with
            keys: hop_num, entity, valid, confidence.  Injected into the
            user prompt so the agent knows the pipeline's self-assessment.

    Returns:
        (system_prompt, user_prompt) tuple.
    """
    # Generate tool definitions dynamically from registry
    tool_definitions = registry.generate_tool_definitions_for_prompt()

    system_prompt = (
        _ROLE_SECTION + "\n\n"
        + tool_definitions + "\n"
        + _PHASE_1_DISCOVERY + "\n"
        + _PHASE_2_VERIFICATION + "\n"
        + _PHASE_3_INVESTIGATION + "\n"
        + _PHASE_4_DECISION + "\n"
        + _RULES_SECTION
    )

    # Build user prompt with question context
    candidates_text = "\n".join(
        f"  - {source}: {answer}"
        for source, answer in pipeline_candidates.items()
        if answer
    )

    # Detect if all candidates agree — warn about confirmation bias
    unique_answers = set(
        v.strip() for v in pipeline_candidates.values()
        if v and v.strip() and v.strip().lower() not in ("unknown", "")
    )
    all_agree_warning = ""
    if len(unique_answers) <= 1 and unique_answers:
        all_agree_warning = (
            "\nWARNING: All pipeline candidates agree on the same answer. "
            "This could indicate confirmation bias. You SHOULD still verify "
            "this answer against evidence (Phase 2) and check if an alternative "
            "entity better matches ALL constraints in the question.\n"
        )

    # Detect candidate disagreement — highlight for focused verification
    disagreement_note = ""
    if len(unique_answers) >= 3:
        disagreement_note = (
            "\nNOTE: Candidates show significant DISAGREEMENT. "
            "Pay extra attention in Phase 2 to verify each distinct candidate "
            "against evidence before deciding.\n"
        )

    # Build hop evaluation summary — alerts the agent when the pipeline's own
    # validation flagged low confidence or invalid results.
    # Why: In Q20, the agent submitted with 0.95 confidence despite Hop 4
    # evaluation returning valid=False, confidence=0.3.  Injecting these
    # signals prevents the agent from rubber-stamping a bad pipeline answer.
    hop_eval_section = ""
    if hop_evaluations:
        hop_eval_lines = []
        any_invalid = False
        for he in hop_evaluations:
            valid = he.get("valid", True)
            conf = he.get("confidence", 1.0)
            entity = he.get("entity", "?")
            hop_n = he.get("hop_num", "?")
            flag = " ⚠️ LOW CONFIDENCE" if conf <= 0.5 or not valid else ""
            hop_eval_lines.append(
                f"  - Hop {hop_n}: entity={entity}, "
                f"valid={valid}, confidence={conf:.2f}{flag}"
            )
            if not valid or conf <= 0.3:
                any_invalid = True
        hop_eval_text = "\n".join(hop_eval_lines)
        hop_eval_section = f"## Pipeline hop evaluations\n{hop_eval_text}\n\n"
        if any_invalid:
            hop_eval_section += (
                "⚠️ WARNING: One or more hops have valid=False or confidence<=0.3. "
                "The pipeline's proposed answer is LIKELY WRONG. You MUST search "
                "for alternative candidates before submitting.\n\n"
            )

    user_prompt = (
        f"## Question\n{question}\n\n"
        f"## Pipeline's proposed answer\n{pipeline_answer}\n\n"
        f"## All candidate answers\n{candidates_text}\n\n"
        f"{all_agree_warning}"
        f"{disagreement_note}"
        f"{hop_eval_section}"
        f"## Evidence index (INDEX.md)\n{index_content}\n\n"
        f"Begin Phase 1: Scan the evidence index above, then proceed through "
        f"the phases. Start by grepping for the key entity from the proposed answer.\n\n"
        f"[Tool calls remaining: {max_iterations}]"
    )

    return system_prompt, user_prompt
