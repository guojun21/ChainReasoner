"""Multi-protocol tool-call parser — XML + JSON block + free-text triple fallback.

Why: The current XML-only parser (<tool_call>...</tool_call>) fails when qwen
outputs malformed XML, uses markdown JSON blocks, or writes free-text function
calls.  This module tries three protocols in order, with JSON repair for common
qwen formatting errors (trailing commas, single quotes, unescaped newlines).

References:
  - nanobot: uses OpenAI Function Calling (most reliable but requires FC support)
  - Current ChainReasoner: XML tag protocol only
"""

import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol 1: XML tag — <tool_call>{"name": "...", "args": {...}}</tool_call>
# ---------------------------------------------------------------------------

_XML_TAG_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def _try_parse_xml_tag_protocol(text: str) -> Optional[Dict[str, Any]]:
    """Parse <tool_call>{...}</tool_call> format.

    Why: This is the primary protocol used in the system prompt.
    Most LLM outputs follow it when instructed clearly.
    """
    match = _XML_TAG_PATTERN.search(text)
    if not match:
        return None
    return _try_parse_json_payload(match.group(1), protocol="xml_tag")


# ---------------------------------------------------------------------------
# Protocol 2: JSON code block — ```json\n{...}\n``` or ```\n{...}\n```
# ---------------------------------------------------------------------------

_JSON_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*\n?\s*(\{.*?\})\s*\n?\s*```",
    re.DOTALL,
)


def _try_parse_json_code_block_protocol(text: str) -> Optional[Dict[str, Any]]:
    """Parse markdown JSON code block containing a tool call.

    Why: Some models wrap their JSON output in markdown code fences
    instead of XML tags, especially when they've seen markdown in training.
    """
    match = _JSON_BLOCK_PATTERN.search(text)
    if not match:
        return None
    return _try_parse_json_payload(match.group(1), protocol="json_block")


# ---------------------------------------------------------------------------
# Protocol 3: Free-text function call — tool_name(arg1="val1", ...)
# or bare JSON object anywhere in text
# ---------------------------------------------------------------------------

_BARE_JSON_PATTERN = re.compile(
    r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"args"\s*:\s*(\{.*?\})\s*\}',
    re.DOTALL,
)


def _try_parse_freeform_function_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract tool call from bare JSON or free-text patterns.

    Why: As a last resort, if the LLM outputs a JSON object with "name" and
    "args" keys without any wrapper, we should still be able to parse it.
    This catches cases where the LLM forgets the XML tags entirely.
    """
    match = _BARE_JSON_PATTERN.search(text)
    if not match:
        return None
    name = match.group(1)
    args_str = match.group(2)
    try:
        args = json.loads(_repair_malformed_json_from_llm(args_str))
        return {"name": name, "args": args if isinstance(args, dict) else {}}
    except (json.JSONDecodeError, ValueError):
        # Try with the full match as a single JSON object
        full_str = match.group(0)
        return _try_parse_json_payload(full_str, protocol="freeform")


# ---------------------------------------------------------------------------
# JSON repair — handles common qwen formatting errors
# ---------------------------------------------------------------------------

def _repair_malformed_json_from_llm(raw_json: str) -> str:
    """Attempt to fix common JSON errors from LLM output.

    Why: qwen and other models frequently produce JSON with trailing commas,
    single quotes instead of double quotes, unescaped newlines in strings,
    and missing closing braces.  Fixing these before json.loads() dramatically
    improves parse success rate.
    """
    text = raw_json.strip()

    # Fix 1: Replace single quotes with double quotes (but not inside strings)
    # Simple heuristic: if there are no double quotes at all, replace singles
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')

    # Fix 2: Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Fix 3: Remove unescaped newlines inside string values
    # Replace literal newlines that appear within JSON string values with \\n
    # We use a simple approach: replace all bare newlines with spaces,
    # since JSON strings should not contain literal newlines.
    text = text.replace("\n", " ").replace("\r", " ")

    # Fix 4: Ensure the string ends with }
    if text and text[-1] != "}" and "{" in text:
        # Count braces
        open_count = text.count("{")
        close_count = text.count("}")
        if open_count > close_count:
            text += "}" * (open_count - close_count)

    return text


# ---------------------------------------------------------------------------
# Shared JSON payload parser
# ---------------------------------------------------------------------------

def _try_parse_json_payload(
    raw_json: str,
    protocol: str = "unknown",
) -> Optional[Dict[str, Any]]:
    """Parse a JSON string into a tool call dict, with repair fallback.

    Returns dict with 'name' and 'args' keys, or None if parsing fails.
    """
    # Try direct parse first
    for attempt_json in [raw_json, _repair_malformed_json_from_llm(raw_json)]:
        try:
            payload = json.loads(attempt_json)
            if isinstance(payload, dict) and "name" in payload:
                result = {
                    "name": str(payload["name"]),
                    "args": payload.get("args", {}),
                    "_protocol": protocol,
                }
                if not isinstance(result["args"], dict):
                    result["args"] = {}
                return result
        except (json.JSONDecodeError, ValueError):
            continue

    logger.debug("Failed to parse tool call JSON (protocol=%s): %s", protocol, raw_json[:200])
    return None


# ---------------------------------------------------------------------------
# Main entry point — triple fallback parser
# ---------------------------------------------------------------------------

def parse_tool_call_from_llm_response(text: str) -> Optional[Dict[str, Any]]:
    """Try three protocols in order: XML tag > JSON block > free-text extraction.

    Why: Different LLMs (and even the same LLM across turns) may use different
    output formats.  Triple fallback maximises parse success without requiring
    OpenAI Function Calling support.

    Returns:
        Dict with 'name', 'args', and '_protocol' keys, or None if no valid
        tool call found in the text.
    """
    if not text or not text.strip():
        return None

    # Protocol 1: XML tag (primary — matches system prompt instructions)
    result = _try_parse_xml_tag_protocol(text)
    if result:
        return result

    # Protocol 2: JSON code block (common fallback for markdown-trained models)
    result = _try_parse_json_code_block_protocol(text)
    if result:
        return result

    # Protocol 3: Bare JSON / free-text (last resort)
    result = _try_parse_freeform_function_call(text)
    if result:
        return result

    return None
