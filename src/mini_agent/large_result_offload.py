"""Large result offload middleware — prevents context explosion from big tool outputs.

Why: The old system hard-truncated tool results at 3000 chars, losing critical
evidence buried in the middle or end of long results.  This module (inspired by
deepagents filesystem.py _process_large_message) writes the full result to the
scratchpad and returns a head+tail preview with a file path, so the agent can
read specific sections via read_file if needed.

References:
  - deepagents: middleware/filesystem.py _process_large_message() (1054-1124)
  - deepagents: _create_content_preview() (376-402)
"""

import hashlib
import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOOL_RESULT_OFFLOAD_CHAR_THRESHOLD = 4000  # ~1000 tokens; above this, offload
PREVIEW_HEAD_LINES = 5
PREVIEW_TAIL_LINES = 5
PREVIEW_LINE_MAX_CHARS = 200

# Tools whose results are always small — skip offload check for efficiency
TOOLS_EXCLUDED_FROM_OFFLOAD = frozenset({
    "list_files",
    "get_index",
    "submit_answer",
})

OFFLOAD_NOTICE_TEMPLATE = """\
Tool result too large ({total_chars} chars), saved to scratchpad at: {file_path}

You can read the full result using: read_file(filename="{file_path}")
Or read specific sections with grep_evidence.

Preview (head + tail):
{preview}
"""


# ---------------------------------------------------------------------------
# Head + tail preview generator
# ---------------------------------------------------------------------------

def _create_head_tail_content_preview(
    content: str,
    head_lines: int = PREVIEW_HEAD_LINES,
    tail_lines: int = PREVIEW_TAIL_LINES,
    line_max_chars: int = PREVIEW_LINE_MAX_CHARS,
) -> str:
    """Create a preview showing the first and last N lines of content.

    Why (from deepagents _create_content_preview): Showing both head and tail
    gives the agent enough context to decide whether to read the full file,
    without consuming excessive context window tokens.
    """
    lines = content.splitlines()

    if len(lines) <= head_lines + tail_lines:
        # Short enough to show in full
        return "\n".join(line[:line_max_chars] for line in lines)

    head = [line[:line_max_chars] for line in lines[:head_lines]]
    tail = [line[:line_max_chars] for line in lines[-tail_lines:]]
    truncated_count = len(lines) - head_lines - tail_lines

    return (
        "\n".join(head)
        + f"\n... [{truncated_count} lines truncated] ...\n"
        + "\n".join(tail)
    )


# ---------------------------------------------------------------------------
# Main offload function
# ---------------------------------------------------------------------------

def offload_large_tool_result_to_scratchpad_file(
    tool_name: str,
    tool_result: str,
    scratchpad: Any,
) -> str:
    """If tool result exceeds threshold, write to scratchpad and return preview.

    Why: Hard truncation at 3000 chars loses evidence that may contain the
    correct answer.  Writing the full result to a file and returning a preview
    preserves all evidence while keeping the context window manageable.

    Args:
        tool_name: Name of the tool that produced the result.
        tool_result: The full tool result string.
        scratchpad: PerQuestionEvidenceScratchpad instance.

    Returns:
        The original result if under threshold, or the offload notice with
        preview if the result was offloaded to a file.
    """
    processed_result, _, _ = offload_large_tool_result_to_scratchpad_file_with_metadata(
        tool_name=tool_name,
        tool_result=tool_result,
        scratchpad=scratchpad,
    )
    return processed_result


def offload_large_tool_result_to_scratchpad_file_with_metadata(
    tool_name: str,
    tool_result: str,
    scratchpad: Any,
) -> Tuple[str, bool, str]:
    """Offload large result and return (processed_result, was_offloaded, file_path).

    Why: MiniAgent iteration logging needs the real offload file path for
    post-mortem analysis.  The legacy function returns only text, so this helper
    exposes metadata without breaking existing callers.
    """
    if not tool_result or len(tool_result) <= TOOL_RESULT_OFFLOAD_CHAR_THRESHOLD:
        return tool_result, False, ""

    if tool_name in TOOLS_EXCLUDED_FROM_OFFLOAD:
        return tool_result, False, ""

    # Generate a unique filename
    content_hash = hashlib.md5(tool_result[:500].encode(errors="replace")).hexdigest()[:6]
    filename = f"large_results/{tool_name}_{content_hash}.txt"

    try:
        # Ensure the large_results directory exists
        large_results_dir = scratchpad.root / "large_results"
        large_results_dir.mkdir(parents=True, exist_ok=True)

        # Write the full result
        file_path = large_results_dir / f"{tool_name}_{content_hash}.txt"
        file_path.write_text(tool_result, encoding="utf-8")

        # Generate preview
        preview = _create_head_tail_content_preview(tool_result)

        logger.info(
            "Offloaded large tool result: tool=%s chars=%d -> %s",
            tool_name, len(tool_result), filename,
        )

        processed_result = OFFLOAD_NOTICE_TEMPLATE.format(
            total_chars=len(tool_result),
            file_path=filename,
            preview=preview,
        )
        return processed_result, True, filename

    except Exception as exc:
        logger.warning("Failed to offload large result for %s: %s", tool_name, exc)
        # Fallback: truncate with notice (better than crashing)
        truncated = tool_result[:TOOL_RESULT_OFFLOAD_CHAR_THRESHOLD]
        processed_result = (
            truncated
            + f"\n... (truncated from {len(tool_result)} chars, offload failed: {exc})"
        )
        return processed_result, False, ""
