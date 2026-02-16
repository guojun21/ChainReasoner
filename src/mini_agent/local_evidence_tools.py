"""Local evidence browsing tools — scratchpad file/search tools for MiniAgent.

Why: These tools let the agentic LLM browse the per-question scratchpad
(evidence files, pages, INDEX.md) without network calls.  They are the
first tools the agent should use before resorting to web_search.

Tools defined here:
  1. ListFilesTool — list all evidence and page files
  2. GetIndexTool — read INDEX.md overview
  3. ReadFileTool — read a specific evidence file
  4. GrepEvidenceTool — regex search across all files
  5. SearchLocalTool — BM25 keyword search over stored evidence
"""

import logging
from typing import Any

from src.mini_agent.tool_base_and_registry import MiniAgentToolBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. ListFilesTool
# ---------------------------------------------------------------------------

class ListEvidenceAndPageFilesTool(MiniAgentToolBase):
    """List all evidence and page files in the scratchpad."""

    name = "list_files"
    description = "List all evidence and page files in the scratchpad."
    when_to_use = "You want to see what evidence files are available."
    when_not_to_use = "You already know the file names from INDEX.md."
    parameters_schema = {"type": "object", "properties": {}, "required": []}

    def __init__(self, scratchpad: Any) -> None:
        self._scratchpad = scratchpad

    def execute(self, **kwargs: Any) -> str:
        """List files with sizes from evidence/ and pages/ subdirectories."""
        files = []
        for subdir_name in ["evidence", "pages", "large_results"]:
            subdir = self._scratchpad.root / subdir_name
            if subdir.exists():
                for f in sorted(subdir.glob("*.*")):
                    size = f.stat().st_size
                    files.append(f"{subdir_name}/{f.name} ({size} bytes)")
        return "\n".join(files) if files else "(no files)"


# ---------------------------------------------------------------------------
# 2. GetIndexTool
# ---------------------------------------------------------------------------

class GetEvidenceIndexOverviewTool(MiniAgentToolBase):
    """Read the INDEX.md hop-by-hop entity summary."""

    name = "get_index"
    description = "Read the INDEX.md overview (hop-by-hop entity summary)."
    when_to_use = "You need an overview of all collected evidence and entities."
    when_not_to_use = "You already read the index (it was provided above)."
    parameters_schema = {"type": "object", "properties": {}, "required": []}

    def __init__(self, scratchpad: Any) -> None:
        self._scratchpad = scratchpad

    def execute(self, **kwargs: Any) -> str:
        """Return the full INDEX.md content."""
        return self._scratchpad.get_index() or "(INDEX.md is empty)"


# ---------------------------------------------------------------------------
# 3. ReadFileTool
# ---------------------------------------------------------------------------

class ReadEvidenceFileTool(MiniAgentToolBase):
    """Read a specific file from the scratchpad."""

    name = "read_file"
    description = "Read a specific file from the scratchpad."
    when_to_use = "You identified a specific evidence file that likely contains the answer."
    when_not_to_use = "You want to search broadly — use grep_evidence or search_local instead."
    parameters_schema = {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Relative path within scratchpad (e.g. evidence/hop1_abc123.md)",
            },
        },
        "required": ["filename"],
    }

    def __init__(self, scratchpad: Any) -> None:
        self._scratchpad = scratchpad

    def execute(self, filename: str = "", **kwargs: Any) -> str:
        """Read a file by relative path within the scratchpad."""
        if not filename:
            return "(read_file requires a 'filename' argument)"

        # Support reading from large_results/ directory too
        result = self._scratchpad.read_evidence(filename)
        if not result:
            # Try direct path under scratchpad root
            full_path = self._scratchpad.root / filename
            if full_path.exists():
                try:
                    result = full_path.read_text(encoding="utf-8")
                except Exception as exc:
                    return f"(error reading {filename}: {exc})"

        return result if result else f"(file not found: {filename})"


# ---------------------------------------------------------------------------
# 4. GrepEvidenceTool
# ---------------------------------------------------------------------------

class GrepEvidenceKeywordSearchTool(MiniAgentToolBase):
    """Keyword search across all evidence files using regex."""

    name = "grep_evidence"
    description = "Keyword search across all evidence files."
    when_to_use = "You know a specific entity name, date, or keyword to search for."
    when_not_to_use = "You want to browse broadly — use get_index or list_files instead."
    parameters_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for (case-insensitive)",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, scratchpad: Any) -> None:
        self._scratchpad = scratchpad

    def execute(self, pattern: str = "", **kwargs: Any) -> str:
        """Search all evidence and page files for the given pattern."""
        if not pattern:
            return "(grep_evidence requires a 'pattern' argument)"

        matches = self._scratchpad.grep_evidence(pattern, max_results=15)
        if matches:
            parts = []
            for m in matches:
                parts.append(f"[{m['file']}] ...{m['context']}...")
            return "\n".join(parts)
        return f"(no matches for pattern: {pattern})"


# ---------------------------------------------------------------------------
# 5. SearchLocalTool
# ---------------------------------------------------------------------------

class SearchLocalBm25EvidenceTool(MiniAgentToolBase):
    """BM25 keyword search over all stored evidence documents."""

    name = "search_local"
    description = "BM25 keyword search over all stored evidence."
    when_to_use = "You want to find evidence related to a topic or phrase."
    when_not_to_use = "You know the exact keyword — use grep_evidence for exact matches."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (keywords or phrase)",
            },
        },
        "required": ["query"],
    }

    def __init__(self, scratchpad: Any) -> None:
        self._scratchpad = scratchpad

    def execute(self, query: str = "", **kwargs: Any) -> str:
        """Search the BM25 index and return top results."""
        if not query:
            return "(search_local requires a 'query' argument)"

        hits = self._scratchpad.search_local(query, top_k=5, min_score=0.5)
        if hits:
            parts = []
            for h in hits:
                parts.append(
                    f"[score={h['_score']:.1f}] {h['title']}\n"
                    f"  URL: {h['url']}\n"
                    f"  {h['content'][:500]}"
                )
            return "\n\n".join(parts)
        return f"(no local results for: {query})"
