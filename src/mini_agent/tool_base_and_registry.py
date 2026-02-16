"""Tool base class and registry — nanobot-style tool management for MiniAgent.

Why: The old monolithic _execute_tool() function used a giant if/elif chain.
This module provides an abstract base class and a registry (inspired by
nanobot tools/base.py + tools/registry.py) so each tool is a self-contained
class with its own validation, execution, and prompt documentation.

References:
  - nanobot: agent/tools/base.py (Tool ABC with validate_params)
  - nanobot: agent/tools/registry.py (ToolRegistry with execute/get_definitions)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class MiniAgentToolBase(ABC):
    """Abstract base class for all MiniAgent evidence-review tools.

    Why: Each tool encapsulates its own name, description, parameter schema,
    usage guidance, and execution logic — making the system extensible without
    touching the main loop.
    """

    name: str = ""
    description: str = ""
    parameters_schema: Dict[str, Any] = {}
    when_to_use: str = ""
    when_not_to_use: str = ""
    is_terminal: bool = False  # True for submit_answer — terminates the loop

    def validate_parameters(self, params: Dict[str, Any]) -> List[str]:
        """Validate tool parameters against the declared schema.

        Why (from nanobot tools/base.py): Catching bad params before execution
        prevents cryptic runtime errors and gives the LLM a clear error message
        so it can self-correct on the next iteration.

        Returns list of error strings (empty if valid).
        """
        errors: List[str] = []
        schema = self.parameters_schema
        if not schema:
            return errors

        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for key in required:
            if key not in params:
                errors.append(f"missing required parameter: {key}")

        for key, value in params.items():
            if key in properties:
                prop_schema = properties[key]
                expected_type = prop_schema.get("type", "")
                if expected_type == "string" and not isinstance(value, str):
                    errors.append(f"parameter '{key}' must be a string, got {type(value).__name__}")
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"parameter '{key}' must be a number, got {type(value).__name__}")

        return errors

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Execute the tool and return a result string.

        Subclasses must implement this.  The return value is shown to the LLM
        as the tool result in the next conversation turn.
        """
        ...

    def generate_prompt_definition(self) -> str:
        """Generate the tool definition text for inclusion in the system prompt.

        Why: Instead of hardcoding tool docs in a giant string, each tool
        generates its own documentation block — keeping docs in sync with code.
        """
        lines = [
            f"- **{self.name}**",
            f"  {self.description}",
        ]
        if self.when_to_use:
            lines.append(f"  WHEN TO USE: {self.when_to_use}")
        if self.when_not_to_use:
            lines.append(f"  WHEN NOT TO USE: {self.when_not_to_use}")

        # Parameter documentation
        props = self.parameters_schema.get("properties", {})
        if props:
            param_parts = []
            for pname, pschema in props.items():
                ptype = pschema.get("type", "any")
                pdesc = pschema.get("description", "")
                param_parts.append(f"{pname}: {ptype}" + (f" — {pdesc}" if pdesc else ""))
            lines.append(f"  Parameters: {', '.join(param_parts)}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class MiniAgentToolRegistry:
    """Registry for managing and executing MiniAgent tools.

    Why (from nanobot tools/registry.py): A central registry provides
    uniform tool discovery, validation, execution, and prompt generation
    without the main loop needing to know about individual tools.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, MiniAgentToolBase] = {}

    def register_tool(self, tool: MiniAgentToolBase) -> None:
        """Register a tool instance by its name."""
        if not tool.name:
            raise ValueError(f"Tool {type(tool).__name__} has no name")
        self._tools[tool.name] = tool
        logger.debug("Registered MiniAgent tool: %s", tool.name)

    def get_tool(self, name: str) -> Optional[MiniAgentToolBase]:
        """Look up a tool by name."""
        return self._tools.get(name)

    def get_all_registered_tool_names(self) -> List[str]:
        """Return all registered tool names in registration order."""
        return list(self._tools.keys())

    def execute_tool_by_name(self, name: str, args: Dict[str, Any]) -> str:
        """Validate parameters and execute a tool by name.

        Why: Centralised execution with validation ensures consistent error
        handling across all tools — the main loop never calls tool.execute()
        directly.

        Returns the tool result string, or an error message if validation
        or execution fails.
        """
        tool = self._tools.get(name)
        if not tool:
            return f"(unknown tool: {name})"

        # Validate parameters
        validation_errors = tool.validate_parameters(args)
        if validation_errors:
            return f"(parameter error for {name}: {'; '.join(validation_errors)})"

        try:
            return tool.execute(**args)
        except Exception as exc:
            logger.warning("Tool execution error: %s(%s): %s", name, args, exc)
            return f"(tool error: {name}: {exc})"

    def generate_tool_definitions_for_prompt(self) -> str:
        """Generate the complete tool definitions block for the system prompt.

        Why: Dynamic generation from the registry means adding a new tool
        automatically updates the prompt — no manual string editing needed.
        """
        lines = [
            "You have these tools. To use a tool, output EXACTLY one <tool_call> tag:",
            '<tool_call>{"name": "TOOL_NAME", "args": {ARGS}}</tool_call>',
            "",
            "## Available Tools",
            "",
        ]
        for idx, tool in enumerate(self._tools.values(), 1):
            lines.append(f"{idx}. {tool.generate_prompt_definition()}")
            lines.append("")

        return "\n".join(lines)
