"""Backward-compatibility shim — moved to console_interface_mcp_service_handler.py."""
from apps.console.console_interface_mcp_service_handler import (  # noqa: F401
    dispatch_console_mcp_service_call as call_mcp_service,
)
