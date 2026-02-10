"""Backward-compatibility shim — moved to web_interface_mcp_service_handler.py."""
from apps.web.web_interface_mcp_service_handler import (  # noqa: F401
    dispatch_web_mcp_service_call as call_mcp_service,
)
