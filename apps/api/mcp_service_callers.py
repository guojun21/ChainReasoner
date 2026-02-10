"""Backward-compatibility shim — moved to mcp_service_dispatch_and_stdio_callers.py."""
from apps.api.mcp_service_dispatch_and_stdio_callers import (  # noqa: F401
    dispatch_mcp_service_call_to_appropriate_transport as call_mcp_service,
)
