"""Backward-compatibility shim — moved to model_context_protocol_transport_clients.py."""
from src.search.model_context_protocol_transport_clients import (  # noqa: F401
    ModelContextProtocolHttpTransportClient as MCPHttpClient,
    ModelContextProtocolStdioTransportClient as MCPStdioClient,
)
