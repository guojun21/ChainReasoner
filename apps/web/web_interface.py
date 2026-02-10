"""Backward-compatibility shim — moved to enhanced_multi_hop_web_interface.py."""
from apps.web.enhanced_multi_hop_web_interface import (  # noqa: F401
    EnhancedMultiHopReasoningWebInterface as EnhancedMultiHopWebInterface,
    main,
)
