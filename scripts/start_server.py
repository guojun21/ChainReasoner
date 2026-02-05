#!/usr/bin/env python3
"""
Simple script to start the API server with error handling
"""

import os
import sys
import traceback
from pathlib import Path

try:
    print("Starting Enhanced MultiHop Agent API Server...")
    print("=" * 70)
    
    base_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(base_dir))

    # Import the server module
    from apps.api.api_server import EnhancedMultiHopAPIServer
    
    print("✓ Successfully imported api_server module")
    
    # Create server instance
    print("Creating server instance...")
    server = EnhancedMultiHopAPIServer()
    
    print("✓ Server instance created successfully")
    print(f"  - API Token: {server.api_token}")
    print(f"  - MCP Services: {len(server.mcp_config.get('mcpServers', {}))}")
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))

    # Start the server
    print("\nStarting Flask server...")
    print(f"  - Host: {host}")
    print(f"  - Port: {port}")
    print("  - Press CTRL+C to quit")
    print("=" * 70)
    
    server.run(host=host, port=port)
    
except Exception as e:
    print("\n❌ Error starting server:")
    print(f"  - Exception type: {type(e).__name__}")
    print(f"  - Error message: {str(e)}")
    print("\nTraceback:")
    traceback.print_exc()
    sys.exit(1)
