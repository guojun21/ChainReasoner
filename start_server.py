#!/usr/bin/env python3
"""
Simple script to start the API server with error handling
"""

import sys
import traceback

try:
    print("Starting Enhanced MultiHop Agent API Server...")
    print("=" * 70)
    
    # Import the server module
    import api_server_enhanced
    
    print("✓ Successfully imported api_server_enhanced module")
    
    # Create server instance
    print("Creating server instance...")
    server = api_server_enhanced.EnhancedMultiHopAPIServer()
    
    print("✓ Server instance created successfully")
    print(f"  - API Token: {server.api_token}")
    print(f"  - MCP Services: {len(server.mcp_config.get('mcpServers', {}))}")
    
    # Start the server
    print("\nStarting Flask server...")
    print(f"  - Host: 0.0.0.0")
    print(f"  - Port: 5000")
    print("  - Press CTRL+C to quit")
    print("=" * 70)
    
    server.run(host='0.0.0.0', port=5000)
    
except Exception as e:
    print("\n❌ Error starting server:")
    print(f"  - Exception type: {type(e).__name__}")
    print(f"  - Error message: {str(e)}")
    print("\nTraceback:")
    traceback.print_exc()
    sys.exit(1)
