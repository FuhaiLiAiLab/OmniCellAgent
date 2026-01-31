#!/bin/bash

# Stop all MCP servers

echo "Stopping all MCP servers..."

# Kill all processes running mcp_servers/*.py
pkill -f "mcp_servers/.*_server.py"

# Wait a moment
sleep 2

# Verify
if pgrep -f "mcp_servers/.*_server.py" > /dev/null; then
    echo "⚠ Some servers still running. Force killing..."
    pkill -9 -f "mcp_servers/.*_server.py"
    sleep 1
fi

echo "✓ All MCP servers stopped"

# Show any remaining processes
remaining=$(pgrep -f "mcp_servers/.*_server.py")
if [ -n "$remaining" ]; then
    echo "⚠ Warning: Some processes may still be running:"
    ps aux | grep "mcp_servers" | grep -v grep
else
    echo "✓ Confirmed: No MCP server processes running"
fi
