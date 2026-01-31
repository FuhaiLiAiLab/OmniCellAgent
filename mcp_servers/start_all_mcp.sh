#!/bin/bash

# Start all MCP servers for OmniCellAgent
# Each server runs as a separate process on its own port

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Conda environment
CONDA_ENV="a2a-dev"

# Create logs directory
LOG_DIR="$PROJECT_ROOT/logs/mcp-servers"
mkdir -p "$LOG_DIR"

echo "=================================="
echo "  Starting OmniCellAgent MCP Servers"
echo "=================================="
echo "Project root: $PROJECT_ROOT"
echo "Logs: $LOG_DIR"
echo ""

# Function to start a server
start_server() {
    local name=$1
    local script=$2
    local port=$3
    
    echo "Starting $name (port $port)..."
    nohup conda run -n "$CONDA_ENV" python "$PROJECT_ROOT/mcp_servers/$script" --sse \
        > "$LOG_DIR/${name}.log" 2>&1 &
    echo "  ✓ $name started (PID: $!)"
}

# Start all servers
start_server "PubMed" "pubmed_server.py" 9001
start_server "WebSearch" "websearch_server.py" 9002
start_server "KnowledgeGraph" "knowledge_graph_server.py" 9003
start_server "ScientistRAG" "scientist_rag_server.py" 9004
start_server "OmicsAnalysis" "omics_server.py" 9005

echo ""
echo "=================================="
echo "  All MCP servers started!"
echo "=================================="
echo ""
echo "Server Status:"
echo "  PubMed:          http://localhost:9001"
echo "  Web Search:      http://localhost:9002"
echo "  Knowledge Graph: http://localhost:9003"
echo "  Scientist RAG:   http://localhost:9004"
echo "  Omics Analysis:  http://localhost:9005"
echo ""
echo "Logs: $LOG_DIR"
echo ""
echo "To stop all servers: bash mcp_servers/stop_all_mcp.sh"
echo "To check status: ps aux | grep mcp_servers"
echo ""
