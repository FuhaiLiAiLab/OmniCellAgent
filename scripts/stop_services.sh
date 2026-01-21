#!/bin/bash

# Get the project root directory (parent of scripts folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Conda environment name
CONDA_ENV="langgraph-dev"

# Read Neo4j path from config (paths.yaml)
NEO4J_HOME=$(conda run -n "$CONDA_ENV" python -c "import yaml; print(yaml.safe_load(open('$PROJECT_ROOT/configs/paths.yaml'))['external']['neo4j_home'])" 2>/dev/null)

# Fallback to default if config read fails
if [ -z "$NEO4J_HOME" ] || [ ! -d "$NEO4J_HOME" ]; then
    NEO4J_HOME="/home/hao/Applications/neo4j-community-2025.10.1"
fi

echo "========================================"
echo "Stopping All Services"
echo "========================================"
echo ""

# Function to stop a process
stop_process() {
    local name=$1
    local pattern=$2
    
    echo -n "Stopping $name... "
    pids=$(ps aux | grep -E "$pattern" | grep -v grep | awk '{print $2}')
    
    if [ -z "$pids" ]; then
        echo "Not running"
        return
    fi
    
    # Try graceful shutdown first
    echo $pids | xargs kill 2>/dev/null
    sleep 2
    
    # Check if still running, force kill if necessary
    pids=$(ps aux | grep -E "$pattern" | grep -v grep | awk '{print $2}')
    if [ -n "$pids" ]; then
        echo -n "(force killing) "
        echo $pids | xargs kill -9 2>/dev/null
        sleep 1
    fi
    
    echo "✓ Stopped"
}

# Stop all services
stop_process "Scientist RAG Tool" "python.*scientist_tool.py"
stop_process "GRetriever Service" "python.*gretriever_service.py"
stop_process "Omic Fetch Analysis" "python.*omic_fetch_analysis_workflow_microservice.py"
stop_process "GLiNER Service" "python.*gliner_service.py"
stop_process "BioBERT Service" "python.*biobert_service.py"
stop_process "Webapp" "python.*webapp/index.py"
stop_process "Ngrok" "ngrok.*8050"

# Stop Neo4j
echo -n "Stopping Neo4j... "
if [ -f "$NEO4J_HOME/bin/neo4j" ]; then
    "$NEO4J_HOME/bin/neo4j" stop > /dev/null 2>&1
    sleep 2
    echo "✓ Stopped"
else
    echo "Neo4j binary not found at $NEO4J_HOME"
fi

echo ""
echo "========================================"
echo "All Services Stopped"
echo "========================================"
echo ""

# Verify nothing is left running
echo "Verifying all services are stopped..."
remaining=$(ps aux | grep -E "(scientist_tool|gretriever_service|omic_fetch|gliner_service|biobert_service|webapp/index|ngrok.*8050)" | grep -v grep | wc -l)

if [ "$remaining" -eq 0 ]; then
    echo "✓ All services successfully stopped"
    exit 0
else
    echo "⚠ Warning: $remaining processes still running"
    echo "Run 'ps aux | grep python' to check"
    exit 1
fi
