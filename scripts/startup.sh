#!/bin/bash

# Get the project root directory (parent of scripts folder)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Conda environment name
CONDA_ENV="langgraph-dev"

# Resolve direct python interpreters (more reliable than `conda run` in
# non-interactive shells where conda's PATH manipulation hasn't been
# initialised). Both envs must exist; bail out with a clear message if not.
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
LG_PY="$CONDA_BASE/envs/$CONDA_ENV/bin/python"
AUTOGEN_PY="$CONDA_BASE/envs/autogen-dev/bin/python"
for py in "$LG_PY" "$AUTOGEN_PY"; do
  if [ ! -x "$py" ]; then
    echo "ERROR: Python interpreter not found at $py" >&2
    echo "       Adjust CONDA_BASE or env names in this script." >&2
    exit 1
  fi
done

# Create logs directory if it doesn't exist
LOG_DIR="$PROJECT_ROOT/logs/service-logs"
mkdir -p "$LOG_DIR"

# Read Neo4j path from config (paths.yaml)
# Using Python to parse YAML and extract the neo4j_home path
NEO4J_HOME=$("$LG_PY" -c "import yaml; print(yaml.safe_load(open('$PROJECT_ROOT/configs/paths.yaml'))['external']['neo4j_home'])" 2>/dev/null)

# Fallback to default if config read fails
if [ -z "$NEO4J_HOME" ] || [ ! -d "$NEO4J_HOME" ]; then
    echo "Warning: Could not read Neo4j path from config or path does not exist"
    NEO4J_HOME="/home/hao/Applications/neo4j-community-2025.10.1"
fi

echo "Starting services from project root: $PROJECT_ROOT"
echo "Neo4j home: $NEO4J_HOME"

# Start all services
# Start Neo4j
nohup "$NEO4J_HOME/bin/neo4j" console > "$NEO4J_HOME/logs/neo4j_log.out" 2>&1 &
echo "Started Neo4j"

# Start Scientist RAG Tool (port 8000)
nohup "$LG_PY" "$PROJECT_ROOT/tools/scientist_rag_tools/scientist_tool.py" > "$LOG_DIR/scientist_tool.log" 2>&1 &
echo "Started Scientist RAG Tool"

# Start GRetriever Service (port 8001) - uses autogen-dev for torch/torch_geometric
nohup "$AUTOGEN_PY" "$PROJECT_ROOT/tools/gretriever_tools/gretriever_service.py" > "$LOG_DIR/gretriever_service_output.log" 2>&1 &
echo "Started GRetriever Service"

# Start Omic Fetch Analysis Workflow Microservice (commented out - already integrated)
# nohup "$LG_PY" "$PROJECT_ROOT/tools/omic_tools/omic_fetch_analysis_workflow_microservice.py" > "$LOG_DIR/omic_load_fetch_service.log" 2>&1 &
# echo "Started Omic Fetch Analysis Service"

# GLiNER Service intentionally disabled — the LangGraph agent path does its
# own NER via the LLM, and the package isn't installed in langgraph-dev.
# Re-enable below if you want text-mode NER for the omic_fetch_analysis CLI.
# nohup "$LG_PY" "$PROJECT_ROOT/tools/omic_tools/microservice/gliner_service.py" > "$LOG_DIR/gliner_service.log" 2>&1 &
# echo "Started GLiNER Service"

# Start BioBERT Service
nohup "$LG_PY" "$PROJECT_ROOT/tools/omic_tools/microservice/biobert_service.py" > "$LOG_DIR/biobert_service.log" 2>&1 &
echo "Started BioBERT Service"

# Start ngrok tunnel
# nohup ngrok http --domain=agent.omni-cells.com 8050 > "$LOG_DIR/ngrok.out" 2>&1 &
# echo "Started ngrok tunnel"

# Start Webapp (port 8050)
nohup "$LG_PY" "$PROJECT_ROOT/webapp/index.py" > "$LOG_DIR/webapp_index.log" 2>&1 &
echo "Started Webapp"

echo ""
echo "All services started. Check logs in: $LOG_DIR"
echo "To check running processes: ps aux | grep python" 