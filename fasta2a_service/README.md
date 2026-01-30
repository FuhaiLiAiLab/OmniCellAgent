# FastA2A Service for OmniCellAgent

This directory contains the FastA2A (Agent-to-Agent) protocol implementation for the OmniCellAgent LangGraph agent.

## Overview

The FastA2A service exposes the OmniCellAgent as an A2A-compliant server, allowing:
- Standard A2A protocol communication
- Structured state and output artifacts
- Multi-turn conversation support with context preservation
- Rich biomedical data artifacts (genes, pathways, papers, plots)

## Architecture

### Components

1. **Storage** (`storage.py`): Persists tasks and conversation context
   - Task storage: A2A protocol format
   - Context storage: Full LangGraph state including tool calls, structured data
   
2. **Worker** (`worker.py`): Executes the LangGraph agent
   - Runs agent queries asynchronously
   - Extracts structured outputs and artifacts
   - Handles state management
   
3. **Broker** (`broker.py`): Schedules and manages task execution
   - In-memory queue for task scheduling
   - Async task execution
   
4. **Server** (`server.py`): FastA2A ASGI application
   - HTTP endpoints for A2A protocol
   - Task submission and status checking
   - Artifact retrieval

### Key Features

**Beyond Simple Text Responses:**
- **Structured State Artifacts**: Export complete LangGraph state including:
  - Research plan with sub-tasks
  - Agent execution logs
  - Shared data (genes, pathways, papers)
  - Process tracking
  
- **Typed Data Artifacts**: Rich biomedical outputs as structured JSON:
  - Gene expression data with statistics
  - Pathway enrichment results
  - Literature references with citations
  - Analysis plots (base64-encoded)
  
- **Context Preservation**: Full conversation history with tool calls and results

## Installation

```bash
# Install FastA2A
pip install fasta2a

# Or install with Pydantic AI
pip install 'pydantic-ai-slim[a2a]'
```

## Usage

### Start the Server

```bash
# Using uvicorn
uvicorn fasta2a_service.server:app --host 0.0.0.0 --port 8000

# Or using the CLI script
python fasta2a_service/server.py --port 8000
```

### Send Requests

```python
import requests

# Submit a new task
response = requests.post(
    "http://localhost:8000/tasks",
    json={
        "prompt": "What are the key dysfunctional signaling targets in microglia of Alzheimer's disease?",
        "context_id": None  # Optional: provide to continue conversation
    }
)
task = response.json()
task_id = task["task_id"]

# Check task status
status = requests.get(f"http://localhost:8000/tasks/{task_id}").json()
print(f"Status: {status['status']}")

# Get artifacts when complete
if status["status"] == "completed":
    artifacts = status.get("artifacts", [])
    for artifact in artifacts:
        if artifact["part_type"] == "DataPart":
            print(f"Structured data: {artifact['data']}")
```

See `examples/client_example.py` for more detailed usage.

## Artifacts

The service produces multiple artifact types:

1. **Final Report** (TextPart): Comprehensive markdown report
2. **Research Plan** (DataPart): Structured task plan
3. **Agent State** (DataPart): Full LangGraph state
4. **Gene Data** (DataPart): Top genes with expression statistics
5. **Pathway Data** (DataPart): Enrichment results
6. **Literature** (DataPart): Paper metadata and citations
7. **Plots** (DataPart): Visualizations (volcano plots, heatmaps)

## Configuration

Edit `storage.py` to configure:
- Storage backend (file-based, database, etc.)
- Context retention policies
- Artifact limits

## Development

```bash
# Run tests
pytest fasta2a_service/tests/

# Type checking
mypy fasta2a_service/

# Linting
ruff check fasta2a_service/
```

## Protocol Compliance

This implementation follows the [A2A Protocol v1.0](https://github.com/google/a2a-protocol) specification.
