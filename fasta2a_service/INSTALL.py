"""
Installation and setup instructions for FastA2A service.
"""

# Required dependencies
DEPENDENCIES = [
    "starlette",
    "uvicorn[standard]",
    "requests",  # For client examples
]

# Optional dependencies
OPTIONAL_DEPENDENCIES = [
    "fasta2a",  # If you want to use the official package (not required for this implementation)
]

print("""
FastA2A OmniCellAgent Service - Installation Guide
==================================================

This service exposes the LangGraph OmniCellAgent via the Agent2Agent (A2A) protocol.

Prerequisites:
--------------
1. Python 3.10+
2. Conda environment activated
3. All OmniCellAgent dependencies installed

Installation Steps:
-------------------

1. Install FastA2A service dependencies:

   conda run -n fasta2a-dev pip install starlette uvicorn[standard] requests

2. (Optional) Install official fasta2a package:

   conda run -n fasta2a-dev pip install fasta2a

   Note: The current implementation is standalone and doesn't require the official package.
   If you want to integrate with the official FastA2A library later, install it.

Running the Service:
--------------------

1. Start the server:

   conda run -n fasta2a-dev python fasta2a_service/server.py --port 8000

   Or with uvicorn directly:

   conda run -n fasta2a-dev uvicorn fasta2a_service.server:app --host 0.0.0.0 --port 8000

2. Test the service:

   conda run -n fasta2a-dev python fasta2a_service/examples/client_example.py

API Endpoints:
--------------

GET  /                           - Health check
POST /tasks                      - Submit new task
GET  /tasks                      - List all tasks
GET  /tasks/{task_id}            - Get task status and results
GET  /contexts/{context_id}      - Get conversation context

Configuration:
--------------

Storage location: ./fasta2a_storage/ (configurable in server.py)
Model: gemini-2.0-flash-exp (configurable in server.py)
Max concurrent tasks: 2 (configurable in server.py)

Key Features Beyond Simple Text:
---------------------------------

1. Structured State Artifacts:
   - Complete LangGraph state with execution details
   - Research plan with sub-task breakdown
   - Process logs with all agent actions

2. Rich Biomedical Data:
   - Gene expression data with statistics (log2FC, p-values)
   - Pathway enrichment results
   - Literature references with citations
   - Analysis plots (volcano plots, heatmaps) as base64

3. Context Preservation:
   - Multi-turn conversations with full state
   - Accumulated data across conversation
   - Message history with tool calls

4. A2A Protocol Compliance:
   - Standard TextPart artifacts for reports
   - DataPart artifacts for structured data
   - Metadata with type information and JSON schemas

Example Usage:
--------------

import requests

# Submit a task
response = requests.post("http://localhost:8000/tasks", json={
    "prompt": "What are the key genes in Alzheimer's disease microglia?",
    "context_id": None  # New conversation
})
task = response.json()

# Check status
status = requests.get(f"http://localhost:8000/tasks/{task['task_id']}").json()

# When completed, get artifacts
if status["status"] == "completed":
    for artifact in status["artifacts"]:
        if artifact["metadata"]["type"] == "gene_expression_data":
            genes = artifact["data"]["result"]
            print(f"Found {len(genes)} genes")

See examples/client_example.py for more detailed examples.

Troubleshooting:
----------------

1. Import errors:
   - Ensure all OmniCellAgent dependencies are installed
   - Check that PYTHONPATH includes project root

2. Model errors:
   - Set GOOGLE_API_KEY environment variable
   - Or modify server.py to use different model

3. Storage errors:
   - Check write permissions for ./fasta2a_storage/
   - Ensure sufficient disk space

4. Memory issues:
   - Reduce max_concurrent_tasks in server.py
   - Use organ filters in omics queries

For more information, see:
- README.md in fasta2a_service/
- FastA2A documentation: https://github.com/pydantic/fasta2a
- A2A Protocol: https://github.com/google/a2a-protocol
""")
