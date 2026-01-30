# A2A Service - Quick Reference

## ✅ Setup Complete!

The FastA2A service has been configured and tested. It exposes OmniCellAgent through the A2A (Agent-to-Agent) protocol for interoperability with GitHub Copilot and other agents.

## 🚀 Quick Start

### 1. Start the Server
```bash
# Using the startup script (recommended)
./fasta2a_service/start_server.sh

# Or manually with nohup
conda activate a2a-dev
cd /home/hao/BioProtocol/OmniCellAgent
nohup python -m uvicorn fasta2a_service.server:app --host 0.0.0.0 --port 8000 > fasta2a_service/server.log 2>&1 &

# Check if server is running
curl http://localhost:8000/tasks
```

### 2. Test the Service
```bash
# Run the test script
./fasta2a_service/run_test.sh

# Or manually
conda activate a2a-dev
python fasta2a_service/test_a2a_connection.py
```

### 3. Stop the Server
```bash
pkill -f "uvicorn fasta2a_service.server"
```

## 📡 API Endpoints

- **POST /tasks** - Submit a new biomedical query
- **GET /tasks/{task_id}** - Get task status and results
- **GET /tasks** - List all tasks
- **GET /contexts/{context_id}** - Get conversation context

## 💡 Example Usage

### Submit a Query (Python)
```python
import requests

response = requests.post(
    "http://localhost:8000/tasks",
    json={
        "prompt": "What are the key dysfunctional genes and pathways in pancreatic ductal adenocarcinoma (PDAC)?",
        "context_id": None
    }
)
task = response.json()
print(f"Task ID: {task['task_id']}")
```

### Submit a Query (curl)
```bash
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the key dysfunctional genes and pathways in Alzheimer'\''s Disease?",
    "context_id": null
  }'
```

### Check Task Status
```bash
# Replace with your task ID
curl http://localhost:8000/tasks/YOUR_TASK_ID
```

## 🔌 Connecting with GitHub Copilot

GitHub Copilot can connect to this A2A service by:

1. **Making HTTP requests** to `http://localhost:8000/tasks`
2. **Submitting queries** with disease/pathway questions
3. **Polling for results** using the task_id
4. **Retrieving artifacts** including:
   - Text reports (markdown)
   - Gene expression data (structured JSON)
   - Pathway enrichment results
   - Visualizations (base64-encoded plots)
   - Full LangGraph state

### Key Features for Copilot Integration

✅ **Standard A2A Protocol** - Compatible with any A2A client  
✅ **Asynchronous Execution** - Long-running analyses don't block  
✅ **Rich Artifacts** - Structured data beyond simple text  
✅ **Context Preservation** - Multi-turn conversations supported  
✅ **Biomedical Intelligence** - Powered by OmniCellAgent  

## 📂 Storage

- **Tasks**: `./fasta2a_storage/tasks/`
- **Contexts**: `./fasta2a_storage/contexts/`
- **Logs**: `./fasta2a_service/server.log`

## 🔧 Configuration

The server uses default agent configuration from `agent/langgraph_agent.py`:
- No model specification needed (uses gemini-3-pro by default)
- Inherits all OmniCellAgent capabilities
- Access to Neo4j, RAG tools, and microservices

## 📝 Example Queries

Try these biomedical research queries:

```
"What are the key dysfunctional genes and pathways in pancreatic ductal adenocarcinoma (PDAC)?"

"What are the key dysfunctional genes and pathways in Alzheimer's Disease?"

"What are the key dysfunctional genes and pathways in Lung adenocarcinoma (LUAD)?"

"Analyze breast cancer: identify relevant genes and therapeutic targets"
```

## 🛠️ Troubleshooting

### Server won't start
```bash
# Check if port 8000 is in use
lsof -i:8000

# Kill existing server
pkill -f "uvicorn fasta2a_service.server"

# Check logs
tail -f fasta2a_service/server.log
```

### Tasks fail immediately
- Ensure Neo4j is running (required for agent)
- Check that API keys are configured in `configs/db.env`
- Verify all microservices are running (see main README)

### Connection refused
- Server may not be fully started (wait 3-5 seconds)
- Check server.log for startup errors
- Verify conda environment is activated

## 📚 Documentation

- Full API spec: See [fasta2a_service/DESIGN.md](DESIGN.md)
- Setup guide: See [fasta2a_service/QUICKSTART.md](QUICKSTART.md)
- Main project: See [README.md](../README.md)
