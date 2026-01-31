# FastA2A Service for OmniCellAgent

Exposes the OmniCellAgent LangGraph agent through the A2A (Agent-to-Agent) protocol.

## Quick Start

### Start Server
```bash
# Start in background  
nohup python -m uvicorn fasta2a_service.server:app --host 0.0.0.0 --port 8021 > fasta2a_service/server.log 2>&1 &

# Or foreground
python -m uvicorn fasta2a_service.server:app --host 0.0.0.0 --port 8021
```

### Test
```bash
python fasta2a_service/test_a2a.py
```

## API Endpoints

- **POST /tasks** - Submit a biomedical research query  
- **GET /tasks/{task_id}** - Get task status and results
- **GET /tasks** - List all tasks
- **GET /contexts/{context_id}** - Get conversation context
- **GET /health** - Health check

## Example Usage

### Submit Query
```bash
curl -X POST http://localhost:8021/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the key dysfunctional genes and pathways in PDAC?",
    "session_id": "pdac-test"
  }'
```

### Check Status
```bash
curl http://localhost:8021/tasks/{task_id}
```

## Features

- **A2A Protocol** - Standard agent-to-agent communication
- **Async Execution** - Non-blocking long-running analyses (timeout: 2000s)  
- **Session Management** - Automatic session IDs for result tracking
- **Rich Artifacts** - Text reports, gene data, pathways, plots
- **Context Preservation** - Multi-turn conversation support

## Architecture

```
fasta2a_service/
├── server.py          # Main A2A server
├── utils.py           # Storage, worker, broker  
├── test_a2a.py        # Test script
└── README.md          # This file
```

## Storage

- Tasks: `./fasta2a_storage/tasks/`
- Contexts: `./fasta2a_storage/contexts/`  
- Logs: `./fasta2a_service/server.log`

## Stop Server

```bash
pkill -f "uvicorn fasta2a_service.server:app"
```

## Notes

- Port 8021 (not 8000 which is used by main webapp)
- Timeout: 2000 seconds for long-running biomedical analyses  
- Results saved in `webapp/sessions/<session_id>/`
