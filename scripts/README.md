# Scripts Directory

This directory contains utility scripts for managing the OmniCellAgent microservices.

## Available Scripts

### 1. `startup.sh` - Start All Services

Starts all microservices in the background.

```bash
cd /home/hao/BioProtocol/OmniCellAgent
bash scripts/startup.sh
```

**Services Started:**
- Neo4j Database (from config)
- Scientist RAG Tool (port 8000)
- GRetriever Service (port 8001)
- Omic Fetch Analysis Workflow Microservice
- GLiNER Service
- BioBERT Service
- Ngrok Tunnel (domain: agent.omni-cells.com -> port 8050)
- Webapp (port 8050)

**Configuration:**
- Reads Neo4j path from `configs/paths.yaml`
- Uses conda environment: `langgraph-dev`
- Logs are saved to `logs/service-logs/`

### 2. `test_services.sh` - Test All Services

Tests each microservice to verify it's running and responding correctly.

```bash
cd /home/hao/BioProtocol/OmniCellAgent
bash scripts/test_services.sh
```

**Tests Performed:**
- Neo4j HTTP endpoint (port 7474)
- Scientist RAG Tool health check and query endpoints
- GRetriever Service health check and query endpoints
- Process checks for GLiNER, BioBERT, and Omic services
- Webapp accessibility (port 8050)
- Ngrok tunnel status

**Output:**
- ✓ Green checkmark for passing tests
- ✗ Red X for failing tests
- Summary with pass/fail counts
- Exit code 0 if all tests pass, 1 otherwise

### 3. `stop_services.sh` - Stop All Services

Gracefully stops all running microservices.

```bash
cd /home/hao/BioProtocol/OmniCellAgent
bash scripts/stop_services.sh
```

**Services Stopped:**
- All Python microservices (graceful shutdown with fallback to force kill)
- Neo4j Database (using `neo4j stop`)
- Ngrok tunnel

**Process:**
1. Attempts graceful shutdown (SIGTERM)
2. Waits 2 seconds
3. Force kills if still running (SIGKILL)
4. Verifies all services stopped

## Quick Command Reference

### Check Running Services
```bash
# All Python services
ps aux | grep python | grep -v grep

# Specific service
ps aux | grep -E "python.*scientist_tool" | grep -v grep

# Check ports
netstat -tlnp | grep -E "8000|8001|8050|7474|7687"
```

### View Logs
```bash
# View all service logs
ls -la logs/service-logs/

# Tail a specific service log
tail -f logs/service-logs/scientist_tool.log
tail -f logs/service-logs/gretriever_service_output.log
tail -f logs/service-logs/webapp_index.log
```

### Manual Service Control
```bash
# Kill specific service
pkill -f "python.*scientist_tool"
pkill -f "python.*gretriever_service"

# Stop Neo4j manually
/home/hao/Applications/neo4j-community-2025.10.1/bin/neo4j stop

# Start Neo4j manually
/home/hao/Applications/neo4j-community-2025.10.1/bin/neo4j start
```

## Curl Examples

### Scientist RAG Tool (port 8000)
```bash
# Health check
curl http://localhost:8000/health

# List authors
curl http://localhost:8000/authors

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"author": "Scientist 1", "question": "What are the latest findings on Alzheimer'\''s disease?"}'
```

### GRetriever Service (port 8001)
```bash
# Health check
curl http://localhost:8001/health

# Query knowledge graph
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "how to treat Alzheimer disease", "max_nodes": 50, "include_description": false}'
```

### Webapp (port 8050)
```bash
# Access web interface
curl http://localhost:8050
# Or open in browser: http://localhost:8050
```

### Neo4j (port 7474)
```bash
# HTTP endpoint
curl http://localhost:7474

# Bolt protocol uses port 7687 (use cypher-shell or Neo4j Browser)
```

## Troubleshooting

### Services Not Starting
1. Check if ports are already in use:
   ```bash
   netstat -tlnp | grep -E "8000|8001|8050"
   ```

2. Check conda environment:
   ```bash
   conda env list
   conda activate langgraph-dev  # or conda init first
   ```

3. View service logs:
   ```bash
   tail -f logs/service-logs/*.log
   ```

### Services Not Responding
1. Give services time to load (especially model-heavy services like GRetriever)
2. Check memory usage:
   ```bash
   free -h
   htop
   ```

3. Restart individual service:
   ```bash
   pkill -f "python.*scientist_tool"
   conda run -n langgraph-dev python tools/scientist_rag_tools/scientist_tool.py &
   ```

### Configuration Issues
- Neo4j path: Edit `configs/paths.yaml` → `external.neo4j_home`
- Conda environment: Edit `CONDA_ENV` variable in scripts
- Port conflicts: Check and modify port numbers in service files

## Notes

- All scripts use relative paths based on project root
- Configuration is centralized in `configs/paths.yaml`
- Services run in background using `nohup`
- Use `conda run -n langgraph-dev` to avoid conda init issues
