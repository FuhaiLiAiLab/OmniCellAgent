# OmniCellAgent MCP Servers

Modular MCP servers exposing biomedical research tools via Model Context Protocol.

## Architecture

```
LangGraph Agent
    ├── PubMed MCP (port 9001)
    ├── WebSearch MCP (port 9002)
    ├── KnowledgeGraph MCP (port 9003)
    ├── ScientistRAG MCP (port 9004)
    └── Omics MCP (port 9005)
```

Each server runs independently - failure of one doesn't affect others.

## Installation

```bash
conda activate a2a-dev
pip install "fastmcp==3.0.0b1"
```

## Usage

### Start All Servers
```bash
bash mcp_servers/start_all_mcp.sh
```

### Stop All Servers
```bash
bash mcp_servers/stop_all_mcp.sh
```

### Individual Servers
```bash
# stdio mode (default, for local MCP clients)
python mcp_servers/pubmed_server.py

# SSE mode (for HTTP clients)
python mcp_servers/pubmed_server.py --sse
```

## MCP Servers

### PubMed (Port 9001)
- **Tool**: `search_pubmed(query, top_k, session_id)`
- **Function**: Search PubMed, download PDFs, extract full-text
- **Output**: `webapp/sessions/{session_id}/pubmed/`

### Web Search (Port 9002)
- **Tool**: `search_web(query, target_results, use_llm_filter)`
- **Function**: Google Custom Search with content extraction

### Knowledge Graph (Port 9003)
- **Tool**: `search_knowledge_graph(query)`
- **Function**: Query Neo4j biomedical knowledge graph
- **Requires**: GRetriever service (port 8001), Neo4j

### Scientist RAG (Port 9004)
- **Tool**: `query_scientist_knowledge(author_name, question)`
- **Function**: RAG over specific scientist's publications
- **Requires**: Scientist RAG service (port 8000)

### Omics Analysis (Port 9005)
- **Tool**: `analyze_omics_data(query, session_id)`
- **Function**: Single-cell RNA-seq analysis (NER, DEG, KEGG)
- **Output**: `webapp/sessions/{session_id}/dataset_outputs/`
- **Requires**: GLiNER (8002), BioBERT (8003), OmniCellTOSG database

## Session Management

Pass `session_id` to organize outputs:

```python
session_id = "research_abc123"

# All tools with same session_id
await search_pubmed(query="...", session_id=session_id)
await analyze_omics_data(query="...", session_id=session_id)

# Files organized under webapp/sessions/research_abc123/
```

## Integration with LangGraph

See [example_langgraph_integration.py](example_langgraph_integration.py) for complete example.

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Connect to server
server_params = StdioServerParameters(
    command="conda",
    args=["run", "-n", "a2a-dev", "python", "mcp_servers/pubmed_server.py"]
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool(
            "search_pubmed",
            arguments={"query": "KRAS mutations", "top_k": 3, "session_id": "test"}
        )
```

## Troubleshooting

**Server won't start:**
```bash
lsof -i :9001  # Check port usage
kill $(lsof -t -i:9001)  # Kill process
```

**Connection errors:**
```bash
tail -f logs/mcp-servers/PubMed.log  # Check logs
ps aux | grep mcp_servers  # Verify running
```

**Dependencies not running:**
```bash
bash scripts/startup.sh  # Start backend services
```

## Port Assignments

| Server | Port | Backend Service | Port |
|--------|------|-----------------|------|
| PubMed | 9001 | - | - |
| WebSearch | 9002 | - | - |
| KnowledgeGraph | 9003 | GRetriever | 8001 |
| ScientistRAG | 9004 | Scientist RAG | 8000 |
| Omics | 9005 | GLiNER, BioBERT | 8002, 8003 |

## Resources

- **FastMCP**: https://gofastmcp.com
- **MCP Spec**: https://github.com/modelcontextprotocol
