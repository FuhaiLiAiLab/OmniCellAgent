# OmniCellAgent MCP Servers

Modular MCP servers exposing biomedical research tools via Model Context Protocol.

## Architecture

```
LangGraph Agent
    ├── PubMed MCP (port 9010)
    ├── WebSearch MCP (port 9011)
    ├── KnowledgeGraph MCP (port 9003)
    ├── ScientistRAG MCP (port 9004)
    ├── Omics MCP (port 9005)
    └── LiteratureSearch MCP (port 9012)
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

### PubMed (Port 9010)
- **Tool**: `search_pubmed(query, top_k, session_id)`
- **Function**: Search PubMed, download PDFs, extract full-text
- **Output**: `webapp/sessions/{session_id}/pubmed/`

### Web Search (Port 9011)
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

### Literature Search (Port 9012)
- **Tools**:
  - `search_pubmed_abstracts_langchain(query, top_k, session_id, doc_content_chars_max)`
  - `search_pubmed_abstracts(query, top_k, session_id)`
  - `search_pubmed_full_papers(query, top_k, session_id, max_message_chars_per_paper)`
  - `search_web_full_text(query, target_results, session_id, use_llm_filter)`
  - `search_literature_bundle(query, abstract_top_k, full_paper_top_k, web_results, session_id, abstract_source)`
- **Function**: Abstract search, PubMed full-paper retrieval, and text-browser literature search
- **Output**: JSON strings with `message` for direct LLM use and `metadata` containing raw abstracts, full paper text, or web page text
- **Storage**: uses the shared configured PubMed JSONL/DOI cache; response payload snapshots are saved under `mcp_servers/literature_outputs/`
- **Agent docs**: [LITERATURE_SEARCH_README.md](LITERATURE_SEARCH_README.md), [LITERATURE_MCP_AGENT_DEV_GUIDE.md](LITERATURE_MCP_AGENT_DEV_GUIDE.md)

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
lsof -i :9010  # Check PubMed MCP port usage
kill $(lsof -t -i:9010)  # Kill process using PubMed MCP port
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
| PubMed | 9010 | - | - |
| WebSearch | 9011 | - | - |
| KnowledgeGraph | 9003 | GRetriever | 8001 |
| ScientistRAG | 9004 | Scientist RAG | 8000 |
| Omics | 9005 | GLiNER, BioBERT | 8002, 8003 |
| LiteratureSearch | 9012 | - | - |

## Resources

- **FastMCP**: https://gofastmcp.com
- **MCP Spec**: https://github.com/modelcontextprotocol
