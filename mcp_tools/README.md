# OmniCellAgent MCP Tools

Model Context Protocol (MCP) server exposing OmniCellAgent's biomedical research capabilities.

## Overview

This MCP server provides AI agents (Claude, GitHub Copilot, etc.) with access to OmniCellAgent's specialized biomedical research tools through the standardized Model Context Protocol.

## Features

### 🔬 Research Tools

1. **PubMed Search** (`search_pubmed`)
   - Search PubMed database
   - Download and extract full-text papers (PDF, XML)
   - Support for PMC, Elsevier, Wiley sources
   - Returns abstracts and full-text content

2. **Web Search** (`search_web`)
   - Google Custom Search integration
   - Intelligent content extraction (w3m)
   - LLM-based relevance filtering
   - Returns formatted search results

3. **Knowledge Graph** (`search_knowledge_graph`)
   - Query Neo4j biomedical knowledge graph
   - Gene-disease associations
   - Protein interactions
   - Pathway information
   - Drug-target relationships

4. **Scientist RAG** (`query_scientist_knowledge`)
   - Author-specific knowledge bases
   - RAG over researcher's publications
   - Citation-backed answers
   - Track scientist contributions

5. **Omics Analysis** (`analyze_omics_data`)
   - Single-cell RNA-seq analysis
   - NER for disease/cell type extraction
   - Differential expression (DESeq2)
   - KEGG pathway enrichment
   - Automatic visualization
   - Comprehensive reports

## Installation

### Prerequisites

```bash
# Ensure you're in the OmniCellAgent conda environment
conda activate a2a-dev  # or langgraph-dev

# Install FastMCP 3.0 (already installed if you set up A2A)
pip install "fastmcp==3.0.0b1"
```

### Environment Setup

Make sure your `.env` file contains:

```bash
# Required for PubMed and Web Search
GOOGLE_API_KEY=your_google_api_key
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id

# Optional API keys for paper access
ELSEVIER_API_KEY=your_elsevier_key
WILEY_TDM_TOKEN=your_wiley_token

# Neo4j for Knowledge Graph (if using local instance)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

## Usage

### Starting the Server

```bash
cd /home/hao/BioProtocol/OmniCellAgent/mcp_tools
conda activate a2a-dev
python server.py
```

The server runs via stdio by default (suitable for local MCP clients like Claude Desktop).

### Using with Claude Desktop

Add to your Claude Desktop MCP configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on Mac):

```json
{
  "mcpServers": {
    "omnicell": {
      "command": "conda",
      "args": [
        "run",
        "-n",
        "a2a-dev",
        "python",
        "/home/hao/BioProtocol/OmniCellAgent/mcp_tools/server.py"
      ]
    }
  }
}
```

### Using with MCP Client SDK

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="conda",
        args=["run", "-n", "a2a-dev", "python", "/path/to/mcp_tools/server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Call a tool
            result = await session.call_tool(
                "search_pubmed",
                arguments={
                    "query": "KRAS mutations in pancreatic cancer",
                    "top_k": 3
                }
            )
            
            print(result.content[0].text)

asyncio.run(main())
```

## Tool Reference

### search_pubmed

Search PubMed and retrieve full-text papers.

**Parameters:**
- `query` (str): Search query
- `top_k` (int, optional): Number of papers (default: 5)
- `session_id` (str, optional): Session identifier for caching

**Returns:** Formatted paper information with full-text content

**Example:**
```python
await session.call_tool(
    "search_pubmed",
    arguments={
        "query": "TP53 function in lung cancer",
        "top_k": 5
    }
)
```

### search_web

Google Custom Search with content extraction.

**Parameters:**
- `query` (str): Search query
- `target_results` (int, optional): Number of results (default: 10)
- `use_llm_filter` (bool, optional): Use LLM filtering (default: True)

**Returns:** Search results with page content

### search_knowledge_graph

Query biomedical knowledge graph.

**Parameters:**
- `query` (str): Natural language query

**Returns:** Knowledge graph results with entities and relationships

### query_scientist_knowledge

RAG over specific author's publications.

**Parameters:**
- `author_name` (str): Full name of scientist
- `question` (str): Question about their work

**Returns:** Answer with citations from author's papers

### analyze_omics_data

Comprehensive single-cell omics analysis.

**Parameters:**
- `query` (str): Analysis query (disease, cell type, etc.)
- `session_id` (str, optional): Session identifier

**Returns:** Analysis report with:
- Top differentially expressed genes
- Enriched pathways
- Sample statistics
- Plot paths

**Note:** Requires OmniCellTOSG database installation. Analysis takes 5-15 minutes.

## Architecture

```
mcp_tools/
├── server.py           # Main MCP server (FastMCP 3.0)
└── README.md          # This file

Dependencies:
├── tools/
│   ├── pubmed_tools/       # PubMed search implementation
│   ├── google_search_tools/ # Web search implementation
│   ├── gretriever_tools/   # Knowledge graph client
│   ├── scientist_rag_tools/ # Author RAG system
│   └── omic_tools/         # Omics analysis workflow
└── utils/
    ├── path_config.py      # Path management
    └── prompt.py           # System prompts
```

## Comparison with A2A Service

| Feature | MCP Server | A2A Server |
|---------|-----------|------------|
| Protocol | MCP (stdio/SSE) | Agent-to-Agent HTTP |
| Use Case | IDE integration (Claude, Copilot) | Agent-to-agent communication |
| Task Management | Synchronous | Async with polling |
| Client Support | MCP clients (Claude Desktop, etc.) | Any HTTP client |
| Best For | Interactive development | Long-running workflows |

## Troubleshooting

### "Module not found" errors

Make sure you're running from the correct directory and environment:
```bash
cd /home/hao/BioProtocol/OmniCellAgent
conda activate a2a-dev
python mcp_tools/server.py
```

### Google API errors

Verify your `.env` file has valid credentials:
```bash
# Check if keys are set
grep GOOGLE_API_KEY .env
grep GOOGLE_SEARCH_ENGINE_ID .env
```

### Neo4j connection errors

Ensure Neo4j is running:
```bash
# Check Neo4j status
sudo systemctl status neo4j
# Or start if needed
sudo systemctl start neo4j
```

### Omics analysis fails

1. Verify OmniCellTOSG database is installed at path specified in `configs/paths.yaml`
2. Check that you have sufficient disk space (datasets can be large)
3. Ensure R is installed with required packages (for enrichment analysis)

## Development

### Adding New Tools

1. Import your tool function in `server.py`
2. Create a wrapper with `@mcp.tool()` decorator
3. Add comprehensive docstring (used for tool description)
4. Follow FastMCP 3.0 patterns for type hints

Example:
```python
@mcp.tool()
async def my_new_tool(query: str, param: int = 10) -> str:
    """
    Tool description (shown to AI agent).
    
    Args:
        query: Query parameter description
        param: Optional parameter description
    
    Returns:
        Description of return value
    """
    # Implementation
    result = await my_async_function(query, param)
    return result
```

### Testing Tools

```python
# Test individual tool
from fastmcp import FastMCP

mcp = FastMCP("Test")

# ... add tools ...

if __name__ == "__main__":
    # Run with --test flag
    import sys
    if "--test" in sys.argv:
        # Test mode - call tools directly
        import asyncio
        result = asyncio.run(search_pubmed("test query", top_k=1))
        print(result)
    else:
        mcp.run()
```

## Resources

- **FastMCP Documentation**: https://gofastmcp.com
- **MCP Specification**: https://github.com/modelcontextprotocol
- **OmniCellAgent**: https://github.com/FuhaiLiAiLab/OmniCellAgent

## License

Same as OmniCellAgent project.
