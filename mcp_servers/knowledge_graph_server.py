"""
Knowledge Graph MCP Server

Exposes GRetriever knowledge graph tools via MCP protocol.
Port: 9003 (SSE) or stdio
"""

import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from fastmcp import FastMCP
from tools.gretriever_tools.gretriever_client import gretriever_tool
from dotenv import load_dotenv

load_dotenv(os.path.join(project_root, '.env'))

# Initialize MCP server
mcp = FastMCP("Knowledge Graph Tools 🧬")


@mcp.tool()
async def search_knowledge_graph(query: str) -> str:
    """
    Search biomedical knowledge graph using GRetriever.
    
    Queries a Neo4j-based knowledge graph containing:
    - Gene-disease associations
    - Protein-protein interactions
    - Pathway information
    - Drug-target relationships
    - Tissue-specific gene expression
    - And more biomedical entities and relationships
    
    The tool uses GRetriever service (port 8001) which combines:
    - Graph traversal algorithms
    - Semantic similarity search
    - LLM-based result synthesis
    
    Args:
        query: Natural language query about biomedical entities/relationships
               Examples:
               - "What genes are associated with Alzheimer's disease?"
               - "Find proteins that interact with TP53"
               - "What pathways involve KRAS?"
    
    Returns:
        Knowledge graph results with:
        - Relevant entities (genes, diseases, pathways, etc.)
        - Relationships between entities
        - Confidence scores
        - Source information
    
    Note:
        - Requires GRetriever service running on port 8001
        - Requires Neo4j database with biomedical knowledge graph
        - See scripts/startup.sh to start GRetriever service
    """
    return await gretriever_tool(query=query)


if __name__ == "__main__":
    import sys
    
    if "--sse" in sys.argv:
        mcp.run(transport="sse", port=9003)
    else:
        mcp.run()
