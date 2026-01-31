"""PubMed MCP Server - Port 9001"""
import os
import sys
from typing import Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from fastmcp import FastMCP
from tools.pubmed_tools.query_pubmed_tool import query_medical_research_async
from dotenv import load_dotenv

load_dotenv(os.path.join(project_root, '.env'))
mcp = FastMCP("PubMed Research Tools 📚")


@mcp.tool()
async def search_pubmed(
    query: str,
    top_k: int = 5,
    session_id: Optional[str] = None
) -> str:
    """
    Search PubMed for medical research papers and retrieve full-text content.
    
    This tool searches the PubMed database, downloads PDFs when available,
    extracts full-text content, and returns comprehensive paper information.
    
    Files are saved to: webapp/sessions/{session_id}/pubmed/
    
    Args:
        query: Search query (e.g., "KRAS mutations in pancreatic cancer")
        top_k: Number of papers to retrieve (default: 5)
        session_id: Session identifier for file organization (default: auto-generated)
    
    Returns:
        Formatted text with:
        - Paper metadata (title, authors, journal, year, DOI, PMID)
        - Abstract
        - Full-text content (when available)
        - File paths to downloaded PDFs/XMLs
    
    Example:
        result = await search_pubmed(
            query="TP53 function in cancer",
            top_k=3,
            session_id="my_research_session"
        )
    
    Note:
        - Papers cached globally by DOI to avoid re-downloads
        - Supports PMC, Elsevier, Wiley sources (requires API keys in .env)
        - Returns abstracts even when full-text unavailable
    """
    if session_id is None:
        import uuid
        session_id = f"pubmed_{uuid.uuid4().hex[:8]}"
    
    return await query_medical_research_async(
        query=query,
        top_k=top_k,
        session_id=session_id
    )


if __name__ == "__main__":
    # Run as SSE server on port 9001 or stdio
    import sys
    
    if "--sse" in sys.argv:
        # Run as HTTP/SSE server
        mcp.run(transport="sse", port=9001)
    else:
        # Run as stdio server (default)
        mcp.run()
