"""
Web Search MCP Server

Exposes Google Custom Search tools via MCP protocol.
Port: 9002 (SSE) or stdio
"""

import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from fastmcp import FastMCP
from tools.google_search_tools.google_search_w3m import web_search_tool
from dotenv import load_dotenv

load_dotenv(os.path.join(project_root, '.env'))

# Initialize MCP server
mcp = FastMCP("Web Search Tools 🌐")


@mcp.tool()
async def search_web(
    query: str,
    target_results: int = 10,
    use_llm_filter: bool = True
) -> str:
    """
    Search the web using Google Custom Search and extract content from pages.
    
    This tool performs Google Custom Search, scrapes content from result pages
    using w3m text browser, and optionally filters results using LLM.
    
    Args:
        query: Search query
        target_results: Number of results to return (default: 10)
        use_llm_filter: Whether to use LLM to filter results for relevance (default: True)
    
    Returns:
        Formatted search results with:
        - URL and title
        - Page content (cleaned text)
        - Relevance score (if LLM filtering enabled)
    
    Example:
        result = await search_web(
            query="latest CRISPR gene editing breakthroughs",
            target_results=5,
            use_llm_filter=True
        )
    
    Note:
        - Requires GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID in .env
        - LLM filtering uses OpenAI to rank results by relevance
        - Content extraction uses w3m for clean text output
    """
    return await web_search_tool(
        query=query,
        target_results=target_results,
        use_llm_filter=use_llm_filter
    )


if __name__ == "__main__":
    import sys
    
    if "--sse" in sys.argv:
        mcp.run(transport="sse", port=9002)
    else:
        mcp.run()
