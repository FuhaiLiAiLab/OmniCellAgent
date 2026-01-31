"""Scientist RAG MCP Server - Port 9004"""
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from fastmcp import FastMCP
from tools.scientist_rag_tools.scientist_tool import scientist_rag_tool_wrapper
from dotenv import load_dotenv

load_dotenv(os.path.join(project_root, '.env'))
mcp = FastMCP("Scientist RAG Tools 👨‍🔬")


@mcp.tool()
async def query_scientist_knowledge(
    author_name: str,
    question: str
) -> str:
    """
    Query RAG system built from specific scientist's publications.
    
    This tool:
    1. Retrieves publications by the specified author from PubMed
    2. Builds a specialized vector database from their work
    3. Answers questions using RAG over their publications
    4. Returns answers with citations
    
    Useful for:
    - Understanding a researcher's specific contributions
    - Finding methodologies used by an author
    - Contextualizing findings within an author's body of work
    - Exploring research themes and evolution
    
    Args:
        author_name: Full name of the scientist/author
                    Examples: "Eric Lander", "Jennifer Doudna", "David Baltimore"
        question: Question to ask about their work
                 Examples:
                 - "What are the main contributions to CRISPR?"
                 - "What methodologies does this author use?"
                 - "How has their research evolved over time?"
    
    Returns:
        Answer based on the author's publications with:
        - Direct answer to the question
        - Citations from relevant papers
        - Quotes from the author's work
        - PubMed IDs for reference
    
    Note:
        - Requires Scientist RAG service running on port 8000
        - Knowledge bases are cached in cache/author_kb/
        - First query for an author may take longer (building KB)
        - See scripts/startup.sh to start Scientist RAG service
    """
    return await scientist_rag_tool_wrapper(
        author_name=author_name,
        question=question
    )


if __name__ == "__main__":
    import sys
    
    if "--sse" in sys.argv:
        mcp.run(transport="sse", port=9004)
    else:
        mcp.run()
