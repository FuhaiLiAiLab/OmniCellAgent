"""
OmniCellAgent MCP Server

FastMCP 3.0 server exposing OmniCellAgent's biomedical research tools via MCP protocol.
Provides tools for:
- PubMed literature search and analysis
- Google web search
- Knowledge graph retrieval (GRetriever)
- Scientist RAG (author-specific knowledge)
- Omics data analysis workflow

All tools are exposed as MCP tools for use with Claude, GitHub Copilot, and other MCP clients.
"""

import os
import sys
import asyncio
from typing import Optional
from fastmcp import FastMCP

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import tool implementations
from tools.pubmed_tools.query_pubmed_tool import query_medical_research_async
from tools.google_search_tools.google_search_w3m import web_search_tool
from tools.gretriever_tools.gretriever_client import gretriever_tool
from tools.scientist_rag_tools.scientist_tool import scientist_rag_tool_wrapper

# Import omic tools
sys.path.insert(0, os.path.join(project_root, 'tools', 'omic_tools'))
from omic_fetch_analysis_workflow import omic_fetch_analysis_workflow

# Load environment
from dotenv import load_dotenv
load_dotenv(os.path.join(project_root, '.env'))

# Initialize FastMCP server
mcp = FastMCP("OmniCellAgent Biomedical Research Tools 🧬")

# ==============================================================================
# PUBMED TOOLS
# ==============================================================================

@mcp.tool()
async def search_pubmed(
    query: str,
    top_k: int = 5,
    session_id: Optional[str] = None
) -> str:
    """
    Search PubMed for medical research papers and retrieve full-text content.
    
    This tool:
    1. Searches PubMed database for relevant papers
    2. Downloads PDFs when available (from PMC, Elsevier, Wiley, etc.)
    3. Extracts full-text content from PDFs
    4. Returns comprehensive paper information including:
       - Title, authors, journal, year
       - Abstract
       - Full-text content (when available)
       - DOI and PubMed ID
    
    Args:
        query: Search query (e.g., "KRAS mutations in pancreatic cancer")
        top_k: Number of papers to retrieve (default: 5)
        session_id: Optional session identifier for caching (default: auto-generated)
    
    Returns:
        Formatted text with paper information and full-text content
    
    Example:
        result = await search_pubmed(
            query="TP53 function in cancer",
            top_k=3
        )
    """
    if session_id is None:
        import uuid
        session_id = f"mcp_{uuid.uuid4().hex[:8]}"
    
    return await query_medical_research_async(
        query=query,
        top_k=top_k,
        session_id=session_id
    )


# ==============================================================================
# GOOGLE SEARCH TOOLS
# ==============================================================================

@mcp.tool()
async def search_web(
    query: str,
    target_results: int = 10,
    use_llm_filter: bool = True
) -> str:
    """
    Search the web using Google Custom Search and extract content from pages.
    
    This tool:
    1. Performs Google Custom Search
    2. Scrapes content from result pages using w3m
    3. Optionally filters results using LLM for relevance
    4. Returns formatted search results with page content
    
    Args:
        query: Search query
        target_results: Number of results to return (default: 10)
        use_llm_filter: Whether to use LLM to filter results for relevance (default: True)
    
    Returns:
        Formatted search results with URLs, titles, and page content
    
    Example:
        result = await search_web(
            query="latest CRISPR gene editing breakthroughs",
            target_results=5
        )
    """
    return await web_search_tool(
        query=query,
        target_results=target_results,
        use_llm_filter=use_llm_filter
    )


# ==============================================================================
# KNOWLEDGE GRAPH TOOLS
# ==============================================================================

@mcp.tool()
async def search_knowledge_graph(query: str) -> str:
    """
    Search biomedical knowledge graph using GRetriever.
    
    This tool queries a Neo4j-based knowledge graph containing:
    - Gene-disease associations
    - Protein-protein interactions
    - Pathway information
    - Drug-target relationships
    - And more biomedical entities and relationships
    
    Args:
        query: Natural language query about biomedical entities/relationships
    
    Returns:
        Knowledge graph results with relevant entities and relationships
    
    Example:
        result = await search_knowledge_graph(
            "What genes are associated with Alzheimer's disease?"
        )
    """
    return await gretriever_tool(query=query)


# ==============================================================================
# SCIENTIST RAG TOOLS
# ==============================================================================

@mcp.tool()
async def query_scientist_knowledge(
    author_name: str,
    question: str
) -> str:
    """
    Query RAG system built from specific scientist's publications.
    
    This tool:
    1. Retrieves publications by the specified author
    2. Builds a specialized knowledge base from their work
    3. Answers questions using RAG over their publications
    
    Useful for:
    - Understanding a researcher's specific contributions
    - Finding methodologies used by an author
    - Contextualizing findings within an author's body of work
    
    Args:
        author_name: Full name of the scientist/author (e.g., "Eric Lander")
        question: Question to ask about their work
    
    Returns:
        Answer based on the author's publications with citations
    
    Example:
        result = await query_scientist_knowledge(
            author_name="Jennifer Doudna",
            question="What are her main contributions to CRISPR technology?"
        )
    """
    return await scientist_rag_tool_wrapper(
        author_name=author_name,
        question=question
    )


# ==============================================================================
# OMICS ANALYSIS TOOLS
# ==============================================================================

@mcp.tool()
async def analyze_omics_data(
    query: str,
    session_id: Optional[str] = None
) -> str:
    """
    Perform comprehensive single-cell omics data analysis workflow.
    
    This tool provides end-to-end analysis:
    1. **Named Entity Recognition (NER)**: Extracts disease, cell type, tissue from query
    2. **Data Retrieval**: Fetches relevant single-cell RNA-seq data from OmniCellTOSG database
    3. **Differential Expression Analysis**: Identifies significantly changed genes
    4. **Pathway Enrichment**: Performs KEGG pathway analysis
    5. **Visualization**: Generates volcano plots and enrichment plots
    6. **Results Summary**: Returns top genes, pathways, and analysis metrics
    
    The tool automatically:
    - Handles data matching with fuzzy search
    - Performs statistical analysis (DESeq2)
    - Creates publication-quality plots
    - Saves all results to session directory
    
    Args:
        query: Natural language query (e.g., "Analyze lung adenocarcinoma single-cell data")
        session_id: Optional session identifier (default: auto-generated)
    
    Returns:
        Comprehensive analysis report including:
        - Top differentially expressed genes
        - Enriched pathways
        - Sample statistics
        - Paths to generated plots
    
    Example:
        result = await analyze_omics_data(
            query="Find key genes in pancreatic ductal adenocarcinoma",
            session_id="pdac_analysis_001"
        )
    
    Note:
        - Requires OmniCellTOSG database to be installed
        - Generates plots saved to webapp/sessions/<session_id>/
        - Analysis can take 5-15 minutes for large datasets
    """
    if session_id is None:
        import uuid
        session_id = f"mcp_omics_{uuid.uuid4().hex[:8]}"
    
    # Run the workflow
    result = await asyncio.to_thread(
        omic_fetch_analysis_workflow,
        query=query,
        session_id=session_id
    )
    
    return result


# ==============================================================================
# SERVER EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Run the MCP server
    # Clients can connect via stdio or SSE
    mcp.run()
