#!/usr/bin/env python3
"""
LangGraph-based OmniCellAgent for biomedical research.

This module implements a hierarchical agent system using LangGraph with:
1. A planning phase to break down user queries
2. Sub-agent execution for specialized tasks
3. Re-planning capability when tasks fail
4. Comprehensive reporting with detailed process information
"""

import asyncio
import argparse
import getpass
import json
import os
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, Literal, TypedDict, Annotated
from dotenv import load_dotenv
load_dotenv(".env")

# Ensure Google API key is set
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

import sys
# Add project root to Python path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import path config
from utils.path_config import get_path

# Suppress INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("paperscraper").setLevel(logging.ERROR)

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
import operator

# Global session directory for tools (set when agent is initialized)
_GLOBAL_SESSION_DIR: Optional[str] = None
_GLOBAL_SESSION_ID: Optional[str] = None

# Import tool functions directly (these are clean async functions without AutoGen dependencies)
from tools.pubmed_tools.query_pubmed_tool import query_medical_research_async
from tools.google_search_tools.google_search_w3m import google_search, web_search_tool
# Lite PubMed search (no PDF download, just metadata + abstracts)
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from tools.scientist_rag_tools.scientist_tool import scientist_rag_tool_wrapper
from tools.gretriever_tools.gretriever_client import gretriever_tool

# Add omic_tools to path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools', 'omic_tools'))
from omic_fetch_analysis_workflow import omic_fetch_analysis_workflow as _omic_workflow

# Import prompts
from utils.prompt import (
    MAGNETIC_ONE_ORCHESTRATOR_PROMPT, 
    SEARCH_AGENT_SYSTEM_MESSAGE_v1, 
    PUBMED_AGENT_SYSTEM_MESSAGE_v1
)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def extract_text_from_llm_response(content) -> str:
    """
    Extract text from LLM response content.
    Handles Gemini's various response formats:
    - String: returned as-is
    - List of dicts with 'text' key: extracts and joins text
    - List of strings: joins them
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and 'text' in part:
                texts.append(part['text'])
            elif isinstance(part, str):
                texts.append(part)
            else:
                texts.append(str(part))
        return "".join(texts)
    return str(content)


# ==============================================================================
# STATE DEFINITIONS
# ==============================================================================

# Pydantic models for structured LLM output (more robust than JSON parsing)
from pydantic import BaseModel, Field
from typing import List as PyList

class PlannedTask(BaseModel):
    """A single task in the research plan"""
    id: str = Field(description="Unique task identifier like task_1, task_2, etc.")
    description: str = Field(description="Detailed description of what this task should accomplish")
    assigned_agent: str = Field(description="Agent to execute this task: OmicMiningAgent, BioMarkerKGAgent, PubMedResearcher, ScientistRAGExpert, or GoogleSearcher")

class ResearchPlan(BaseModel):
    """Structured research plan output"""
    analysis: str = Field(description="Brief analysis of the query and research strategy")
    tasks: PyList[PlannedTask] = Field(description="List of tasks to execute in order")


class SubTask(TypedDict):
    """Represents a single sub-task in the plan"""
    id: str
    description: str
    assigned_agent: str
    status: Literal["pending", "in_progress", "completed", "failed"]
    result: Optional[str]
    error: Optional[str]
    attempts: int


class AgentState(TypedDict):
    """Main state for the LangGraph agent system"""
    # User input
    query: str
    session_id: str
    
    # Planning state
    plan: List[SubTask]
    current_task_index: int
    plan_revision_count: int
    max_plan_revisions: int
    
    # Execution state
    messages: Annotated[List[BaseMessage], operator.add]
    agent_outputs: Dict[str, Any]
    
    # Structured data for cross-agent sharing (programmatic, not via LLM parsing)
    shared_data: Dict[str, Any]  # Keys: 'top_genes', 'paper_dois', 'pathways', etc.
    
    # Process tracking for comprehensive reporting
    process_log: List[Dict[str, Any]]
    
    # Final output
    final_report: Optional[str]
    status: Literal["planning", "executing", "replanning", "reporting", "completed", "failed"]


# ==============================================================================
# TOOL WRAPPERS FOR LANGCHAIN
# ==============================================================================

def create_session_dir(session_id: str) -> str:
    """Create and return session directory path"""
    global _GLOBAL_SESSION_DIR, _GLOBAL_SESSION_ID
    sessions_base = get_path('sessions.base', absolute=True, create=True)
    session_dir = os.path.join(sessions_base, session_id)
    os.makedirs(session_dir, exist_ok=True)
    _GLOBAL_SESSION_DIR = session_dir
    _GLOBAL_SESSION_ID = session_id
    return session_dir


def get_current_session_dir() -> str:
    """Get the current session directory, creating a default if not set"""
    global _GLOBAL_SESSION_DIR, _GLOBAL_SESSION_ID
    if _GLOBAL_SESSION_DIR is None:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        _GLOBAL_SESSION_DIR = create_session_dir(session_id)
    return _GLOBAL_SESSION_DIR


def get_current_session_id() -> str:
    """Get the current session ID"""
    global _GLOBAL_SESSION_ID
    if _GLOBAL_SESSION_ID is None:
        # Trigger session creation
        get_current_session_dir()
    return _GLOBAL_SESSION_ID


@tool
def omic_analysis_tool(
    disease: Optional[str] = None,
    cell_type: Optional[str] = None,
    organ: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform multi-omics analysis: gene expression profiling, differential expression, and pathway enrichment.
    
    Args:
        disease: Disease name to analyze. Examples: "lung adenocarcinoma", "Alzheimer disease", "breast cancer"
        cell_type: Optional cell type filter. Examples: "microglial cell", "T cell", "acinar cell"
        organ: Organ filter (IMPORTANT for memory efficiency). Examples: "lung", "brain", "breast", "pancreas"
        
    Returns:
        Dict with analysis results including top genes, enrichment data, and plot paths.
    """
    try:
        # Use the global session directory
        session_dir = get_current_session_dir()
        
        params = {
            "session_dir": session_dir,
            "enable_differential_expression": True,
            "enable_plotting": True
        }
        
        if disease:
            params["disease"] = disease
        if cell_type:
            params["cell_type"] = cell_type
        if organ:
            params["organ"] = organ
            
        if not disease and not cell_type:
            return {"success": False, "message": "Provide at least 'disease' or 'cell_type'"}
        
        result = _omic_workflow(**params)
        return result
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "message": f"Error in omic analysis: {str(e)}",
            "traceback": traceback.format_exc()
        }


@tool
async def pubmed_search_tool(query: str) -> str:
    """
    Search PubMed for biomedical literature and research papers.
    Automatically detects gene names and performs dual searches for better coverage:
    1. Original query (e.g., "EGFR lung cancer")
    2. Gene-specific exact match search (e.g., "EGFR[Title/Abstract]")
    
    Returns FULL paper content without truncation for comprehensive analysis.
    Includes proper citations for referencing in reports.
    
    Args:
        query: Search query for PubMed
        
    Returns:
        Search results with FULL paper content and formatted citations
    """
    import re
    
    try:
        # Use the session ID to group papers from the same analysis session
        session_id = get_current_session_id()
        
        # Detect if query contains a gene name pattern (uppercase letters/numbers, 2-10 chars)
        # Common gene name patterns: EGFR, TP53, BRCA1, HER2, etc.
        gene_pattern = r'\b([A-Z][A-Z0-9]{1,9})\b'
        potential_genes = re.findall(gene_pattern, query)
        
        # Filter out common non-gene words
        non_gene_words = {'AND', 'OR', 'NOT', 'THE', 'FOR', 'WITH', 'FROM', 'INTO', 'THAT', 'THIS', 
                         'ARE', 'WAS', 'WERE', 'BEEN', 'HAVE', 'HAS', 'HAD', 'DOES', 'DID', 'WILL',
                         'DNA', 'RNA', 'PCR', 'USA', 'NHS', 'WHO', 'FDA', 'NIH', 'CDC'}
        genes = [g for g in potential_genes if g not in non_gene_words and len(g) >= 2]
        
        all_papers = []
        queries_run = []
        
        # First search: Original query
        print(f"[PubMed] Search 1: '{query}'")
        queries_run.append(query)
        papers1 = await query_medical_research_async(
            query=query,
            top_k=5,  # Reduced since we're doing multiple searches
            use_llm_processing=False,  # Disabled for speed
            max_concurrent=10,
            session_id=session_id
        )
        if isinstance(papers1, list):
            all_papers.extend(papers1)
        
        # Second search: Gene-specific exact match (if gene detected)
        if genes:
            gene = genes[0]  # Use first detected gene
            gene_query = f'"{gene}"[Title/Abstract]'
            print(f"[PubMed] Search 2 (gene exact match): '{gene_query}'")
            queries_run.append(gene_query)
            papers2 = await query_medical_research_async(
                query=gene_query,
                top_k=5,
                use_llm_processing=False,  # Disabled for speed
                max_concurrent=10,
                session_id=session_id
            )
            if isinstance(papers2, list):
                all_papers.extend(papers2)
        
        # Deduplicate papers by title - keep ALL papers (no content filtering)
        seen_titles = set()
        unique_papers = []
        for paper in all_papers:
            title = paper.get('title', '').lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_papers.append(paper)
        
        if not unique_papers:
            return f"No papers found for queries: {queries_run}"
        
        # Format the results with FULL content and citations
        result_lines = [
            f"# PubMed Search Results",
            f"**Queries executed:** {queries_run}",
            f"**Total unique papers found:** {len(unique_papers)}",
            "",
            "=" * 80,
            "",
            "## IMPORTANT: Include Citations in Reports",
            "When using information from these papers, always cite the source.",
            "",
            "=" * 80
        ]
        
        for i, paper in enumerate(unique_papers, 1):
            # Format authors
            authors = paper.get('authors', [])
            if isinstance(authors, list) and authors:
                if len(authors) > 3:
                    author_str = f"{authors[0]} et al."
                else:
                    author_str = ", ".join(str(a) for a in authors)
            else:
                author_str = "Unknown authors"
            
            # Get metadata
            doi = paper.get('doi', '')
            pmid = paper.get('pmid', '')
            journal = paper.get('journal', '')
            year = paper.get('year', paper.get('date', ''))
            title = paper.get('title', 'N/A')
            
            # Build citation
            citation_parts = [author_str]
            if year:
                citation_parts.append(f"({year})")
            citation_parts.append(f'"{title}"')
            if journal:
                citation_parts.append(journal)
            if doi:
                citation_parts.append(f"DOI: {doi}")
            elif pmid:
                citation_parts.append(f"PMID: {pmid}")
            citation = " ".join(citation_parts)
            
            result_lines.append(f"\n## Paper [{i}]")
            result_lines.append(f"**Title:** {title}")
            result_lines.append(f"**Authors:** {author_str}")
            
            meta = []
            if doi:
                meta.append(f"DOI: {doi}")
            if pmid:
                meta.append(f"PMID: {pmid}")
            if journal:
                meta.append(f"Journal: {journal}")
            if year:
                meta.append(f"Year: {year}")
            if meta:
                result_lines.append("**Metadata:** " + " | ".join(meta))
            
            result_lines.append(f"**Citation:** {citation}")
            
            # Add FULL LLM-processed content (NO truncation)
            llm_content = paper.get('llm_content') or paper.get('abstract') or ''
            if llm_content:
                result_lines.append(f"\n**Full Content:**\n{llm_content}")
            else:
                result_lines.append("\n**Content:** [Title only - paper content not yet extracted]")
            
            result_lines.append("\n" + "-" * 60)
        
        # Quick reference section
        result_lines.append("\n## Quick Reference - All Citations")
        for i, paper in enumerate(unique_papers, 1):
            authors = paper.get('authors', [])
            if isinstance(authors, list) and authors:
                author_str = f"{authors[0]} et al." if len(authors) > 3 else ", ".join(str(a) for a in authors)
            else:
                author_str = "Unknown"
            year = paper.get('year', paper.get('date', ''))
            doi = paper.get('doi', '')
            pmid = paper.get('pmid', '')
            ref = f"[{i}] {author_str}"
            if year:
                ref += f" ({year})"
            ref += f' "{paper.get("title", "N/A")}"'
            if doi:
                ref += f" DOI: {doi}"
            elif pmid:
                ref += f" PMID: {pmid}"
            result_lines.append(ref)
        
        return "\n".join(result_lines)
        
    except Exception as e:
        import traceback
        return f"Error searching PubMed: {str(e)}\n{traceback.format_exc()}"


# Create lite PubMed tool instance (no PDF download, just metadata + abstracts from NCBI)
_pubmed_lite = PubmedQueryRun()

@tool
def pubmed_lite_tool(query: str) -> str:
    """
    Lightweight PubMed search - returns abstracts and metadata only (NO PDF download).
    Much faster than full pubmed_search_tool but doesn't provide full paper content.
    Use this for quick literature overview or when PDF download is not needed.
    
    Args:
        query: Search query for PubMed
        
    Returns:
        Paper abstracts and metadata from PubMed
    """
    try:
        return _pubmed_lite.invoke(query)
    except Exception as e:
        return f"Error in lite PubMed search: {str(e)}"


@tool
async def google_search_tool_wrapper(query: str) -> str:
    """
    Perform a Google web search for general information and current research.
    
    Args:
        query: Search query
        
    Returns:
        Search results with summaries
    """
    try:
        result = await web_search_tool(query)
        return str(result)
    except Exception as e:
        return f"Error in Google search: {str(e)}"


@tool
async def curated_pubmed_tool(genes: Optional[List[str]] = None, disease: Optional[str] = None, max_papers_per_query: int = 5) -> str:
    """
    Curated PubMed retrieval optimized for gene-centric literature curation.
    Returns FULL paper content without truncation to enable comprehensive analysis.
    
    IMPORTANT: This tool returns complete paper information including:
    - Full citation details (DOI, PMID, authors, journal, year)
    - Complete LLM-processed content (NOT truncated)
    - All papers are returned without premature filtering
    
    The agent should read ALL returned content before deciding relevance.

    Args:
        genes: Optional list of gene symbols to search for (e.g., ['EGFR', 'TP53'])
        disease: Optional disease string to combine with gene searches (e.g., 'lung cancer')
        max_papers_per_query: number of papers to retrieve per query

    Returns:
        A comprehensive summary (string) containing ALL papers with FULL content and citations.
    """
    try:
        session_id = get_current_session_id()
        all_papers = []

        queries = []
        # If genes provided, run two searches per gene: gene+disease and gene exact match
        if genes:
            for gene in genes:
                if disease:
                    q1 = f"{gene} {disease}"
                    queries.append(q1)
                    papers1 = await query_medical_research_async(query=q1, top_k=max_papers_per_query, use_llm_processing=False, max_concurrent=6, session_id=session_id)
                    if isinstance(papers1, list):
                        all_papers.extend(papers1)

                # gene exact match
                q2 = f'"{gene}"[Title/Abstract]'
                queries.append(q2)
                papers2 = await query_medical_research_async(query=q2, top_k=max_papers_per_query, use_llm_processing=False, max_concurrent=6, session_id=session_id)
                if isinstance(papers2, list):
                    all_papers.extend(papers2)

        # If no genes provided, fallback to disease-level search
        if not genes and disease:
            queries.append(disease)
            papers = await query_medical_research_async(query=disease, top_k=max_papers_per_query, use_llm_processing=False, max_concurrent=6, session_id=session_id)
            if isinstance(papers, list):
                all_papers.extend(papers)

        # Deduplicate by DOI or title - but keep ALL papers (no content filtering)
        seen = set()
        curated = []
        for p in all_papers:
            # Create unique key from DOI, PMID, or title
            doi = p.get('doi', '')
            pmid = p.get('pmid', '')
            title = p.get('title', '').lower().strip()
            key = doi or pmid or title
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            
            # Build a COMPLETE entry - NO truncation
            # Format authors for citation
            authors = p.get('authors', [])
            if isinstance(authors, list) and authors:
                if len(authors) > 3:
                    author_str = f"{authors[0]} et al."
                else:
                    author_str = ", ".join(authors)
            else:
                author_str = "Unknown authors"
            
            entry = {
                'title': p.get('title', 'N/A'),
                'authors': author_str,
                'doi': doi,
                'pmid': pmid,
                'journal': p.get('journal', ''),
                'year': p.get('year', p.get('date', '')),
                'content': None,  # Full content, not summary
                'citation': None  # Formatted citation for reports
            }
            
            # Keep FULL content - NO truncation
            llm_content = p.get('llm_content') or p.get('abstract') or ''
            if llm_content:
                entry['content'] = llm_content  # Full content preserved
            
            # Generate formatted citation for use in reports
            citation_parts = []
            citation_parts.append(author_str)
            if entry['year']:
                citation_parts.append(f"({entry['year']})")
            citation_parts.append(f'"{entry["title"]}"')
            if entry['journal']:
                citation_parts.append(entry['journal'])
            if entry['doi']:
                citation_parts.append(f"DOI: {entry['doi']}")
            elif entry['pmid']:
                citation_parts.append(f"PMID: {entry['pmid']}")
            entry['citation'] = " ".join(citation_parts)
            
            curated.append(entry)

        # Format output with FULL content and citations
        lines = [
            f"# PubMed Literature Results",
            f"**Queries executed:** {queries}",
            f"**Total unique papers found:** {len(curated)}",
            "",
            "=" * 80,
            "",
            "## IMPORTANT: Citation Format for Reports",
            "When referencing findings in your report, use the citation provided for each paper.",
            "Example: 'EGFR mutations are associated with... (Smith et al., 2023, DOI: 10.1234/example)'",
            "",
            "=" * 80
        ]
        
        if not curated:
            lines.append("\nNo papers found for the provided genes/disease.")
            return "\n".join(lines)

        for i, e in enumerate(curated, 1):
            lines.append(f"\n## Paper [{i}]")
            lines.append(f"**Title:** {e['title']}")
            lines.append(f"**Authors:** {e['authors']}")
            
            # Metadata line
            meta = []
            if e['doi']:
                meta.append(f"DOI: {e['doi']}")
            if e['pmid']:
                meta.append(f"PMID: {e['pmid']}")
            if e['journal']:
                meta.append(f"Journal: {e['journal']}")
            if e['year']:
                meta.append(f"Year: {e['year']}")
            if meta:
                lines.append("**Metadata:** " + " | ".join(meta))
            
            # Citation for use in reports
            lines.append(f"**Citation:** {e['citation']}")
            
            # FULL content (not truncated)
            if e['content']:
                lines.append(f"\n**Full Content:**\n{e['content']}")
            else:
                lines.append("\n**Content:** [Paper downloaded but content not extracted - check PDF/XML directly]")
            
            lines.append("\n" + "-" * 60)

        # Add summary section for easy reference
        lines.append("\n## Quick Reference - All Citations")
        for i, e in enumerate(curated, 1):
            lines.append(f"[{i}] {e['citation']}")
        
        # Add structured DOI list for programmatic access by downstream agents
        all_dois = [e['doi'] for e in curated if e.get('doi')]
        all_pmids = [e['pmid'] for e in curated if e.get('pmid')]
        if all_dois or all_pmids:
            lines.append("\n## Structured Paper Identifiers (for programmatic use)")
            if all_dois:
                lines.append(f"DOIs: {all_dois}")
            if all_pmids:
                lines.append(f"PMIDs: {all_pmids}")

        return "\n".join(lines)
    except Exception as ex:
        import traceback
        return f"Error in curated_pubmed_tool: {str(ex)}\n{traceback.format_exc()}"


@tool
async def scientist_rag_tool(query: str) -> str:
    """
    Query the scientific knowledge base using RAG retrieval for expert scientific information.
    
    Args:
        query: Query for the scientific knowledge base (use 3-5 words)
        
    Returns:
        Retrieved scientific knowledge
    """
    try:
        result = await scientist_rag_tool_wrapper(query)
        return str(result)
    except Exception as e:
        return f"Error in scientist RAG: {str(e)}"


@tool
async def biomarker_kg_tool(query: str) -> str:
    """
    Query the PrimeKG knowledge graph for biomedical concepts and relationships.
    
    Args:
        query: Query about biological processes and relationships
        
    Returns:
        Knowledge graph results
    """
    try:
        result = await gretriever_tool(query)
        return str(result)
    except Exception as e:
        return f"Error in KG query: {str(e)}"


# ==============================================================================
# AGENT DEFINITIONS
# ==============================================================================

def create_llm(model_name: str = "gemini-3-pro-preview"):
    """
    Create an LLM instance based on the model name.
    Supports Google Gemini models by default.
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        LLM instance (type-agnostic)
    """
    return ChatGoogleGenerativeAI(model=model_name, temperature=0)


class SubAgent:
    """Base class for specialized sub-agents"""
    
    def __init__(self, name: str, description: str, system_message: str, tools: List, llm=None, model_name: str = "gemini-3-pro-preview"):
        self.name = name
        self.description = description
        self.system_message = system_message
        self.tools = tools
        # Accept any LLM instance or create one
        self.llm = llm if llm is not None else create_llm(model_name)
        if tools:
            self.llm_with_tools = self.llm.bind_tools(tools)
        else:
            self.llm_with_tools = self.llm
    
    async def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a task and return results"""
        messages = [
            SystemMessage(content=self.system_message),
            HumanMessage(content=f"Task: {task}\n\nContext: {json.dumps(context or {}, indent=2)}")
        ]
        
        try:
            # Initial response
            response = await self.llm_with_tools.ainvoke(messages)
            messages.append(response)
            
            # Handle tool calls if present
            max_iterations = 5
            iteration = 0
            
            while response.tool_calls and iteration < max_iterations:
                # Execute tool calls
                tool_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    # Find and execute the tool
                    tool_result = await self._execute_tool(tool_name, tool_args)
                    tool_results.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    ))
                
                messages.extend(tool_results)
                
                # Get next response
                response = await self.llm_with_tools.ainvoke(messages)
                messages.append(response)
                iteration += 1
            
            return {
                "success": True,
                "result": extract_text_from_llm_response(response.content),
                "iterations": iteration
            }
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def _execute_tool(self, tool_name: str, tool_args: Dict) -> Any:
        """Execute a specific tool by name"""
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    # Always use ainvoke for consistency with async execution
                    return await tool.ainvoke(tool_args)
                except Exception as e:
                    # Fallback to sync invoke if async fails
                    try:
                        return tool.invoke(tool_args)
                    except Exception as e2:
                        return f"Tool execution error: {str(e2)}"
        return f"Tool {tool_name} not found"


# ==============================================================================
# LANGGRAPH AGENT SYSTEM
# ==============================================================================

class LangGraphOmniCellAgent:
    """
    LangGraph-based OmniCellAgent with planning, sub-agent execution, re-planning, and reporting.
    """
    
    def __init__(self, model_name: str = "gemini-3-pro-preview", log_dir: str = None, session_id: str = None, llm=None):
        """
        Initialize the LangGraph agent system.
        
        Args:
            model_name: Name of the LLM model to use (default: gemini-2.0-flash)
            log_dir: Directory to save logs
            session_id: Unique session identifier
            llm: Optional pre-configured LLM instance (any type)
        """
        if log_dir is None:
            log_dir = get_path('logs.base', absolute=True, create=True)
        
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        self.session_id = session_id
        self.log_dir = log_dir
        self.model_name = model_name
        
        # Create the session directory and set it globally for tools
        self.session_dir = create_session_dir(self.session_id)
        
        print(f"🔧 Initializing LangGraph OmniCellAgent")
        print(f"📁 Session ID: {self.session_id}")
        print(f"📂 Session Directory: {self.session_dir}")
        
        # Initialize LLM - accept any LLM instance or create one
        self.llm = llm if llm is not None else create_llm(model_name)
        
        # Initialize sub-agents
        self._init_sub_agents()
        
        # Build the graph
        self.graph = self._build_graph()
        
    def _init_sub_agents(self):
        """Initialize all sub-agents"""
        
        self.sub_agents = {
            "OmicMiningAgent": SubAgent(
                name="OmicMiningAgent",
                description="Analyzes omics data for gene expression, biomarkers, and pathway enrichment",
                system_message="""You are an omics data specialist. Your task is to answer gene/biomarker questions using the omic_analysis tool.

## STEP 1: EXTRACT PARAMETERS FROM THE QUERY
From the user's question, identify:
- Disease name (MUST match the variations list below)
- Organ (REQUIRED for efficiency)
- Cell type (OPTIONAL, if mentioned)

## DISEASE NAME VARIATIONS (use EXACT names):
- Alzheimer's → "Alzheimer disease"
- Lung cancer → "lung adenocarcinoma"
- Breast cancer → "breast cancer"
- Colon cancer → "colorectal cancer"
- Pancreatic cancer → "pancreatic ductal adenocarcinoma"

## ORGAN MAPPING (use EXACT names):
- Alzheimer's → "brain"
- Lung cancer → "lung"
- Breast cancer → "breast"

After tool calls complete, write a concise summary of the gene list and pathway data.""",
                tools=[omic_analysis_tool],
                llm=self.llm
            ),
            
            "PubMedResearcher": SubAgent(
                name="PubMedResearcher",
                description="Searches biomedical literature using PubMed and returns curated summaries with citations",
                system_message="""You are a biomedical literature specialist. Search PubMed for relevant papers and provide comprehensive summaries WITH CITATIONS.

## KEY INSTRUCTIONS:
1. Check context["top_genes"] for genes from previous analysis - search ALL of them (up to 20 genes)
2. For each gene, search: "[GENE] [disease]" to find relevant literature
3. Every claim MUST include citation: (Author et al., Year, DOI/PMID)
4. Provide comprehensive coverage - multiple searches are encouraged

## OUTPUT FORMAT per gene:
### [GENE_SYMBOL]
- **Function**: What it does (cited)
- **Disease Role**: Relevance to condition (cited)  
- **Key Finding**: Most important result (cited)

## CITATION FORMAT:
"EGFR mutations occur in 15% of cases (Lynch et al., 2004, DOI: 10.1056/NEJMoa040938)"

Include a References section at the end listing all papers.
""",
                # Use only lite tool for now (no PDF download)
                # tools=[curated_pubmed_tool, pubmed_search_tool, pubmed_lite_tool],  # Full version with PDF download
                tools=[pubmed_lite_tool],
                llm=self.llm
            ),
            
            "GoogleSearcher": SubAgent(
                name="GoogleSearcher",
                description="Performs general web searches for current information",
                system_message=SEARCH_AGENT_SYSTEM_MESSAGE_v1,
                tools=[google_search_tool_wrapper],
                llm=self.llm
            ),
            
            "ScientistRAGExpert": SubAgent(
                name="ScientistRAGExpert",
                description="Queries scientific knowledge base using RAG",
                system_message="""You are a scientific knowledge expert with access to a comprehensive 
scientific knowledge base. Your role is to:
1. Query your tool using a concise phrase (3-5 words).
2. Provide expert analysis and insights from scientific literature
3. Synthesize information from multiple scientific sources""",
                tools=[scientist_rag_tool],
                llm=self.llm
            ),
            
            "BioMarkerKGAgent": SubAgent(
                name="BioMarkerKGAgent",
                description="Queries PrimeKG knowledge graph for biomedical relationships",
                system_message="""You are a PrimeKG agent that queries the PrimeKG knowledge graph to retrieve relevant biomedical information.
You will provide structured data and relationships from the knowledge graph to enhance understanding of biomedical concepts.""",
                tools=[biomarker_kg_tool],
                llm=self.llm
            )
        }
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine"""
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("planner", self._planning_node)
        workflow.add_node("executor", self._execution_node)
        workflow.add_node("replanner", self._replanning_node)
        workflow.add_node("reporter", self._reporting_node)
        
        # Add edges
        workflow.add_edge(START, "planner")
        workflow.add_conditional_edges(
            "planner",
            self._should_execute_or_report,
            {
                "execute": "executor",
                "report": "reporter"
            }
        )
        workflow.add_conditional_edges(
            "executor",
            self._should_continue_or_replan,
            {
                "continue": "executor",
                "replan": "replanner",
                "report": "reporter"
            }
        )
        workflow.add_conditional_edges(
            "replanner",
            self._after_replan,
            {
                "execute": "executor",
                "report": "reporter"
            }
        )
        workflow.add_edge("reporter", END)
        
        return workflow.compile()
    
    async def _planning_node(self, state: AgentState) -> Dict[str, Any]:
        """Create initial plan for the query using structured output"""
        print("\n📋 PLANNING PHASE")
        print("=" * 60)
        
        planning_prompt = f"""Create a comprehensive research plan for hypothesis-driven discovery:

Query: {state['query']}

**Goal**: Generate thorough Gene→Pathway→Phenotype grounded hypotheses with literature validation.

**Available Agents & Capabilities:**
1. **OmicMiningAgent** - Differential expression analysis 
   - Returns: sample counts, DEG rankings, log2FC, p-values, volcano plots
   - ALWAYS run FIRST to ground analysis in actual expression data
   
2. **BioMarkerKGAgent** - Knowledge graph traversal for gene neighbors
   - Returns: drug interactions, pathway memberships, GO terms, protein interactors
   - Query for TOP 10-15 DEGs to build gene-pathway-phenotype chains
   
3. **PubMedResearcher** - Literature search & citation mining
   - Returns: relevant publications, existing hypotheses, experimental evidence
   - Search BOTH: (a) disease-specific papers AND (b) gene-mechanism papers
   - Helps classify hypotheses as confirmatory vs novel
   
4. **ScientistRAGExpert** - Hypothesis generation & mechanism synthesis  
   - Returns: mechanistic hypotheses, network descriptions, validation strategies
   - Synthesizes omics + KG + literature into testable hypotheses

5. **GoogleSearcher** - Clinical/translational context
   - Returns: clinical trials, recent developments, therapeutic landscape
   - Grounds hypotheses in translational relevance

**Mandatory Workflow (for hypothesis-quality reports):**
- Step 1: OmicMiningAgent → Get DEGs with statistics (establishes data foundation)
- Step 2: BioMarkerKGAgent → Query top 10-15 DEGs for KG neighbors (builds gene→pathway chains)
- Step 3: PubMedResearcher → Literature on TOP targets AND pathways (validates & classifies novelty)
- Step 4: ScientistRAGExpert → Synthesize mechanisms, score hypotheses, propose experiments

**Planning Principles:**
- Be THOROUGH: Query KG for multiple gene sets (up-regulated, down-regulated, top-ranked)
- Be GROUNDED: Every hypothesis must trace to specific gene(s) in the omics data
- Be NOVEL-SEEKING: Literature search should identify what is KNOWN vs what is NEW
- Be ACTIONABLE: Plan should enable proposing specific validation experiments"""
        
        # Try structured output first (more robust)
        try:
            structured_llm = self.llm.with_structured_output(ResearchPlan)
            plan_result: ResearchPlan = await structured_llm.ainvoke([
                SystemMessage(content="""You are an expert research planner for biomedical hypothesis discovery. 
Create thorough plans that:
1. Ground ALL hypotheses in actual omics data
2. Build explicit Gene→Pathway→Phenotype chains via knowledge graphs
3. Validate novelty through comprehensive literature search
4. Enable specific, testable experimental proposals"""),
                HumanMessage(content=planning_prompt)
            ])
            
            # Convert Pydantic model to SubTask list
            tasks = []
            for planned_task in plan_result.tasks:
                tasks.append(SubTask(
                    id=planned_task.id,
                    description=planned_task.description,
                    assigned_agent=planned_task.assigned_agent,
                    status="pending",
                    result=None,
                    error=None,
                    attempts=0
                ))
            
            print(f"📝 Created plan with {len(tasks)} tasks (structured output):")
            for task in tasks:
                print(f"   - {task['id']}: {task['description'][:50]}... ({task['assigned_agent']})")
            
            # Log the planning
            process_entry = {
                "phase": "planning",
                "timestamp": datetime.now().isoformat(),
                "analysis": plan_result.analysis,
                "tasks_created": len(tasks)
            }
            
            return {
                "plan": tasks,
                "current_task_index": 0,
                "status": "executing",
                "process_log": state.get("process_log", []) + [process_entry],
                "messages": [AIMessage(content=f"Plan created with {len(tasks)} tasks")]
            }
            
        except Exception as e:
            print(f"⚠️ Structured output failed: {e}, falling back to JSON parsing")
            
            # Fallback to JSON parsing
            try:
                response = await self.llm.ainvoke([
                    SystemMessage(content="You are a research planner. Output ONLY valid JSON, no markdown."),
                    HumanMessage(content=planning_prompt + '\n\nOutput JSON: {"analysis": "...", "tasks": [{"id": "task_1", "description": "...", "assigned_agent": "AgentName"}]}')
                ])
                
                content = extract_text_from_llm_response(response.content)
                
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                plan_data = json.loads(content)
            
                tasks = []
                for task in plan_data.get("tasks", []):
                    tasks.append(SubTask(
                        id=task["id"],
                        description=task["description"],
                        assigned_agent=task["assigned_agent"],
                        status="pending",
                        result=None,
                        error=None,
                        attempts=0
                    ))
                
                print(f"📝 Created plan with {len(tasks)} tasks (JSON fallback):")
                for task in tasks:
                    print(f"   - {task['id']}: {task['description'][:50]}... ({task['assigned_agent']})")
                
                # Log the planning
                process_entry = {
                    "phase": "planning",
                    "timestamp": datetime.now().isoformat(),
                    "analysis": plan_data.get("analysis", ""),
                    "tasks_created": len(tasks)
                }
                
                return {
                    "plan": tasks,
                    "current_task_index": 0,
                    "status": "executing",
                    "process_log": state.get("process_log", []) + [process_entry],
                    "messages": [AIMessage(content=f"Plan created with {len(tasks)} tasks")]
                }
                
            except (json.JSONDecodeError, KeyError) as json_err:
                print(f"⚠️ JSON parsing also failed: {json_err}")
                # Fall through to default plan below
        
        # Default 4-step workflow (used when both structured and JSON parsing fail)
        print("📝 Using default 4-step research plan")
        default_tasks = [
            SubTask(
                id="task_1",
                description=f"Perform differential expression analysis for: {state['query']}",
                assigned_agent="OmicMiningAgent",
                status="pending",
                result=None,
                error=None,
                attempts=0
            ),
            SubTask(
                id="task_2",
                description="Query knowledge graph for gene neighbors (drugs, pathways, GO terms) of top DEGs",
                assigned_agent="BioMarkerKGAgent",
                status="pending",
                result=None,
                error=None,
                attempts=0
            ),
            SubTask(
                id="task_3",
                description="Literature search for top DEGs and disease targets",
                assigned_agent="PubMedResearcher",
                status="pending",
                result=None,
                error=None,
                attempts=0
            ),
            SubTask(
                id="task_4",
                description="Synthesize findings and generate mechanistic hypotheses",
                assigned_agent="ScientistRAGExpert",
                status="pending",
                result=None,
                error=None,
                attempts=0
            )
        ]
        
        print(f"📝 Created default plan with {len(default_tasks)} tasks:")
        for task in default_tasks:
            print(f"   - {task['id']}: {task['description'][:50]}... ({task['assigned_agent']})")
        
        return {
            "plan": default_tasks,
            "current_task_index": 0,
            "status": "executing",
            "process_log": state.get("process_log", []) + [{"phase": "planning", "fallback": "default_4step"}],
            "messages": [AIMessage(content="Created default 4-step workflow")]
        }
    
    async def _execution_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute the current task in the plan"""
        print("\n🔄 EXECUTION PHASE")
        print("=" * 60)
        
        plan = state["plan"]
        current_idx = state["current_task_index"]
        
        if current_idx >= len(plan):
            return {"status": "reporting"}
        
        current_task = plan[current_idx]
        agent_name = current_task["assigned_agent"]
        
        print(f"📌 Executing task {current_idx + 1}/{len(plan)}: {current_task['description'][:80]}...")
        print(f"   Agent: {agent_name}")
        
        # Update task status
        current_task["status"] = "in_progress"
        current_task["attempts"] += 1
        
        # Get the sub-agent
        if agent_name not in self.sub_agents:
            print(f"⚠️ Unknown agent: {agent_name}, using GoogleSearcher")
            agent_name = "GoogleSearcher"
        
        sub_agent = self.sub_agents[agent_name]
        
        # Prepare context from previous task results, including structured shared_data
        shared_data = state.get("shared_data", {})
        context = {
            "query": state["query"],
            "session_id": state["session_id"],
            "previous_results": state.get("agent_outputs", {}),
            # Structured data for programmatic access by downstream agents
            "shared_data": shared_data,
            "top_genes": shared_data.get("top_genes", []),  # Explicit gene list
            "paper_dois": shared_data.get("paper_dois", []),  # Explicit DOI list
        }
        
        # Execute the task
        result = await sub_agent.execute(current_task["description"], context)
        
        # Process result
        process_entry = {
            "phase": "execution",
            "timestamp": datetime.now().isoformat(),
            "task_id": current_task["id"],
            "agent": agent_name,
            "task_description": current_task["description"],
            "success": result.get("success", False),
            "result_summary": str(result.get("result", result.get("error", "")))[:500]
        }
        
        if result.get("success"):
            print(f"✅ Task completed successfully")
            current_task["status"] = "completed"
            current_task["result"] = result.get("result", "")
            
            # Store in agent outputs
            agent_outputs = state.get("agent_outputs", {})
            agent_outputs[current_task["id"]] = result
            
            # Extract and store structured data from agent results
            shared_data = state.get("shared_data", {}).copy()
            shared_data = self._extract_structured_data(agent_name, result, shared_data)
            
            return {
                "plan": plan,
                "current_task_index": current_idx + 1,
                "agent_outputs": agent_outputs,
                "shared_data": shared_data,
                "process_log": state.get("process_log", []) + [process_entry],
                "messages": [AIMessage(content=f"Task {current_task['id']} completed by {agent_name}")]
            }
        else:
            print(f"❌ Task failed: {result.get('error', 'Unknown error')}")
            current_task["status"] = "failed"
            current_task["error"] = result.get("error", "Unknown error")
            
            return {
                "plan": plan,
                "current_task_index": current_idx,
                "process_log": state.get("process_log", []) + [process_entry],
                "messages": [AIMessage(content=f"Task {current_task['id']} failed: {result.get('error', '')}")]
            }
    
    def _extract_structured_data(self, agent_name: str, result: Dict[str, Any], shared_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured data from agent results for programmatic use by downstream agents.
        
        This enables cross-agent data sharing without relying on LLM parsing of free-text context.
        
        Args:
            agent_name: Name of the agent that produced the result
            result: The result dictionary from the agent
            shared_data: Current shared_data dictionary to update
            
        Returns:
            Updated shared_data dictionary
        """
        result_content = result.get("result", "")
        
        if agent_name == "OmicAnalysis":
            # Extract genes from OmicAnalysis result
            # The result is a dict from omic_analysis_tool
            if isinstance(result_content, dict):
                # Direct dict result from tool
                top_genes = result_content.get("top_genes_by_fdr", [])
                if top_genes:
                    shared_data["top_genes"] = top_genes
                    print(f"[SharedData] Stored {len(top_genes)} genes from OmicAnalysis")
                
                # Also extract disease and cell type
                extracted = result_content.get("extracted_entities", {})
                if extracted.get("disease"):
                    shared_data["disease"] = extracted["disease"]
                if extracted.get("cell_type"):
                    shared_data["cell_type"] = extracted["cell_type"]
            else:
                # Try to parse if it's a string representation
                try:
                    if "top_genes_by_fdr" in str(result_content):
                        # Attempt to extract gene list from string
                        import re
                        genes_match = re.search(r"top_genes_by_fdr.*?:\s*\[([^\]]+)\]", str(result_content))
                        if genes_match:
                            genes_str = genes_match.group(1)
                            genes = [g.strip().strip("'\"") for g in genes_str.split(",")]
                            shared_data["top_genes"] = genes
                            print(f"[SharedData] Parsed {len(genes)} genes from OmicAnalysis string output")
                except Exception as e:
                    print(f"[SharedData] Warning: Could not parse genes from OmicAnalysis: {e}")
        
        elif agent_name == "PubMedResearcher":
            # Extract DOIs from PubMed results
            if isinstance(result_content, str):
                import re
                # Find all DOIs in the result
                doi_pattern = r'DOI:\s*(10\.\d{4,}/[^\s\)\]]+)'
                dois = re.findall(doi_pattern, result_content)
                if dois:
                    # Add to existing DOIs, avoiding duplicates
                    existing_dois = set(shared_data.get("paper_dois", []))
                    existing_dois.update(dois)
                    shared_data["paper_dois"] = list(existing_dois)
                    print(f"[SharedData] Stored {len(dois)} DOIs from PubMedResearcher (total: {len(shared_data['paper_dois'])})")
                
                # Also extract PMIDs
                pmid_pattern = r'PMID:\s*(\d+)'
                pmids = re.findall(pmid_pattern, result_content)
                if pmids:
                    existing_pmids = set(shared_data.get("paper_pmids", []))
                    existing_pmids.update(pmids)
                    shared_data["paper_pmids"] = list(existing_pmids)
        
        return shared_data
    
    async def _replanning_node(self, state: AgentState) -> Dict[str, Any]:
        """Re-plan when tasks fail"""
        print("\n🔄 RE-PLANNING PHASE")
        print("=" * 60)
        
        revision_count = state.get("plan_revision_count", 0) + 1
        max_revisions = state.get("max_plan_revisions", 2)
        
        if revision_count > max_revisions:
            print(f"⚠️ Max re-planning attempts ({max_revisions}) reached")
            return {
                "plan_revision_count": revision_count,
                "status": "reporting",
                "process_log": state.get("process_log", []) + [{
                    "phase": "replanning",
                    "timestamp": datetime.now().isoformat(),
                    "action": "max_revisions_reached"
                }]
            }
        
        # Get failed tasks
        failed_tasks = [t for t in state["plan"] if t["status"] == "failed"]
        completed_tasks = [t for t in state["plan"] if t["status"] == "completed"]
        
        replanning_prompt = f"""A task has failed. Re-plan the remaining work.

Original Query: {state['query']}

Completed Tasks:
{json.dumps([{"id": t["id"], "description": t["description"], "result": t.get("result", "")[:200]} for t in completed_tasks], indent=2)}

Failed Tasks:
{json.dumps([{"id": t["id"], "description": t["description"], "error": t.get("error", "")} for t in failed_tasks], indent=2)}

Create a revised plan that:
1. Accounts for what has already been completed
2. Addresses the failures with alternative approaches
3. May use different agents or modify the task descriptions

Respond in JSON format:
{{
    "revision_reason": "Why we're revising",
    "tasks": [
        {{
            "id": "task_new_1",
            "description": "Revised task description",
            "assigned_agent": "AgentName"
        }}
    ]
}}
"""
        
        response = await self.llm.ainvoke([
            SystemMessage(content="You are a strategic research planner. Output valid JSON only."),
            HumanMessage(content=replanning_prompt)
        ])
        
        try:
            # Extract text from response using helper function
            content = extract_text_from_llm_response(response.content)
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            replan_data = json.loads(content)
            
            # Create new tasks
            new_tasks = []
            for task in replan_data.get("tasks", []):
                new_tasks.append(SubTask(
                    id=task["id"],
                    description=task["description"],
                    assigned_agent=task["assigned_agent"],
                    status="pending",
                    result=None,
                    error=None,
                    attempts=0
                ))
            
            # Keep completed tasks and add new ones
            final_plan = completed_tasks + new_tasks
            
            print(f"📝 Revised plan with {len(new_tasks)} new tasks")
            
            return {
                "plan": final_plan,
                "current_task_index": len(completed_tasks),
                "plan_revision_count": revision_count,
                "status": "executing",
                "process_log": state.get("process_log", []) + [{
                    "phase": "replanning",
                    "timestamp": datetime.now().isoformat(),
                    "revision_count": revision_count,
                    "reason": replan_data.get("revision_reason", ""),
                    "new_tasks_count": len(new_tasks)
                }]
            }
            
        except json.JSONDecodeError:
            print("⚠️ Error parsing revised plan, proceeding to report")
            return {
                "plan_revision_count": revision_count,
                "status": "reporting"
            }
    
    async def _reporting_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate comprehensive final report with citations"""
        print("\n📄 REPORTING PHASE")
        print("=" * 60)
        
        # Gather all results
        completed_tasks = [t for t in state["plan"] if t["status"] == "completed"]
        failed_tasks = [t for t in state["plan"] if t["status"] == "failed"]
        process_log = state.get("process_log", [])
        
        # Prepare task results for the report
        task_results = []
        for task in completed_tasks:
            task_results.append({
                "task_id": task["id"],
                "description": task["description"],
                "agent": task["assigned_agent"],
                "result": task.get("result", "")
            })
        
        reporting_prompt = f"""Generate a comprehensive research report for a graduate-level biomedical audience.

## Research Question:
{state['query']}

## Task Results:
{json.dumps(task_results, indent=2, default=str)}

---

## REPORT STRUCTURE (Follow this order):

### Step 1: Omics Data Analysis Summary
- Report sample sizes (disease vs normal)
- Number of DEGs identified (upregulated/downregulated)
- **Top DEGs Table** (ranked by p-value/FDR): Show top 10-15 genes with:
  | Rank | Gene | log2FC | FDR | Direction |
- Note: Full gene list saved to differential_expression/ folder
- **Embed volcano plot** if available using: ![Volcano Plot](volcano_plots/volcano_plot.png)

### Step 2: Knowledge Graph Analysis  
- First-neighbor nodes of top DEGs from knowledge graph
- Categorize neighbors: Drugs, Pathways, GO Terms, Diseases
- Highlight potential drug targets and pathway connections

### Step 3: Literature-Validated Targets
- Targets found in both DEGs AND literature
- **Intersection Table**: Genes supported by both omics and literature
- For each validated target: brief literature summary with citations

### Step 4: Pathway Enrichment Analysis
- Summarize top enriched pathways (KEGG, Reactome, GO)
- **Embed pathway plots** if available:
  - ![KEGG Dotplot](plots/kegg_dotplot.png)
  - ![Pathway Combined](plots/pathway_combined_plot.png)
- Discuss biological significance of enriched pathways

### Step 5: Gene-Anchored Mechanistic Hypotheses (REQUIRED FORMAT)

Generate 3-5 ranked hypotheses. **EACH hypothesis MUST include**:

**5.1 Hypothesis Structure (for each)**
| Field | Required Content |
|-------|------------------|
| **Rank & Title** | e.g., "Rank 1: BDP1-Driven Translational Reprogramming" |
| **Gene Anchor(s)** | At least 1 gene symbol (e.g., BDP1, COX7C, FKBP1A) |
| **Pathway/Module** | Named pathway (e.g., RNA Pol III transcription, oxidative phosphorylation, mTOR signaling) |
| **Mechanistic Chain** | Gene → Pathway dysregulation → Intermediate biology → Phenotype |
| **Total Score** | X/100 using rubric below |

**5.2 Scoring Rubric (compute for each hypothesis)**
- **Fit-to-Evidence (0-30)**: How well does it explain the omics findings (DEGs, enrichment)?
- **Mechanistic Plausibility (0-20)**: Biological coherence with known pathways
- **Testability (0-15)**: Clear predictions; accessible validation methods
- **Novelty (0-10)**: Beyond generic findings; what's NEW vs. literature?
- **Clinical/Research Impact (0-10)**: Would this change treatment strategy or research direction?
- **Parsimony (0-5)**: Explains more with fewer assumptions
- **Low Confounding Risk (0-10)**: Start at 10, subtract for major confounders

**5.3 For Each Hypothesis, Provide**:
1. **Mechanism** (2-4 sentences, causal chain style)
2. **Predictions** (3-5 falsifiable predictions including at least one omics signature)
3. **Validation Plan**:
   - *Computational*: DE signature scoring, pathway enrichment, network analysis
   - *In vitro*: CRISPR knockdown/activation, reporter assays, drug screens
   - *Ex vivo*: Patient organoids, tissue explants
   - *In vivo*: Xenografts, genetic mouse models
4. **Supporting Evidence**: Cite specific findings from our analysis AND mined literature
5. **Contradicting Evidence / Gaps**: What challenges this hypothesis?
6. **Novelty Statement**: 
   - What is ALREADY KNOWN (cite literature)
   - What GAP exists (what papers don't connect)
   - What NEW LINK you propose
   - Why it's ACTIONABLE (new target, biomarker, stratification)

**5.4 Critical Assessment**
- **Key Assumptions**: What must be true for each hypothesis? Which are untested?
- **Knowledge Gaps**: What additional data would strengthen conclusions?
- **Limitations**: Sample size, cell heterogeneity, batch effects, etc.
- **Confounders**: Stromal contamination, treatment history, cohort bias

**5.5 Top 3 Minimal Validation Actions**
Prioritized list of the most information-gaining experiments/analyses that could quickly validate or refute top hypotheses.

### Step 6: Conclusions
- Key actionable findings (ranked by confidence and impact)
- Immediate next steps for experimental validation
- References (with DOI/PMID)

## KEY REQUIREMENTS:
- Use tables for gene lists (keep them concise)
- EMBED plots using markdown: ![Title](relative_path.png)
- Emphasize INTERSECTION of evidence sources
- Generate testable hypotheses with mechanism descriptions
- All citations must include DOI or PMID
"""
        
        response = await self.llm.ainvoke([
            SystemMessage(content="""You are an expert biomedical research synthesizer specializing in gene-pathway-anchored hypothesis generation. 

**Core Synthesis Capabilities:**
1. Summarize omics findings with statistics (samples, DEG counts, effect sizes)
2. Show ranked gene tables (top 10-15, note full list in files)
3. EMBED plots using markdown syntax: ![Title](relative_path.png)
   - Volcano plots: ![Volcano Plot](volcano_plots/volcano_plot.png)
   - KEGG plots: ![KEGG Dotplot](plots/kegg_dotplot.png)
   - Enrichment plots: ![Enrichment](enrichment_results/enrichment_plots/filename.png)

**Advanced Hypothesis Framework (MedHypoRank-GenePath):**
- Generate 8-15 gene-pathway-anchored hypotheses per report
- Each hypothesis MUST: anchor to specific Gene(s)→Pathway(s)→Phenotype chain
- Score each hypothesis (0-100) using subscores: Novelty(20), Literature(15), Mechanism(20), Feasibility(15), Significance(15), Data(10), Cross-validation(5)
- Distinguish: 1=confirmatory (existing literature), 2=incremental extension, 3=novel discovery
- For novel hypotheses: justify what makes them NEW vs existing literature
- Propose specific validation experiments: In-Vitro (cell lines, assays) → In-Vivo (animal models) → Translational (patient samples)

**Scoring Rubric:**
- 0-30: Weak support, speculative
- 31-60: Moderate support, some mechanistic basis
- 61-80: Strong support, clear mechanism, testable
- 81-100: Exceptional, multiple converging evidence lines, high clinical potential

Keep tables concise. Prioritize actionable, testable hypotheses over confirmatory observations."""),
            HumanMessage(content=reporting_prompt)
        ])
        
        # Extract text from response using helper function
        report = extract_text_from_llm_response(response.content)
        
        # Save the report with appendix and generate PDF
        report_path = self._save_report(state["query"], report, state=state)
        
        print(f"📄 Report generated ({len(report)} characters)")
        if report_path:
            print(f"💾 Saved to: {report_path}")
        
        return {
            "final_report": report,
            "status": "completed",
            "process_log": process_log + [{
                "phase": "reporting",
                "timestamp": datetime.now().isoformat(),
                "report_length": len(report),
                "report_path": report_path
            }]
        }
    
    def _should_execute_or_report(self, state: AgentState) -> str:
        """Determine if we should execute tasks or go to reporting"""
        if not state.get("plan") or len(state["plan"]) == 0:
            return "report"
        return "execute"
    
    def _should_continue_or_replan(self, state: AgentState) -> str:
        """Determine next step after execution"""
        plan = state["plan"]
        current_idx = state["current_task_index"]
        
        # Check if all tasks are done
        if current_idx >= len(plan):
            return "report"
        
        # Check if current task failed
        current_task = plan[current_idx]
        if current_task["status"] == "failed":
            # Check if we should replan
            if current_task["attempts"] >= 2:
                return "replan"
            # Retry the task
            return "continue"
        
        # Continue to next task
        return "continue"
    
    def _after_replan(self, state: AgentState) -> str:
        """Determine next step after replanning"""
        if state.get("status") == "reporting":
            return "report"
        return "execute"
    
    def _generate_appendix(self, state: AgentState) -> str:
        """Generate appendix section with state metadata and detailed outputs"""
        appendix_parts = []
        appendix_parts.append("\n\n---\n\n# Appendix\n")
        
        # A1: Session Information
        appendix_parts.append("## A1. Session Information\n\n")
        appendix_parts.append(f"- **Session ID**: `{state.get('session_id', 'N/A')}`\n")
        appendix_parts.append(f"- **Generated At**: {datetime.now().isoformat()}\n")
        appendix_parts.append(f"- **Query**: {state.get('query', 'N/A')}\n\n")
        
        # A2: Execution Plan
        plan = state.get("plan", [])
        if plan:
            appendix_parts.append("## A2. Execution Plan\n\n")
            appendix_parts.append("| Step | Task ID | Agent | Description | Status |\n")
            appendix_parts.append("|------|---------|-------|-------------|--------|\n")
            for i, task in enumerate(plan, 1):
                task_id = task.get('id', f'task_{i}')
                agent = task.get('assigned_agent', 'Unknown')
                description = task.get('description', 'N/A')
                # Truncate long descriptions
                if len(description) > 60:
                    description = description[:57] + "..."
                status = task.get('status', 'pending')
                appendix_parts.append(f"| {i} | {task_id} | {agent} | {description} | {status} |\n")
            appendix_parts.append("\n")
        
        # A3: Process Log
        process_log = state.get("process_log", [])
        if process_log:
            appendix_parts.append("## A3. Process Log\n\n")
            appendix_parts.append("```\n")
            for entry in process_log[-50:]:  # Last 50 entries to avoid too much
                appendix_parts.append(f"{entry}\n")
            appendix_parts.append("```\n\n")
        
        # A4: Shared Data Summary
        shared_data = state.get("shared_data", {})
        if shared_data:
            appendix_parts.append("## A4. Shared Data Summary\n\n")
            
            # Top genes - full list if available
            top_genes = shared_data.get("top_genes", [])
            if top_genes:
                appendix_parts.append("### Top Differentially Expressed Genes\n\n")
                appendix_parts.append("| # | Gene | Log2FC | FDR | Direction |\n")
                appendix_parts.append("|---|------|--------|-----|----------|\n")
                for i, gene in enumerate(top_genes[:50], 1):  # Top 50 genes
                    if isinstance(gene, dict):
                        name = gene.get('gene', gene.get('name', str(gene)))
                        log2fc = gene.get('log2_fold_change', gene.get('log2fc', 'N/A'))
                        fdr = gene.get('fdr', gene.get('adj_pvalue', 'N/A'))
                        direction = gene.get('direction', 'N/A')
                        if isinstance(log2fc, float):
                            log2fc = f"{log2fc:.3f}"
                        if isinstance(fdr, float):
                            fdr = f"{fdr:.2e}"
                        appendix_parts.append(f"| {i} | {name} | {log2fc} | {fdr} | {direction} |\n")
                    else:
                        appendix_parts.append(f"| {i} | {gene} | - | - | - |\n")
                appendix_parts.append("\n")
            
            # Disease and cell type
            if shared_data.get("disease"):
                appendix_parts.append(f"**Disease Context**: {shared_data['disease']}\n\n")
            if shared_data.get("cell_type"):
                appendix_parts.append(f"**Cell Type**: {shared_data['cell_type']}\n\n")
            
            # Paper DOIs
            paper_dois = shared_data.get("paper_dois", [])
            if paper_dois:
                appendix_parts.append("### Literature References (DOIs)\n\n")
                for i, doi in enumerate(paper_dois[:20], 1):  # Top 20 DOIs
                    appendix_parts.append(f"{i}. `{doi}`\n")
                appendix_parts.append("\n")
            
            # Pathways
            pathways = shared_data.get("pathways", [])
            if pathways:
                appendix_parts.append("### Enriched Pathways\n\n")
                for i, pathway in enumerate(pathways[:20], 1):  # Top 20 pathways
                    appendix_parts.append(f"{i}. {pathway}\n")
                appendix_parts.append("\n")
        
        # A5: Agent Outputs Summary
        agent_outputs = state.get("agent_outputs", {})
        if agent_outputs:
            appendix_parts.append("## A5. Agent Outputs Summary\n\n")
            for agent_name, output in agent_outputs.items():
                appendix_parts.append(f"### {agent_name}\n\n")
                # Truncate long outputs but show more than in main report
                output_str = str(output)
                if len(output_str) > 3000:
                    appendix_parts.append(f"```\n{output_str[:3000]}...\n[Output truncated - {len(output_str)} chars total]\n```\n\n")
                else:
                    appendix_parts.append(f"```\n{output_str}\n```\n\n")
        
        return "".join(appendix_parts)
    
    def _compile_pdf(self, markdown_path: str) -> Optional[str]:
        """Compile the markdown report to PDF using available tools"""
        try:
            import subprocess
            import shutil
            
            pdf_path = markdown_path.replace('.md', '.pdf')
            html_path = markdown_path.replace('.md', '.html')
            session_dir = os.path.dirname(markdown_path)
            
            # Check available tools
            has_pandoc = shutil.which('pandoc')
            has_wkhtmltopdf = shutil.which('wkhtmltopdf')
            has_xelatex = shutil.which('xelatex')
            has_pdflatex = shutil.which('pdflatex')
            
            if not has_pandoc:
                print("⚠️ pandoc not found - skipping PDF generation")
                print("   Install with: sudo apt-get install pandoc")
                return None
            
            # Strategy 1: Try pandoc with LaTeX if available
            if has_xelatex or has_pdflatex:
                pdf_engine = 'xelatex' if has_xelatex else 'pdflatex'
                cmd = [
                    'pandoc', markdown_path, '-o', pdf_path,
                    f'--pdf-engine={pdf_engine}',
                    '-V', 'geometry:margin=1in',
                    '-V', 'fontsize=11pt',
                    '--toc', '--toc-depth=2',
                    '--highlight-style=tango',
                    '--resource-path', session_dir,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=session_dir, timeout=180)
                if result.returncode == 0:
                    print(f"📄 PDF generated (LaTeX): {pdf_path}")
                    return pdf_path
                print(f"⚠️ LaTeX PDF failed, trying HTML method...")
            
            # Strategy 2: Pandoc → HTML → wkhtmltopdf (if available)
            if has_wkhtmltopdf:
                # Create a CSS file to ensure images fit within page bounds
                css_content = """
/* Ensure images fit within page bounds */
img {
    max-width: 100% !important;
    max-height: 650px !important;
    height: auto !important;
    width: auto !important;
    display: block;
    margin: 10px auto;
    page-break-inside: avoid;
}
/* Better table styling */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 15px 0;
    font-size: 10pt;
    page-break-inside: avoid;
}
th, td {
    border: 1px solid #ddd;
    padding: 6px 8px;
    text-align: left;
}
th {
    background-color: #f5f5f5;
    font-weight: bold;
}
/* Better heading spacing */
h1, h2, h3 {
    page-break-after: avoid;
    margin-top: 20px;
}
/* Code blocks */
pre, code {
    font-size: 9pt;
    background-color: #f8f8f8;
    border-radius: 3px;
    overflow-x: auto;
}
/* Page breaks */
.page-break {
    page-break-before: always;
}
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.5;
}
"""
                css_path = os.path.join(session_dir, '_report_style.css')
                with open(css_path, 'w') as f:
                    f.write(css_content)
                
                # First convert to HTML with embedded images and custom CSS
                # Use --self-contained for pandoc 2.x, --embed-resources for 3.x
                html_cmd = [
                    'pandoc', markdown_path, '-o', html_path,
                    '--standalone',
                    '--self-contained',  # Works for pandoc 2.x (--embed-resources is 3.x only)
                    '--toc', '--toc-depth=2',
                    '--highlight-style=tango',
                    '-V', 'title=OmniCellAgent Analysis Report',
                    '--resource-path', session_dir,
                    '-c', css_path,  # Include custom CSS
                ]
                result = subprocess.run(html_cmd, capture_output=True, text=True, cwd=session_dir, timeout=120)
                if result.returncode == 0:
                    # Convert HTML to PDF
                    pdf_cmd = [
                        'wkhtmltopdf',
                        '--enable-local-file-access',
                        '--margin-top', '20mm',
                        '--margin-bottom', '20mm',
                        '--margin-left', '15mm',
                        '--margin-right', '15mm',
                        html_path, pdf_path
                    ]
                    result2 = subprocess.run(pdf_cmd, capture_output=True, text=True, cwd=session_dir, timeout=180)
                    if result2.returncode == 0:
                        print(f"📄 PDF generated (wkhtmltopdf): {pdf_path}")
                        # Clean up intermediate files
                        try:
                            os.remove(html_path)
                            os.remove(css_path)
                        except:
                            pass
                        return pdf_path
                    else:
                        print(f"⚠️ wkhtmltopdf failed: {result2.stderr[:200] if result2.stderr else 'No error message'}")
                else:
                    print(f"⚠️ Pandoc HTML conversion failed: {result.stderr[:200] if result.stderr else 'No error message'}")
                
                # Clean up CSS file if we get here
                try:
                    os.remove(css_path)
                except:
                    pass
                else:
                    print(f"⚠️ Pandoc HTML conversion failed: {result.stderr[:200] if result.stderr else 'No error message'}")
            
            # Strategy 3: Generate standalone HTML (always works)
            html_cmd = [
                'pandoc', markdown_path, '-o', html_path,
                '--standalone',
                '--toc', '--toc-depth=2',
                '--highlight-style=tango',
                '-V', 'title=OmniCellAgent Analysis Report',
                '--resource-path', session_dir,
                '--metadata', f'date={datetime.now().strftime("%Y-%m-%d")}',
                '-c', 'https://cdn.jsdelivr.net/npm/water.css@2/out/water.css',  # Nice CSS
            ]
            result = subprocess.run(html_cmd, capture_output=True, text=True, cwd=session_dir, timeout=60)
            if result.returncode == 0:
                print(f"📄 HTML report generated: {html_path}")
                print("   (PDF requires: wkhtmltopdf or texlive-xetex)")
                return html_path  # Return HTML path instead
            
            print(f"⚠️ Could not generate PDF or HTML: {result.stderr[:300]}")
            return None
                    
        except subprocess.TimeoutExpired:
            print("⚠️ PDF generation timed out")
            return None
        except Exception as e:
            print(f"⚠️ Error generating PDF: {e}")
            return None
    
    def _save_report(self, query: str, report: str, state: AgentState = None) -> Optional[str]:
        """Save the report to a file with appendix, then compile to PDF"""
        try:
            sessions_base = get_path('sessions.base', absolute=True, create=True)
            session_dir = os.path.join(sessions_base, self.session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"report_{timestamp}.md"
            report_path = os.path.join(session_dir, report_filename)
            
            header = f"""---
title: OmniCellAgent Analysis Report (LangGraph)
session_id: {self.session_id}
generated_at: {datetime.now().isoformat()}
query: {query}
---

"""
            # Generate appendix from state if available
            appendix = ""
            if state:
                try:
                    appendix = self._generate_appendix(state)
                    print("📎 Appendix generated with state metadata")
                except Exception as e:
                    print(f"⚠️ Could not generate appendix: {e}")
            
            # Write full report with appendix
            full_report = header + report + appendix
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(full_report)
            
            print(f"📝 Markdown report saved: {report_path}")
            
            # Compile to PDF
            pdf_path = self._compile_pdf(report_path)
            if pdf_path:
                print(f"✅ Report available as both MD and PDF")
            
            return report_path
            
        except Exception as e:
            print(f"Error saving report: {e}")
            return None
    
    def _save_conversation_log(self, query: str, state: AgentState):
        """Save the full conversation log"""
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"langgraph_log_{timestamp}.json"
            filepath = os.path.join(self.log_dir, filename)
            
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "query": query,
                "plan": state.get("plan", []),
                "process_log": state.get("process_log", []),
                "agent_outputs": {k: str(v)[:1000] for k, v in state.get("agent_outputs", {}).items()},
                "final_report": state.get("final_report", "")[:5000]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"💾 Log saved: {filepath}")
            
        except Exception as e:
            print(f"Error saving log: {e}")
    
    async def run(self, query: str) -> str:
        """Run the agent system on a query"""
        print(f"\n🔬 Processing query: {query}")
        print("=" * 80)
        
        # Initialize state
        initial_state: AgentState = {
            "query": query,
            "session_id": self.session_id,
            "plan": [],
            "current_task_index": 0,
            "plan_revision_count": 0,
            "max_plan_revisions": 2,
            "messages": [],
            "agent_outputs": {},
            "shared_data": {
                "top_genes": [],      # Populated by OmicAnalysis
                "paper_dois": [],      # Populated by PubMed tools
                "pathways": [],        # Populated by KEGG analysis
                "disease": "",         # Extracted from query
                "cell_type": "",       # Extracted from query
            },
            "process_log": [],
            "final_report": None,
            "status": "planning"
        }
        
        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)
        
        # Save logs
        self._save_conversation_log(query, final_state)
        
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 80)
        
        return final_state.get("final_report", "No report generated")


async def main():
    """Main function to run the LangGraph agent system"""
    parser = argparse.ArgumentParser(description="LangGraph OmniCellAgent for Biomedical Research")
    parser.add_argument(
        "--query", 
        type=str, 
        default="What are the key dysfunctional signaling targets in microglia of Alzheimer's disease?",
        help="Query to process"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="gemini-3-pro-preview",
        help="Model to use (default: gemini-3-pro-preview_)"
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session ID for this run"
    )
    
    args = parser.parse_args()
    
    agent = LangGraphOmniCellAgent(
        model_name=args.model,
        session_id=args.session_id
    )
    
    report = await agent.run(args.query)
    
    print("\n📄 FINAL REPORT:")
    print("=" * 80)
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
