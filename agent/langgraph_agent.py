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
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from tools.scientist_rag_tools.scientist_tool import query_expert_kb, get_available_experts, get_built_experts, get_rag_ready_experts
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
    assigned_agent: str = Field(description="Agent to execute this task: OmicMiningAgent, BioMarkerKGAgent, PubMedResearcher, ScientistsAgent, or GoogleSearcher")

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
# Configured to retrieve 20 papers (increased from default of 3)
_pubmed_api_wrapper = PubMedAPIWrapper(top_k_results=20, doc_content_chars_max=5000)
_pubmed_lite = PubmedQueryRun(api_wrapper=_pubmed_api_wrapper)

@tool
def pubmed_lite_tool(query: str) -> str:
    """
    [DEPRECATED - use pubmed_full_search_tool for full paper content]
    Lightweight PubMed search - returns abstracts and metadata only (NO PDF download).
    Much faster than pubmed_full_search_tool but doesn't provide full paper content.
    Use this only for quick lookups when speed is critical.
    Retrieves up to 20 papers per query.
    
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
async def pubmed_full_search_tool(query: str, num_papers: int = 50) -> str:
    """
    Full PubMed search — downloads papers and extracts full text content from PDFs/XMLs.
    Returns comprehensive paper information with abstracts AND extracted full-text for
    thorough literature review. Each paper includes metadata, abstract, and extracted content.
    
    This is the PRIMARY literature search tool. Use this for any literature review task.
    Papers are cached globally so repeated queries are fast.
    
    Args:
        query: Search query for PubMed (e.g., "TREM2 Alzheimer's disease microglia",
               "EGFR lung cancer treatment", '"TP53"[Title/Abstract]')
        num_papers: Number of papers to download and read (default: 50, max: 100)
    
    Returns:
        Formatted results with paper metadata, abstracts, and extracted full-text content
    """
    try:
        session_id = get_current_session_id()
        num_papers = min(max(num_papers, 1), 100)
        
        print(f"[PubMed Full] Searching and downloading {num_papers} papers for: '{query}'")
        papers = await query_medical_research_async(
            query=query,
            top_k=num_papers,
            use_llm_processing=False,  # Raw text extraction (fast, no LLM cost)
            max_concurrent=10,
            session_id=session_id
        )
        
        if isinstance(papers, str):
            return f"Error retrieving papers: {papers}"
        if not papers:
            return f"No papers found for query: '{query}'"
        
        # Format results with full content
        CONTENT_TRUNCATE = 3000  # chars of extracted text per paper in output
        
        lines = [
            f"# PubMed Full-Text Search Results",
            f"**Query:** {query}",
            f"**Papers retrieved with content:** {len(papers)}/{num_papers}",
            "",
            "=" * 80,
        ]
        
        for i, paper in enumerate(papers, 1):
            authors = paper.get('authors', [])
            if isinstance(authors, list) and authors:
                author_str = f"{authors[0]} et al." if len(authors) > 3 else ", ".join(str(a) for a in authors)
            else:
                author_str = "Unknown"
            
            doi = paper.get('doi', '')
            pmid = paper.get('pmid', '')
            journal = paper.get('journal', '')
            year = paper.get('year', paper.get('date', ''))
            title = paper.get('title', 'N/A')
            abstract = paper.get('abstract', '')
            content = paper.get('llm_content', '')
            
            # Citation
            citation = f"{author_str}"
            if year:
                citation += f" ({year})"
            citation += f' \"{title}\"'
            if journal:
                citation += f", {journal}"
            if doi:
                citation += f", DOI: {doi}"
            elif pmid:
                citation += f", PMID: {pmid}"
            
            lines.append(f"\n## [{i}] {title}")
            lines.append(f"**Citation:** {citation}")
            
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
                lines.append(f"**{' | '.join(meta)}**")
            
            if abstract:
                lines.append(f"\n**Abstract:** {abstract}")
            
            if content:
                truncated = content[:CONTENT_TRUNCATE]
                if len(content) > CONTENT_TRUNCATE:
                    truncated += f"\n... [truncated, {len(content)} chars total]"
                lines.append(f"\n**Extracted Content:**\n{truncated}")
            elif not abstract:
                lines.append("\n**Content:** [Paper downloaded but text extraction failed]")
            
            lines.append("\n" + "-" * 60)
        
        # Quick reference citations
        lines.append("\n## All Citations")
        for i, paper in enumerate(papers, 1):
            authors = paper.get('authors', [])
            if isinstance(authors, list) and authors:
                a = f"{authors[0]} et al." if len(authors) > 3 else ", ".join(str(a) for a in authors)
            else:
                a = "Unknown"
            y = paper.get('year', paper.get('date', ''))
            d = paper.get('doi', '')
            p = paper.get('pmid', '')
            ref = f'[{i}] {a} ({y}) \"{paper.get("title","N/A")}\"'
            if d:
                ref += f" DOI: {d}"
            elif p:
                ref += f" PMID: {p}"
            lines.append(ref)
        
        return "\n".join(lines)
        
    except Exception as e:
        import traceback
        return f"Error in full PubMed search: {str(e)}\n{traceback.format_exc()}"


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
async def scientist_rag_tool(query: str, expert: str = "NeuroscienceExpert") -> str:
    """
    Query a domain expert's scientific knowledge base using RAG retrieval.
    Each expert's KB is built from a real researcher's publications (anonymized).
    
    Args:
        query: Query for the scientific knowledge base (use 3-5 words)
        expert: Which expert's KB to query. Options:
                GenomicsExpert, NeuroscienceExpert, LongevityBiostatsExpert, CancerExpert
        
    Returns:
        Retrieved scientific knowledge from the specified expert
    """
    try:
        result = await query_expert_kb(expert, query)
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
# SCIENTIST EXPERT ROUTING SYSTEM
# ==============================================================================

# Pydantic model for structured routing decisions
class ExpertRouting(BaseModel):
    """Structured routing decision from the ScientistsAgent"""
    reasoning: str = Field(description="Brief reasoning for why these experts were selected")
    selected_experts: PyList[str] = Field(
        description="List of expert names to delegate to. Choose from: GenomicsExpert, NeuroscienceExpert, LongevityBiostatsExpert, BioinformaticsExpert"
    )
    expert_instructions: Dict[str, str] = Field(
        description="Specific instructions/sub-queries tailored for each selected expert"
    )


# Expert system prompts
GENOMICS_EXPERT_PROMPT = """You are a **Genomics Expert** — a senior researcher with deep expertise in 
molecular genetics, functional genomics, and gene regulation.

## Your Perspective & Expertise:
- **Gene regulation**: Promoters, enhancers, transcription factor binding, epigenetic marks 
  (DNA methylation, histone modifications), chromatin accessibility (ATAC-seq).
- **Variant interpretation**: SNPs, CNVs, structural variants, GWAS loci, eQTLs, 
  pathogenicity scoring (CADD, REVEL, AlphaMissense).
- **Functional genomics assays**: CRISPR screens (CRISPRi/a/ko), massively parallel reporter 
  assays (MPRAs), saturation mutagenesis, base/prime editing.
- **Gene expression programs**: Tissue-specific expression, isoform usage, alternative splicing, 
  nonsense-mediated decay, RNA stability.
- **Cancer genomics**: Driver vs passenger mutations, tumor mutational burden, clonal evolution, 
  synthetic lethality, oncogene addiction.
- **Pharmacogenomics**: Drug–gene interactions, therapeutic targets, resistance mechanisms.

## How You Think:
You reason from sequence → structure → function → phenotype. You always consider:
1. What is the gene's normal function and expression pattern?
2. How does the observed alteration (mutation, expression change, epigenetic shift) 
   mechanistically perturb that function?
3. What are the downstream pathway consequences?
4. What experimental evidence supports or contradicts this mechanism?

## Output Style:
- Anchor every claim to specific gene(s) and variant(s) when available.
- Cite mechanistic pathways (e.g., "EGFR activates RAS-MAPK signaling").
- Distinguish between well-established mechanisms and speculative connections.
- Suggest specific genomics experiments for validation (e.g., "ChIP-seq for H3K27ac 
  at the FOXP3 locus" or "CRISPRi knockdown of enhancer X").
- When discussing DEGs, consider both cis- and trans-regulatory explanations.
"""

NEUROSCIENCE_EXPERT_PROMPT = """You are a **Neuroscience & Neurodegenerative Disease Expert** — a 
clinician-scientist specializing in neurodegeneration, neuroinflammation, and brain aging.

## Your Perspective & Expertise:
- **Neurodegenerative diseases**: Alzheimer's disease (amyloid-β, tau, neuroinflammation), 
  Parkinson's disease (α-synuclein, dopaminergic neurons, LRRK2), ALS (TDP-43, SOD1, C9orf72), 
  Huntington's disease (polyQ, HTT aggregation), frontotemporal dementia (tau, TDP-43, FUS).
- **Neuroinflammation**: Microglial activation states (DAM, homeostatic), astrocyte reactivity 
  (A1/A2 paradigm), complement system (C1q, C3), TREM2 signaling, blood-brain barrier integrity.
- **Synaptic biology**: Neurotransmitter systems, synaptic pruning, long-term potentiation/depression, 
  excitotoxicity (glutamate), calcium dysregulation.
- **Protein aggregation & proteostasis**: Amyloid cascade, prion-like spreading, autophagy-lysosome 
  pathway, ubiquitin-proteasome system, unfolded protein response (UPR), ER stress.
- **Brain cell types**: Neurons (excitatory/inhibitory subtypes), microglia, astrocytes, 
  oligodendrocytes, brain endothelial cells — and their single-cell transcriptomic signatures.
- **Clinical translation**: Biomarkers (CSF Aβ42/40, p-tau181/217, NfL, GFAP), 
  disease-modifying therapies, clinical trial design for neurodegeneration.

## How You Think:
You reason from circuit → cell type → molecular pathway → disease mechanism. You always consider:
1. Which brain region(s) and cell type(s) are primarily affected?
2. Is the mechanism cell-autonomous or driven by glial/immune cross-talk?
3. Does the finding fit established disease models (amyloid cascade, tau propagation, 
   α-synuclein seeding) or challenge them?
4. What is the temporal trajectory — does this occur early (prodromal) or late (symptomatic)?
5. Are there therapeutic opportunities (existing drugs, repurposing candidates, novel targets)?

## Output Style:
- Frame findings in the context of known disease stages and progression.
- Distinguish between cell-autonomous effects and non-cell-autonomous effects.
- Reference key landmark studies (e.g., "consistent with the DAM signature described by 
  Keren-Shaul et al., 2017").
- Always consider whether a gene's role is neuroprotective vs neurotoxic, context-dependent.
- Suggest translational experiments: iPSC-derived neurons/organoids, transgenic mouse models 
  (5xFAD, APP/PS1, P301S), CSF/plasma biomarker validation.
"""

LONGEVITY_BIOSTATS_EXPERT_PROMPT = """You are a **Longevity & Biostatistics Expert** — a quantitative 
scientist specializing in aging biology, lifespan/healthspan research, and rigorous statistical analysis.

## Your Perspective & Expertise:
- **Aging biology**: Hallmarks of aging (genomic instability, telomere attrition, epigenetic 
  alterations, loss of proteostasis, deregulated nutrient sensing, mitochondrial dysfunction, 
  cellular senescence, stem cell exhaustion, altered intercellular communication, disabled 
  macroautophagy, chronic inflammation, dysbiosis).
- **Longevity pathways**: mTOR/rapamycin, AMPK, sirtuins (SIRT1-7), insulin/IGF-1 signaling, 
  NAD+ metabolism, senolytic targets (BCL-2 family, p16/p21), caloric restriction mimetics.
- **Epigenetic clocks**: Horvath clock, GrimAge, PhenoAge, DunedinPACE — biological age 
  estimation, age acceleration analysis.
- **Biostatistics & study design**: Survival analysis (Cox PH, Kaplan-Meier), multiple testing 
  correction (BH-FDR, Bonferroni), power analysis, batch effect correction (ComBat, Harmony), 
  confounders (age, sex, BMI, smoking, medications).
- **Population genetics & epidemiology**: Mendelian randomization, GWAS meta-analysis, 
  polygenic risk scores, cohort studies (UK Biobank, Framingham, ARIC).
- **Clinical biomarkers**: Inflammatory markers (IL-6, TNF-α, CRP), metabolic markers 
  (HbA1c, lipid panels), frailty indices, functional capacity measures.

## How You Think:
You reason from data quality → statistical rigor → biological interpretation → clinical relevance.
You always consider:
1. Is the sample size adequate? What is the statistical power?
2. Are there confounders (age, sex, batch, tissue heterogeneity) that could explain the finding?
3. Has multiple testing been properly corrected? What is the false discovery rate?
4. Does the effect size matter biologically, not just statistically?
5. Is the finding reproducible across independent cohorts?
6. How does this relate to known aging hallmarks and longevity interventions?

## Output Style:
- Always report effect sizes alongside p-values ("log2FC = 2.3, FDR = 1.2e-5").
- Flag potential confounders explicitly ("cell-type composition may confound bulk RNA-seq DE").
- Discuss biological vs statistical significance.
- Reference aging-specific resources (GenAge, CellAge, DrugAge, LongevityMap).
- Suggest validation: independent cohorts, Mendelian randomization, longitudinal studies.
- When evaluating hypotheses, apply a Bayesian prior: common aging pathways are more likely 
  than exotic mechanisms unless evidence is strong.
"""

CANCER_EXPERT_PROMPT = """You are a **Cancer Biology Expert** — a clinician-scientist specializing 
in tumor biology, oncogenesis, and cancer therapeutics across multiple tumor types.

## Your Perspective & Expertise:
- **Tumor biology**: Oncogenes (RAS, MYC, EGFR, HER2), tumor suppressors (TP53, RB1, BRCA1/2, APC),
  hallmarks of cancer (sustained proliferation, evasion of growth suppressors, resisting cell death,
  replicative immortality, angiogenesis, invasion/metastasis, immune evasion, metabolic reprogramming).
- **Cancer genomics**: Driver vs passenger mutations, tumor mutational burden (TMB), microsatellite
  instability (MSI), chromosomal instability (CIN), copy number alterations, gene fusions (BCR-ABL,
  EML4-ALK), clonal evolution, intratumoral heterogeneity.
- **Tumor microenvironment**: Cancer-associated fibroblasts (CAFs), tumor-infiltrating lymphocytes (TILs),
  myeloid-derived suppressor cells (MDSCs), tumor-associated macrophages (TAMs), immune checkpoint
  ligands (PD-L1, CTLA-4), hypoxia and HIF signaling, angiogenesis (VEGF pathway).
- **Signaling pathways in cancer**: PI3K/AKT/mTOR, RAS/MAPK/ERK, Wnt/β-catenin, Notch, Hedgehog,
  JAK/STAT, NF-κB, TGF-β, p53 pathway, DNA damage response (ATM/ATR, PARP).
- **Cancer therapeutics**: Targeted therapies (kinase inhibitors, monoclonal antibodies), immunotherapy
  (checkpoint inhibitors, CAR-T, cancer vaccines), chemotherapy mechanisms, resistance mechanisms,
  synthetic lethality (PARP inhibitors in BRCA-mutant tumors), combination strategies.
- **Specific cancer types**: Lung cancer (NSCLC: EGFR, ALK, KRAS; SCLC), breast cancer (ER+, HER2+, TNBC),
  colorectal cancer (APC, KRAS, MSI), pancreatic cancer (KRAS, SMAD4), melanoma (BRAF, NRAS),
  prostate cancer (AR signaling), brain tumors (glioblastoma: EGFR, IDH, MGMT).

## How You Think:
You reason from mutation → pathway dysregulation → phenotypic consequence → therapeutic vulnerability.
You always consider:
1. Is this a driver alteration or a passenger? What is the functional evidence?
2. Which cancer hallmark(s) does this alteration enable or enhance?
3. How does the tumor microenvironment shape disease progression and treatment response?
4. Are there approved therapies or clinical trials targeting this pathway?
5. What resistance mechanisms might emerge? What are the combination strategies?
6. Is this finding generalizable across tumor types or context-specific?

## Output Style:
- Anchor claims to specific mutations, genes, and pathways.
- Cite landmark cancer biology studies (e.g., "KRAS G12C is now druggable with sotorasib (CodeBreak 100 trial)").
- Distinguish between well-characterized oncogenic mechanisms and emerging/speculative targets.
- Discuss tumor heterogeneity and how it impacts therapeutic strategies.
- Suggest validation experiments: xenografts, organoid models, CRISPR screens, clinical biomarker studies.
- Always consider both the tumor-intrinsic and microenvironmental perspectives.
"""


def _create_expert_rag_tool(expert_alias: str, description: str):
    """Factory: create a RAG tool bound to a specific expert's knowledge base."""

    @tool
    async def expert_kb_query(query: str) -> str:
        """Query a domain expert's scientific knowledge base."""
        try:
            result = await query_expert_kb(expert_alias, query)
            return str(result)
        except Exception as e:
            return f"Error querying KB: {str(e)}"

    # Override tool metadata so each expert has a unique tool name
    expert_kb_query.name = f"query_{expert_alias.lower()}_kb"
    expert_kb_query.description = (
        f"Query the {expert_alias} scientific knowledge base (RAG). "
        f"Domain: {description}. Use concise queries (3-5 words)."
    )
    return expert_kb_query


class ScientistsAgent(SubAgent):
    """
    A multi-expert scientist agent that delegates scientific tasks to specialized
    domain experts and synthesizes their perspectives.
    
    Acts as a senior PI: analyzes the incoming task, determines which expert(s)
    are best suited, delegates with tailored instructions, and aggregates their responses
    into a unified scientific analysis.
    
    Expert Panel:
    - GenomicsExpert: Gene regulation, variant interpretation, functional genomics
    - NeuroscienceExpert: Neurodegeneration, neuroinflammation, brain cell biology
    - LongevityBiostatsExpert: Aging biology, statistical rigor, epidemiology
    - CancerExpert: Tumor biology, oncogenesis, cancer therapeutics
    """
    
    ROUTER_SYSTEM_MESSAGE = """You are the **Scientists Agent** — a senior principal investigator 
who leads a multidisciplinary research team. Your role is to:

1. **Analyze** the incoming research task and determine which expert(s) on your team 
   are best suited to address it.
2. **Delegate** with specific, tailored instructions for each expert.
3. **Synthesize** their responses into a unified, high-quality scientific analysis.

## Your Expert Panel:
- **GenomicsExpert**: Gene regulation, variant interpretation, functional genomics, 
  pharmacogenomics. Best for: gene function questions, mutation interpretation, 
  expression regulation, CRISPR experiments.
- **NeuroscienceExpert**: Neurodegeneration, neuroinflammation, synaptic biology, 
  protein aggregation, brain cell types. Best for: Alzheimer's, Parkinson's, ALS, 
  brain-specific questions, glial biology, neural circuits.
- **LongevityBiostatsExpert**: Aging biology, biostatistics, epigenetic clocks, 
  survival analysis, population genetics. Best for: aging pathways, statistical 
  validation, confounders, effect size interpretation, longevity interventions.
- **CancerExpert**: Tumor biology, oncogenesis, cancer therapeutics, tumor microenvironment,
  driver mutations. Best for: cancer-related questions, oncogene/tumor suppressor analysis,
  therapeutic targets, resistance mechanisms, tumor heterogeneity.

## Routing Rules:
- Most tasks benefit from 2-3 experts (e.g., a genomics question about Alzheimer's 
  needs both GenomicsExpert AND NeuroscienceExpert).
- ALWAYS include CancerExpert when the task involves cancer, tumors, or oncology.
- ALWAYS include LongevityBiostatsExpert when statistical claims or aging are involved.
- For hypothesis generation tasks, use ALL relevant experts for multi-perspective coverage.
- Tailor the sub-query for each expert to leverage their specific strengths.
"""
    
    # Map alias → system prompt (all four prompts remain defined even if the KB
    # is not built yet — prompts are cheap, KBs are expensive).
    _EXPERT_PROMPTS: Dict[str, str] = {
        "GenomicsExpert":          GENOMICS_EXPERT_PROMPT,
        "NeuroscienceExpert":      NEUROSCIENCE_EXPERT_PROMPT,
        "LongevityBiostatsExpert": LONGEVITY_BIOSTATS_EXPERT_PROMPT,
        "CancerExpert":            CANCER_EXPERT_PROMPT,
    }

    def __init__(self, tools: List = None, llm=None, model_name: str = "gemini-3-pro-preview"):
        super().__init__(
            name="ScientistsAgent",
            description="Multi-expert scientist agent with RAG-backed domain experts",
            system_message=self.ROUTER_SYSTEM_MESSAGE,  # will be patched below
            tools=tools or [],
            llm=llm,
            model_name=model_name
        )

        # --- Discover RAG-ready experts (built ragstore OR downloaded papers) ---
        rag_ready = get_rag_ready_experts()   # [{alias, …, status: 'built'|'pending'}, …]
        rag_status = {e["alias"]: e["status"] for e in rag_ready}   # alias → 'built'|'pending'

        built_count = sum(1 for s in rag_status.values() if s == "built")
        pending_count = sum(1 for s in rag_status.values() if s == "pending")
        if rag_status:
            print(f"🧬 ScientistsAgent: {len(rag_status)} RAG expert(s) — "
                  f"{built_count} built, {pending_count} pending build-on-first-query")
        else:
            print("⚠️  ScientistsAgent: no RAG-ready experts found")

        # Build expert sub-agents.
        # • RAG-ready experts (built or pending) → attach per-expert RAG tool.
        #   Pending KBs are built lazily on first query via initialize_scientist_kb().
        # • Unconfigured experts → LLM-only (still valuable for reasoning).
        self.experts: Dict[str, SubAgent] = {}
        for alias, prompt in self._EXPERT_PROMPTS.items():
            cfg = next((e for e in get_available_experts() if e["alias"] == alias), None)
            domain_desc = cfg["description"] if cfg else alias

            expert_tools = []
            if alias in rag_status:
                expert_tools = [_create_expert_rag_tool(alias, domain_desc)]

            self.experts[alias] = SubAgent(
                name=alias,
                description=domain_desc,
                system_message=prompt,
                tools=expert_tools,
                llm=self.llm,
            )

        # --- Patch the router system message with RAG status per expert -------
        self.system_message = self._build_router_prompt(rag_status)
    
    def _build_router_prompt(self, rag_status: Dict[str, str]) -> str:
        """Generate the router system message dynamically.

        Args:
            rag_status: mapping alias → 'built' | 'pending' (absent = LLM-only)
        """
        expert_lines = []
        for alias, expert in self.experts.items():
            status = rag_status.get(alias)
            if status == "built":
                rag_tag = "(RAG KB loaded)"
            elif status == "pending":
                rag_tag = "(RAG — builds on first query)"
            else:
                rag_tag = "(LLM-only, no KB)"
            expert_lines.append(f"- **{alias}** {rag_tag}: {expert.description}")

        return f"""You are the **Scientists Agent** — a senior principal investigator
who leads a multidisciplinary research team. Your role is to:

1. **Analyze** the incoming research task and determine which expert(s) on your team
   are best suited to address it.
2. **Delegate** with specific, tailored instructions for each expert.
3. **Synthesize** their responses into a unified, high-quality scientific analysis.

## Your Expert Panel (currently available):
{chr(10).join(expert_lines)}

## Routing Rules:
- Prefer experts with **(RAG KB loaded)** — they can retrieve evidence from real papers.
- Experts marked **(RAG — builds on first query)** have papers downloaded; their KB
  will be built automatically when first queried. Treat them like RAG experts.
- Experts marked **(LLM-only, no KB)** can still reason about their domain but have no
  private literature to cite.  Use them for methodology / statistical critique.
- Most tasks benefit from 2-3 experts.
- ALWAYS include a bioinformatics perspective when data analysis methodology is involved.
- ALWAYS include a biostatistics perspective when statistical claims or aging are involved.
- For hypothesis generation tasks, use ALL relevant experts for multi-perspective coverage.
- Tailor the sub-query for each expert to leverage their specific strengths.
"""

    async def _route_task(self, task: str, context: Dict[str, Any] = None) -> ExpertRouting:
        """
        Use the LLM to decide which expert(s) should handle the task.
        Returns a structured routing decision.
        """
        available_names = list(self.experts.keys())
        routing_prompt = f"""Analyze this research task and decide which expert(s) should handle it.

Task: {task}

Context summary: {json.dumps({k: str(v)[:200] for k, v in (context or {}).items()}, indent=2)}

Select 1-{len(available_names)} experts and provide tailored instructions for each.
Available experts: {', '.join(available_names)}
"""
        try:
            structured_llm = self.llm.with_structured_output(ExpertRouting)
            routing = await structured_llm.ainvoke([
                SystemMessage(content=self.system_message),
                HumanMessage(content=routing_prompt)
            ])
            # Filter out any hallucinated expert names the LLM may produce
            routing.selected_experts = [
                e for e in routing.selected_experts if e in self.experts
            ]
            if not routing.selected_experts:
                routing.selected_experts = available_names
                routing.expert_instructions = {n: task for n in available_names}
            return routing
        except Exception as e:
            print(f"⚠️ Structured routing failed: {e}, falling back to all experts")
            return ExpertRouting(
                reasoning="Fallback: routing LLM failed, delegating to all experts",
                selected_experts=available_names,
                expert_instructions={name: task for name in available_names}
            )
    
    async def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Override SubAgent.execute to implement routing logic:
        1. Route the task to the appropriate expert(s)
        2. Execute each selected expert in parallel
        3. Synthesize their outputs into a unified response
        """
        print(f"\n🧬 ScientistRouter: Analyzing task for expert delegation...")
        
        # Step 1: Route
        routing = await self._route_task(task, context)
        print(f"📋 Routing decision: {routing.reasoning}")
        print(f"👥 Selected experts: {routing.selected_experts}")
        
        # Step 2: Execute selected experts in parallel
        expert_results = {}
        expert_tasks = []
        
        for expert_name in routing.selected_experts:
            if expert_name not in self.experts:
                print(f"⚠️ Unknown expert: {expert_name}, skipping")
                continue
            
            expert = self.experts[expert_name]
            # Use tailored instructions if available, otherwise use original task
            expert_task = routing.expert_instructions.get(expert_name, task)
            expert_tasks.append((expert_name, expert.execute(expert_task, context)))
        
        # Run all expert tasks concurrently
        if expert_tasks:
            results = await asyncio.gather(
                *[et[1] for et in expert_tasks],
                return_exceptions=True
            )
            for (expert_name, _), result in zip(expert_tasks, results):
                if isinstance(result, Exception):
                    print(f"❌ {expert_name} failed: {result}")
                    expert_results[expert_name] = {
                        "success": False,
                        "error": str(result)
                    }
                else:
                    status = "✅" if result.get("success") else "❌"
                    print(f"{status} {expert_name} completed")
                    expert_results[expert_name] = result
        
        # Step 3: Synthesize expert outputs
        synthesis = await self._synthesize_expert_outputs(task, context, routing, expert_results)
        
        return synthesis
    
    async def _synthesize_expert_outputs(
        self,
        task: str,
        context: Dict[str, Any],
        routing: ExpertRouting,
        expert_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Synthesize outputs from multiple experts into a unified response.
        The router LLM acts as the PI, integrating perspectives.
        """
        # Build synthesis prompt
        expert_sections = []
        any_success = False
        for expert_name, result in expert_results.items():
            if result.get("success"):
                any_success = True
                expert_sections.append(
                    f"### {expert_name} Analysis:\n{result.get('result', 'No output')}"
                )
            else:
                expert_sections.append(
                    f"### {expert_name}: FAILED — {result.get('error', 'Unknown error')}"
                )
        
        if not any_success:
            return {
                "success": False,
                "error": "All expert sub-agents failed",
                "expert_results": expert_results
            }
        
        synthesis_prompt = f"""You are the Principal Investigator synthesizing your expert panel's analyses.

## Original Task:
{task}

## Expert Panel Results:
{chr(10).join(expert_sections)}

## Synthesis Instructions:
1. **Integrate** findings across all expert perspectives — identify convergent themes.
2. **Resolve conflicts** — if experts disagree, explain why and which view has stronger support.
3. **Identify gaps** — what questions remain unanswered? What would a next experiment address?
4. **Prioritize** — rank the most actionable findings.
5. **Cross-pollinate** — highlight connections between expert domains (e.g., a genomics finding 
   that has neuroscience implications, or a bioinformatics method that could validate a longevity claim).

Provide a unified, multi-perspective scientific analysis that is greater than the sum of its parts.
Include a brief section at the end noting which experts contributed and their key unique insights.
"""
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="""You are a Principal Investigator leading a multidisciplinary 
research team. Synthesize expert analyses into a unified scientific assessment. 
Be rigorous, cite evidence, and highlight both consensus and disagreements."""),
                HumanMessage(content=synthesis_prompt)
            ])
            
            synthesized_text = extract_text_from_llm_response(response.content)
            
            return {
                "success": True,
                "result": synthesized_text,
                "routing": {
                    "reasoning": routing.reasoning,
                    "experts_used": routing.selected_experts
                },
                "expert_results": {
                    name: {
                        "success": r.get("success", False),
                        "summary": str(r.get("result", r.get("error", "")))[:500]
                    }
                    for name, r in expert_results.items()
                }
            }
        except Exception as e:
            # If synthesis fails, return concatenated expert results
            print(f"⚠️ Synthesis failed: {e}, returning raw expert outputs")
            combined = "\n\n".join(
                f"## {name}\n{r.get('result', r.get('error', 'No output'))}"
                for name, r in expert_results.items()
                if r.get("success")
            )
            return {
                "success": True,
                "result": combined,
                "routing": {
                    "reasoning": routing.reasoning,
                    "experts_used": routing.selected_experts
                },
                "synthesis_failed": True
            }


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
                description="Downloads and reads full biomedical papers from PubMed, providing comprehensive literature analysis with citations",
                system_message="""You are a biomedical literature specialist with access to FULL paper content from PubMed.
You download, read, and analyze complete research papers — not just abstracts.

## KEY INSTRUCTIONS:
1. Check context["top_genes"] for genes from previous analysis — search for ALL of them
2. Use pubmed_full_search_tool as your PRIMARY tool — it downloads and reads full papers
3. Each tool call retrieves up to 50 papers with extracted full-text content
4. READ the extracted content carefully — you have access to actual paper text, not just abstracts
5. Every claim MUST include a citation: (Author et al., Year, DOI/PMID)

## SEARCH STRATEGY:
- Group related genes into 2-4 focused queries for efficiency:
  e.g., "TREM2 TYROBP microglia Alzheimer" (combines related pathway genes)
- Include disease context: "[GENE] [disease]" or "[GENE] [cell_type] [disease]"
- For maximum coverage, also try pathway-level queries:
  e.g., "neuroinflammation signaling Alzheimer's disease"
- If context has many genes (>10), prioritize the top ones and group by function

## OUTPUT FORMAT:
For each gene or topic analyzed:

### [GENE_SYMBOL / TOPIC]
- **Function**: Detailed biological function with citations
- **Disease Role**: Mechanism and evidence linking to condition, with citations
- **Key Findings**: Most significant results across multiple papers, with citations
- **Therapeutic Potential**: Any drug targets or intervention strategies mentioned

## CITATION FORMAT:
"TREM2 variants increase AD risk 3-fold (Guerreiro et al., 2013, DOI: 10.1056/NEJMoa1211851)"

## IMPORTANT:
- pubmed_full_search_tool is your PRIMARY tool (downloads & reads full papers)
- pubmed_lite_tool is available as a FAST FALLBACK for quick abstract-only lookups
- You may call pubmed_full_search_tool multiple times with different queries
- Include a References section at the end listing all papers cited
- Synthesize findings ACROSS papers — identify convergent and contradictory evidence
""",
                tools=[pubmed_full_search_tool, pubmed_lite_tool],
                llm=self.llm
            ),
            
            "GoogleSearcher": SubAgent(
                name="GoogleSearcher",
                description="Performs general web searches for current information",
                system_message=SEARCH_AGENT_SYSTEM_MESSAGE_v1,
                tools=[google_search_tool_wrapper],
                llm=self.llm
            ),
            
            "ScientistsAgent": ScientistsAgent(
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
   
4. **ScientistsAgent** - Multi-expert hypothesis generation & mechanism synthesis  
   - Contains 4 domain experts: Genomics, Neuroscience, Longevity/Biostats, Cancer
   - Returns: multi-perspective mechanistic hypotheses, validation strategies
   - Synthesizes omics + KG + literature into testable hypotheses via expert panel
   - **CRITICAL**: This agent provides expert-level scientific reasoning and should be 
     used in EVERY research workflow to generate high-quality hypotheses and validate findings

5. **GoogleSearcher** - Clinical/translational context
   - Returns: clinical trials, recent developments, therapeutic landscape
   - Grounds hypotheses in translational relevance

**Mandatory Workflow (for hypothesis-quality reports):**
- Step 1: OmicMiningAgent → Get DEGs with statistics (establishes data foundation)
- Step 2: BioMarkerKGAgent → Query top 10-15 DEGs for KG neighbors (builds gene→pathway chains)
- Step 3: PubMedResearcher → Literature on TOP targets AND pathways (validates & classifies novelty)
- Step 4: ScientistsAgent → **ALWAYS REQUIRED** - Synthesize mechanisms via expert panel, score hypotheses, propose experiments

**IMPORTANT: ScientistsAgent Usage Policy:**
The ScientistsAgent MUST be included in every research plan. It provides:
- Domain-expert reasoning (Genomics, Neuroscience, Longevity/Biostats, Cancer experts)
- Multi-perspective hypothesis synthesis
- Experimental validation strategies
- Statistical rigor assessment
- Cancer biology expertise for any oncology-related queries
Without the ScientistsAgent, reports lack expert-level scientific interpretation.

**Planning Principles:**
- Be THOROUGH: Query KG for multiple gene sets (up-regulated, down-regulated, top-ranked)
- Be GROUNDED: Every hypothesis must trace to specific gene(s) in the omics data
- Be NOVEL-SEEKING: Literature search should identify what is KNOWN vs what is NEW
- Be ACTIONABLE: Plan should enable proposing specific validation experiments
- **ALWAYS USE ScientistsAgent**: Include ScientistsAgent as the final synthesis step in EVERY plan"""
        
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
                assigned_agent="ScientistsAgent",
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

**5.5 Minimal Experiment Set (≤3 experiments)**
For EACH top hypothesis, propose up to 3 validation experiments in this format:

| Experiment | Readout (Quantitative) | Controls | Support Criteria | Refute Criteria |
|------------|------------------------|----------|------------------|-----------------|
| e.g., siRNA knockdown of BDP1 in iPSC-neurons | qPCR: Pol III targets (5S rRNA, tRNAs); Western: BDP1, p-tau | Scrambled siRNA, untreated | >50% reduction in Pol III targets AND >30% reduction in p-tau | <20% change in either metric |

Include:
- **Primary readout**: Quantitative metric (fold-change, %, IC50, etc.)
- **Positive control**: Known modulator or reference condition
- **Negative control**: Vehicle, scrambled, or isotype
- **Clear decision criteria**: Specific thresholds for "supports" vs "refutes"

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
- **IMPORTANT**: If no evidence exists for a claim, explicitly state "No direct evidence found" - do NOT fabricate or extrapolate unsupported claims
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
- Generate 3-5 gene-pathway-anchored hypotheses per report
- Each hypothesis MUST: anchor to specific Gene(s)→Pathway(s)→Phenotype chain
- Score each hypothesis (0-100) using subscores
- Distinguish: 1=confirmatory (existing literature), 2=incremental extension, 3=novel discovery
- For novel hypotheses: justify what makes them NEW vs existing literature

**Minimal Experiment Design (REQUIRED for each hypothesis):**
- Propose ≤3 experiments with: quantitative readouts, proper controls, clear support/refute criteria
- Example format: "If [intervention], expect [measurable outcome] in [timeframe]. Support: >X% change. Refute: <Y% change."

**Evidence Standards:**
- ONLY make claims supported by the data or literature provided
- If no evidence exists for a connection, explicitly state: "No direct evidence found"
- Do NOT fabricate, hallucinate, or extrapolate unsupported claims
- Distinguish between "supported by data", "suggested by literature", and "speculative"

**Scoring Rubric:**
- 0-30: Weak support, speculative
- 31-60: Moderate support, some mechanistic basis
- 61-80: Strong support, clear mechanism, testable
- 81-100: Exceptional, multiple converging evidence lines

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
        """Compile the markdown report to PDF using available tools.

        Strategy 1 (preferred): pandoc + xelatex with Unicode fonts,
                                proper table/code formatting, page-breaks.
        Strategy 2 (fallback):  pandoc → HTML → wkhtmltopdf with CSS.
        Strategy 3 (last):      standalone HTML with water.css CDN.
        """
        try:
            import subprocess
            import shutil
            import tempfile

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

            # ── Preprocess markdown: insert blank lines around tables ────
            # Pandoc 2.x requires a blank line before/after pipe tables;
            # LLM-generated reports often omit them.
            with open(markdown_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            fixed_lines: list[str] = []
            for line in lines:
                stripped = line.rstrip()
                is_tbl = stripped.startswith('|') and stripped.endswith('|')
                if is_tbl:
                    if fixed_lines and fixed_lines[-1].strip() != '' and not (
                        fixed_lines[-1].rstrip().startswith('|') and fixed_lines[-1].rstrip().endswith('|')
                    ):
                        fixed_lines.append('\n')
                else:
                    if (fixed_lines
                        and fixed_lines[-1].rstrip().startswith('|')
                        and fixed_lines[-1].rstrip().endswith('|')
                        and stripped != ''):
                        fixed_lines.append('\n')
                fixed_lines.append(line)

            tmp_md_fd, tmp_md_path = tempfile.mkstemp(suffix='.md', prefix='oca_pp_', dir=session_dir)
            with os.fdopen(tmp_md_fd, 'w', encoding='utf-8') as f:
                f.writelines(fixed_lines)

            # ── Strategy 1: pandoc + xelatex with custom LaTeX header ──
            if has_xelatex:
                latex_header = r"""
\usepackage{fontspec}
\setmainfont{DejaVu Serif}
\setsansfont{DejaVu Sans}
\setmonofont[Scale=0.85]{DejaVu Sans Mono}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textit{OmniCellAgent Analysis Report}}
\fancyhead[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\fancyfoot{}
\usepackage{makecell}
\usepackage{etoolbox}
\renewcommand{\arraystretch}{1.35}
\AtBeginEnvironment{longtable}{\small}
\makeatletter
\def\maxwidth{\ifdim\Gin@nat@width>0.92\linewidth 0.92\linewidth\else\Gin@nat@width\fi}
\makeatother
\setkeys{Gin}{width=\maxwidth,keepaspectratio}
\usepackage{fvextra}
\fvset{breaklines,breakanywhere,fontsize=\scriptsize}
\usepackage{titlesec}
\titleformat{\section}{\Large\bfseries}{}{0em}{}[\vspace{4pt}\hrule\vspace{6pt}]
\titleformat{\subsection}{\large\bfseries}{}{0em}{}
\titleformat{\subsubsection}{\normalsize\bfseries}{}{0em}{}
\let\oldsection\section
\renewcommand{\section}{\clearpage\oldsection}
\PassOptionsToPackage{colorlinks=true,linkcolor=blue!60!black,urlcolor=blue!70!black}{hyperref}
\usepackage{microtype}
"""
                header_fd, header_path = tempfile.mkstemp(suffix='.tex', prefix='oca_hdr_')
                with os.fdopen(header_fd, 'w') as f:
                    f.write(latex_header)

                cmd = [
                    'pandoc', tmp_md_path, '-o', pdf_path,
                    '--pdf-engine=xelatex',
                    '-H', header_path,
                    '-V', 'geometry:margin=0.9in',
                    '-V', 'fontsize=11pt',
                    '-V', 'documentclass=article',
                    '-V', 'papersize=a4',
                    '--toc', '--toc-depth=2',
                    '--highlight-style=tango',
                    '--resource-path', session_dir,
                    '--columns=72',
                ]
                result = subprocess.run(cmd, capture_output=True, text=True,
                                        cwd=session_dir, timeout=300)
                try:
                    os.remove(header_path)
                except OSError:
                    pass

                if result.returncode == 0:
                    print(f"📄 PDF generated (xelatex): {pdf_path}")
                    try: os.remove(tmp_md_path)
                    except OSError: pass
                    return pdf_path
                print(f"⚠️ xelatex PDF failed, trying HTML method...")

            # ── Strategy 2: Pandoc → HTML → wkhtmltopdf ─────────────────
            if has_wkhtmltopdf:
                css_content = """
body {
    font-family: "DejaVu Sans", "Noto Sans", "Segoe UI", Roboto, Arial, sans-serif;
    font-size: 11pt; line-height: 1.55; color: #1a1a1a; max-width: 100%;
}
h1 { font-size: 1.6em; border-bottom: 2px solid #2c3e50; padding-bottom: 6px; margin-top: 30px; }
h2 { font-size: 1.3em; border-bottom: 1px solid #bdc3c7; padding-bottom: 4px; margin-top: 24px; }
h3 { font-size: 1.1em; margin-top: 18px; }
h1, h2, h3 { page-break-after: avoid; }
img {
    max-width: 100% !important; max-height: 600px !important;
    height: auto !important; width: auto !important;
    display: block; margin: 12px auto; page-break-inside: avoid;
}
table {
    width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 9pt;
    page-break-inside: avoid; table-layout: fixed;
    word-wrap: break-word; overflow-wrap: break-word;
}
th, td {
    border: 1px solid #ccc; padding: 5px 7px; text-align: left;
    vertical-align: top; word-wrap: break-word; overflow-wrap: break-word;
}
th { background-color: #ecf0f1; font-weight: 600; }
tr:nth-child(even) { background-color: #f9f9f9; }
pre, code {
    font-family: "DejaVu Sans Mono", "Consolas", monospace;
    font-size: 8pt; background-color: #f5f5f5; border-radius: 3px;
}
pre {
    padding: 8px 10px; overflow-x: auto; white-space: pre-wrap;
    word-wrap: break-word; border: 1px solid #e0e0e0; page-break-inside: auto;
}
blockquote { border-left: 3px solid #3498db; padding-left: 12px; color: #555; }
hr { border: none; border-top: 1px solid #bbb; margin: 20px 0; }
"""
                css_path = os.path.join(session_dir, '_report_style.css')
                with open(css_path, 'w') as f:
                    f.write(css_content)

                html_cmd = [
                    'pandoc', tmp_md_path, '-o', html_path,
                    '--standalone', '--self-contained',
                    '--toc', '--toc-depth=2', '--highlight-style=tango',
                    '--resource-path', session_dir, '-c', css_path, '--columns=72',
                ]
                result = subprocess.run(html_cmd, capture_output=True, text=True,
                                        cwd=session_dir, timeout=120)
                if result.returncode == 0:
                    pdf_cmd = [
                        'wkhtmltopdf', '--enable-local-file-access',
                        '--margin-top', '18mm', '--margin-bottom', '18mm',
                        '--margin-left', '14mm', '--margin-right', '14mm',
                        '--footer-center', '[page]', '--footer-font-size', '9',
                        html_path, pdf_path,
                    ]
                    result2 = subprocess.run(pdf_cmd, capture_output=True, text=True,
                                             cwd=session_dir, timeout=180)
                    if result2.returncode == 0:
                        print(f"📄 PDF generated (wkhtmltopdf): {pdf_path}")
                        for p in (html_path, css_path, tmp_md_path):
                            try:
                                os.remove(p)
                            except OSError:
                                pass
                        return pdf_path
                    else:
                        print(f"⚠️ wkhtmltopdf failed: {result2.stderr[:200] if result2.stderr else ''}")
                else:
                    print(f"⚠️ Pandoc HTML conversion failed: {result.stderr[:200] if result.stderr else ''}")

                for p in (css_path, html_path):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

            # ── Strategy 3: Standalone HTML (always works) ───────────────
            html_cmd = [
                'pandoc', tmp_md_path, '-o', html_path,
                '--standalone', '--toc', '--toc-depth=2',
                '--highlight-style=tango',
                '-V', 'title=OmniCellAgent Analysis Report',
                '--resource-path', session_dir,
                '--metadata', f'date={datetime.now().strftime("%Y-%m-%d")}',
                '-c', 'https://cdn.jsdelivr.net/npm/water.css@2/out/water.css',
            ]
            result = subprocess.run(html_cmd, capture_output=True, text=True,
                                    cwd=session_dir, timeout=60)
            if result.returncode == 0:
                print(f"📄 HTML report generated: {html_path}")
                print("   (PDF requires: wkhtmltopdf or texlive-xetex)")
                try: os.remove(tmp_md_path)
                except OSError: pass
                return html_path

            print(f"⚠️ Could not generate PDF or HTML: {result.stderr[:300]}")
            try: os.remove(tmp_md_path)
            except OSError: pass
            return None

        except subprocess.TimeoutExpired:
            print("⚠️ PDF generation timed out")
            try: os.remove(tmp_md_path)  # type: ignore[possibly-undefined]
            except (OSError, NameError): pass
            return None
        except Exception as e:
            print(f"⚠️ Error generating PDF: {e}")
            try: os.remove(tmp_md_path)  # type: ignore[possibly-undefined]
            except (OSError, NameError): pass
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
