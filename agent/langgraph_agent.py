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
import re
import time
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, Literal, TypedDict, Annotated
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# Gemini uses GEMINI_API_KEY; GOOGLE_API_KEY is only a fallback because it may hold a
# Custom-Search-only key (see GOOGLE_SEARCH_API_KEY in tools/google_search_tools)
if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Gemini API key: ")

import sys
# Add project root to Python path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import path config
from utils.path_config import get_path
from utils.usage_tracker import (UsageTracker, LLMUsageCallback, usage_label,
                                 set_current_tracker, get_current_tracker)

# Suppress INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("paperscraper").setLevel(logging.ERROR)

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk, SystemMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool
import operator

# Global session directory for tools (set when agent is initialized)
_GLOBAL_SESSION_DIR: Optional[str] = None
_GLOBAL_SESSION_ID: Optional[str] = None

# Import tool functions directly (these are clean async functions without AutoGen dependencies)
from tools.pubmed_tools.query_pubmed_tool import query_medical_research_async
from tools.pubmed_tools.get_papers_info_tools import NUM_PAPERS_TO_DOWNLOAD
from tools.google_search_tools.google_search_w3m import google_search, web_search_tool
# Lite PubMed search (no PDF download, just metadata + abstracts)
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from tools.scientist_rag_tools.scientist_tool import query_expert_kb, get_available_experts, get_built_experts, get_rag_ready_experts
from tools.gretriever_tools.gretriever_client import gretriever_tool

# Add omic_tools to path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools', 'omic_tools'))
from omic_fetch_analysis_workflow import omic_fetch_analysis_workflow as _omic_workflow
from tools.omic_tools.cell_type_lookup import available_cell_types, format_cell_types, read_lookup_metadata

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
    organ: Optional[str] = None,
    label: str = "disease",
    control_cell_type: Optional[str] = None,
    disease_cell_label: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run a single-cell RNA-seq (scRNA-seq) cohort analysis: gene expression
    profiling, differential expression, and KEGG pathway enrichment.

    DATA TYPE: This tool operates on a COHORT of scRNA-seq data drawn from
    OmniCellTOSG. Samples are metacells (SEACells aggregates of about 200 cells
    of one cell type from one source file), NOT single cells, donors or bulk
    RNA-seq libraries. Reports based on this tool's output MUST describe the
    analysis as single-cell at metacell level, not bulk.

    Args:
        disease: Disease name to analyze. Examples: "lung adenocarcinoma",
            "Alzheimer disease", "breast cancer". For label="disease", if no
            disease has this exact name, no analysis is run and the result lists
            the closest disease names.
        cell_type: Cell type filter. Examples: "microglial cell", "astrocyte",
            "acinar cell". Required for label="disease": without it no analysis
            is run, and the result lists the cell types available for the
            disease and organ.
        organ: Organ filter (IMPORTANT for memory efficiency). Examples:
            "lung", "brain", "breast", "pancreas".
        label: Column used to define the two groups for differential expression.
            - "disease" (default): disease cells vs normal cells that pass the
              same cell-type (or `control_cell_type`) and organ filters. Both
              groups come only from datasets that hold disease and normal cells;
              the normal cells are not matched by protocol, age or sex. The DE
              model adjusts for source dataset and for chrY expression (male RNA
              share), not for age. Requires `disease` to be set.
            - "gender": female vs male within the queried subset (e.g. within
              a disease cohort or within an organ).
            - "cell_type": cell-type-stratified comparison.
        control_cell_type: Optional cell type of the normal (control) group when
            it differs from `cell_type`, e.g. tumour cells, which have no normal
            counterpart with the same name. Only for label="disease".
        disease_cell_label: Optional original (source-study) label that the
            disease cells must also carry, e.g. "Malignant epithelial cell".
            Only for label="disease".

    Returns:
        Dict with analysis results including top genes, enrichment data,
        plot paths, and provenance fields:
            - `data_type`: always "single-cell RNA-seq (scRNA-seq) cohort"
            - `sample_unit`: what `num_samples` counts (metacells)
            - `top_genes_by_fdr`: the differentially expressed genes with lowest FDR
            - `enrichment_top_terms`: significant Enrichr terms (adjusted p < 0.05)
              of the up- and down-regulated lists, per library; an empty list
              means no term of that library was significant
            - `qc_highest_mean_expression_genes`: most abundant genes, a check
              of cell identity, not a disease signal
            - `requested_label`: the label you requested
            - `actual_label`: the label actually used for the DE comparison
              (may differ if automatic fallback kicked in)
            - `label_fallback_message`: human-readable explanation when the
              requested label had <2 classes and a fallback was chosen
              (None when no fallback was needed)
    """
    try:
        # Use the global session directory
        session_dir = get_current_session_dir()
        # Each cell-type run gets its own subfolder, so several OmicMiningAgent
        # tasks in one session (e.g. CASE_STUDY_COHORTS) do not overwrite each other.
        if cell_type:
            slug = re.sub(r"[^a-z0-9]+", "_", cell_type.split(";")[0].lower()).strip("_")
            suffix = "" if label == "disease" else f"_{label}"
            session_dir = os.path.join(session_dir, f"omic_{slug}{suffix}")
        # Set in code from CASE_STUDY_COHORTS, never by the LLM, so it stays out of
        # prompts, tool arguments and the tool result.
        keep_unknown_donor = (cell_type or "").lower() in case_study_keep_unknown_cell_types(disease)

        params = {
            "session_dir": session_dir,
            "enable_differential_expression": True,
            "enable_plotting": True,
            "label": label,
            "keep_unknown_donor": keep_unknown_donor,
        }

        if disease:
            params["disease"] = disease
        if cell_type:
            params["cell_type"] = cell_type
        if organ:
            params["organ"] = organ
        if control_cell_type:
            params["control_cell_type"] = control_cell_type
        if disease_cell_label:
            params["disease_cell_label"] = disease_cell_label

        if not disease and not cell_type:
            return {"success": False, "message": "Provide at least 'disease' or 'cell_type'"}

        # Pooling every cell type would mix disease effects with cell-type composition, so a
        # disease-vs-normal run needs a cell type and a database disease name, and the cell type
        # must meet the analysis minimum, which is what the lookup lists; otherwise return the
        # cell types or disease names to choose from instead. The lookup counts one cell type
        # for both groups, so a cohort with its own control cell type or disease-cell label is
        # left to the sampler, which enforces the same minimum.
        lookup = (_cell_type_lookup(disease, organ)
                  if (label or "disease").lower() == "disease" and disease else None)
        reason = None
        if lookup is not None:
            minimum = lookup["min_metacells"]
            listed = {name.lower() for name in lookup["cell_types"]["cell_type"]}
            if not lookup["disease_found"]:
                reason, problem = "disease_not_found", "the disease name was not found"
            elif not cell_type:
                reason, problem = ("cell_type_required",
                                   "a disease-vs-normal comparison needs one cell type, and none was given")
            elif not (control_cell_type or disease_cell_label) and cell_type.lower() not in listed:
                reason, problem = ("cell_type_below_minimum",
                                   f'cell type "{cell_type}" does not have at least {minimum} disease and '
                                   f"{minimum} normal metacells from datasets that hold both groups, "
                                   f"the minimum for an analysis")
        if reason:
            return {
                "success": False,
                "reason": reason,
                "message": f"No analysis was run: {problem}.\n\n" + format_cell_types(lookup),
                "cell_type_options": {
                    "disease": lookup["disease_name"] or disease,
                    "organ": organ,
                    "cell_types": lookup["cell_types"]["cell_type"].tolist(),
                    "disease_candidates": lookup["disease_candidates"],
                },
            }

        result = _omic_workflow(**params)
        # Cohort diagnostics stay in the session files (cohort_diagnostics.json,
        # COHORT_DIAGNOSTICS.txt); no agent, prompt or report receives them.
        if isinstance(result, dict):
            result.pop("cohort_diagnostics", None)
            result.pop("cohort_verdict", None)
            # The most abundant genes only confirm the cell type; pass them under a name that says so
            names = result.pop("top_gene_names", None) or []
            values = result.pop("top_gene_values", None) or []
            result.pop("top_gene_indices", None)
            if names:
                result["qc_highest_mean_expression_genes"] = {
                    "note": ("Highest mean expression (CP10K) over both groups: a check that the intended "
                             "cell type was retrieved, not a disease signal or differential expression."),
                    "genes": [{"gene": g, "mean_cp10k": round(float(v), 1)} for g, v in zip(names, values)],
                }
        return result

    except Exception as e:
        import traceback
        return {
            "success": False,
            "message": f"Error in omic analysis: {str(e)}",
            "traceback": traceback.format_exc()
        }


def _cell_type_lookup(disease: Optional[str], organ: Optional[str]) -> Dict[str, Any]:
    """available_cell_types for a disease and organ, counted the way omic_analysis_tool selects
    each cell type (keep_unknown_donor set from CASE_STUDY_COHORTS)."""
    metadata = read_lookup_metadata(os.path.join(
        get_path("external.omnicell_data_root", absolute=True), "cell_metadata_with_mappings.parquet"))
    return available_cell_types(metadata, disease, organ,
                                keep_unknown_donor=case_study_keep_unknown_cell_types(disease))


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
async def pubmed_full_search_tool(query: str, num_papers: int = NUM_PAPERS_TO_DOWNLOAD) -> str:
    """
    Full PubMed search — downloads papers and extracts full text content from PDFs/XMLs.
    Returns comprehensive paper information with abstracts AND extracted full-text for
    thorough literature review. Each paper includes metadata, abstract, and extracted content.
    
    This is the PRIMARY literature search tool. Use this for any literature review task.
    Papers are cached globally so repeated queries are fast.
    
    Args:
        query: Search query for PubMed (e.g., "TREM2 Alzheimer's disease microglia",
               "EGFR lung cancer treatment", '"TP53"[Title/Abstract]'). Put OR-lists in
               parentheses: "(TREM2 OR TYROBP) AND Alzheimer's disease"
        num_papers: Number of papers to download and read (default and max set by
                    NUM_PAPERS_TO_DOWNLOAD in tools/pubmed_tools/get_papers_info_tools.py)

    Returns:
        Formatted results with paper metadata, abstracts, and extracted full-text content
    """
    try:
        session_id = get_current_session_id()
        num_papers = min(max(num_papers, 1), NUM_PAPERS_TO_DOWNLOAD)
        
        print(f"[PubMed Full] Searching and downloading {num_papers} papers for: '{query}'")
        search_stats = {}
        papers = await query_medical_research_async(
            query=query,
            top_k=num_papers,
            use_llm_processing=True,  # Per-paper LLM summary of the full PDF/XML text
            max_concurrent=10,
            session_id=session_id,
            stats_out=search_stats
        )

        if search_stats:
            t, tokens = search_stats["timing_seconds"], search_stats["llm"]["tokens_total"]
            print(f"[PubMed Full] {t['total']}s total ({t['search_and_download']}s search+download, "
                  f"{t['summarization']}s summarization), {tokens['total_token_count']:,} summarizer tokens")
            tracker = get_current_tracker()
            if tracker is not None:
                tracker.record_pubmed_call(query, search_stats)

        if isinstance(papers, str):
            return f"Error retrieving papers: {papers}"
        if not papers:
            return f"No papers found for query: '{query}'"
        
        # Format results with full content
        CONTENT_TRUNCATE = None  # chars of extracted text per paper in output
        
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
                if CONTENT_TRUNCATE is not None and len(content) > CONTENT_TRUNCATE:
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
        result = str(await gretriever_tool(query))
    except Exception as e:
        result = f"Error in KG query: {str(e)}"
    _log_kg_call(query, result)
    return result


def _log_kg_call(query: str, result: str) -> None:
    """Append one knowledge-graph call to <session_dir>/kg_calls.jsonl (tools/logs/gr.json keeps only the last)."""
    try:
        with open(os.path.join(get_current_session_dir(), "kg_calls.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"),
                                     "query": query, "result": result}, ensure_ascii=False) + "\n")
    except OSError as error:
        print(f"[KG] Could not log the call: {error}")


def format_enrichment_terms(terms_by_direction: Dict[str, Any]) -> str:
    """Markdown lines for enrichment_top_terms of one omics run (see summarize_enrichment)."""
    if not terms_by_direction:
        return "  - no enrichment results\n"
    lines = []
    for direction in ("up", "down"):
        for library, terms in (terms_by_direction.get(direction) or {}).items():
            if not terms:
                lines.append(f"  - {direction}-regulated, {library}: no term with adjusted p < 0.05")
                continue
            listed = "; ".join(f"{t['term']} (adj. p {t['adjusted_p']:.2g}, {t['n_genes']} genes: "
                               f"{', '.join(t['genes'])})" for t in terms)
            lines.append(f"  - {direction}-regulated, {library}: {listed}")
    return "\n".join(lines) + "\n"


# ==============================================================================
# AGENT DEFINITIONS
# ==============================================================================

# Model of the sub-agents (OmicMiningAgent, BioMarkerKGAgent, PubMedResearcher, GoogleSearcher,
# ScientistsAgent and its experts)
DEFAULT_MODEL = "gemini-3.8-flash"
# Model of the planner, replanner, reporter and report revision
LEAD_MODEL = "gemini-4-argon"
LEAD_REASONING_EFFORT = "high"
# langchain_google_genai 3.1 cannot send thinking_level (google-ai-generativelanguage, up to 0.12.1, has no
# such field), so the lead model goes through Gemini's OpenAI-compatible endpoint, which maps
# reasoning_effort onto the thinking level (checked 2026-10-25: "high" thinks, "low" and no setting do not)
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Output tokens per lead-model reply, thinking included. Gemini 4 stops a normal reply at 32,768 tokens and a
# streamed one at 65,536; longer output ("long decode") is not available on the OpenAI-compatible endpoint
# (Gemini API docs, long decode, read 2026-10-25).
LEAD_MAX_OUTPUT_TOKENS = 65536


def create_llm(model_name: str = DEFAULT_MODEL):
    """
    Create an LLM instance based on the model name.
    Supports Google Gemini models by default.
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        LLM instance (type-agnostic)
    """
    # Pass the key explicitly: langchain_google_genai otherwise prefers GOOGLE_API_KEY
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)


def _hold_back_usage(chunk: ChatGenerationChunk, last_usage):
    """Strip a streamed chunk's usage report; return the chunk and the latest report seen so far."""
    usage = getattr(chunk.message, "usage_metadata", None)
    if not usage:
        return chunk, last_usage
    message = chunk.message.model_copy(update={"usage_metadata": None})
    return ChatGenerationChunk(message=message, generation_info=chunk.generation_info), usage


class LastUsageChatOpenAI(ChatOpenAI):
    """
    ChatOpenAI for Gemini's OpenAI-compatible endpoint, which sends running usage totals with every
    streamed chunk. LangChain adds the chunks' usage together, so one streamed reply was recorded as
    54M input tokens for a prompt of about 50k (LUAD_20261025 report re-run, 2026-10-25). Each stream
    here carries only its last usage report, on a final empty chunk.
    """

    def _stream(self, *args, **kwargs):
        last_usage = None
        for chunk in super()._stream(*args, **kwargs):
            chunk, last_usage = _hold_back_usage(chunk, last_usage)
            yield chunk
        if last_usage:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata=last_usage))

    async def _astream(self, *args, **kwargs):
        last_usage = None
        async for chunk in super()._astream(*args, **kwargs):
            chunk, last_usage = _hold_back_usage(chunk, last_usage)
            yield chunk
        if last_usage:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata=last_usage))


def create_lead_llm(model_name: str = LEAD_MODEL, reasoning_effort: str = LEAD_REASONING_EFFORT):
    """Gemini model for planning and reporting, at the given thinking level (none | low | medium | high).
    Long replies go through stream_long_reply: only a streamed reply can use all of LEAD_MAX_OUTPUT_TOKENS."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return LastUsageChatOpenAI(model=model_name, base_url=GEMINI_OPENAI_BASE_URL, api_key=api_key,
                      temperature=0, reasoning_effort=reasoning_effort,
                      max_tokens=LEAD_MAX_OUTPUT_TOKENS, stream_usage=True)


# finish_reason of a reply that stopped at the output cap (OpenAI-compatible endpoint, native Gemini)
_CUT_OFF_FINISH_REASONS = {"length", "max_tokens"}

# Appended to a report whose reply stopped at the output cap, so the reader sees that its end is missing
REPORT_CUT_OFF_NOTE = ("\n\n---\n\n**This report is incomplete: the model reached its output limit "
                       "(thinking included), so everything after this point is missing.**\n")


async def stream_long_reply(llm, messages, label: str):
    """
    Get one long reply from llm through a stream, which on Gemini 4 raises the output cap from 32,768
    to 65,536 tokens. Returns (message, cut_off); cut_off is True when the reply stopped at the cap.
    If the stream fails, the call is repeated without streaming, which caps the reply at 32,768 tokens.
    """
    try:
        response = None
        async for chunk in llm.astream(messages):
            response = chunk if response is None else response + chunk
        if response is None:
            raise ValueError("the stream returned nothing")
    except Exception as e:
        print(f"⚠️ Streaming the {label} failed ({type(e).__name__}: {str(e)[:200]}); repeating the call "
              f"without streaming, which caps the reply at 32,768 tokens")
        response = await llm.ainvoke(messages)
    finish_reason = str((response.response_metadata or {}).get("finish_reason") or "").lower()
    cut_off = finish_reason in _CUT_OFF_FINISH_REASONS
    if cut_off:
        print(f"⚠️ The {label} stopped at the model's output cap (finish_reason {finish_reason}); its end is missing")
    return response, cut_off


# Tool outputs that carry no data: no hits, an unreachable service, or an error the tool reports as text
# (google_search_w3m reports an HTTP 403 as status "no_results"). Matched against the start of the output.
_NO_DATA_OUTPUT = re.compile(
    r'^(No papers found|No good PubMed Result|Error |Error:|Connection error to GRetriever|'
    r'Timeout connecting to GRetriever|GRetriever error|Tool execution error|Tool \S+ not found)'
    r'|^\{"query": .{0,500}?"status": "(no_results|error)"', re.S)


# omic_analysis_tool results that ran nothing and list the values to choose from instead
_OPTION_REASONS = {"cell_type_required", "disease_not_found", "cell_type_below_minimum"}


def _cell_type_options(tool_outputs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The reason and options of the last omic_analysis_tool call that ran nothing because no
    cell type was given, the disease was not found or the cell type is below the analysis
    minimum (None if there is no such call)."""
    for call in reversed(tool_outputs or []):
        out = call["output"]
        if (call["tool"] == "omic_analysis_tool" and isinstance(out, dict)
                and out.get("reason") in _OPTION_REASONS):
            return {"reason": out["reason"], **out["cell_type_options"]}
    return None


def _tool_output_has_data(output: Any) -> bool:
    """False for empty results, error messages and failure dicts (e.g. omic_analysis_tool success=False)."""
    if output is None or (isinstance(output, (str, list, dict)) and not output):
        return False
    if isinstance(output, dict):
        return output.get("success", True) is not False
    return not _NO_DATA_OUTPUT.search(str(output).lstrip())


class SubAgent:
    """Base class for specialized sub-agents"""
    
    def __init__(self, name: str, description: str, system_message: str, tools: List, llm=None, model_name: str = DEFAULT_MODEL):
        self.name = name
        self.description = description
        self.system_message = system_message
        self.tools = tools
        # Accept any LLM instance or create one
        self.llm = llm if llm is not None else create_llm(model_name)
        if tools:
            self.llm_with_tools = self.llm.bind_tools(tools)
            # Same tool declarations but calls disabled, for the closing answer once the tool budget is spent
            self.llm_no_tool_calls = self.llm.bind_tools(tools, tool_choice="none")
        else:
            self.llm_with_tools = self.llm
            self.llm_no_tool_calls = self.llm

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
            n_tool_calls = 0
            n_with_data = 0
            tool_errors = []
            # Raw tool return values (e.g. the omics result dict), for structured hand-off;
            # the executor removes them before the result is stored or passed on as context
            tool_outputs = []

            while response.tool_calls and iteration < max_iterations:
                # Execute tool calls
                tool_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    # Find and execute the tool
                    tool_result, tool_error = await self._execute_tool(tool_name, tool_args)
                    n_tool_calls += 1
                    if tool_error:
                        tool_errors.append(tool_error)
                    else:
                        tool_outputs.append({"tool": tool_name, "args": tool_args, "output": tool_result})
                        if _tool_output_has_data(tool_result):
                            n_with_data += 1
                    tool_results.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    ))

                messages.extend(tool_results)

                # Get next response
                response = await self.llm_with_tools.ainvoke(messages)
                messages.append(response)
                iteration += 1

            # Tool budget spent with calls still pending: the last message has no answer text.
            # Drop the unexecuted calls and ask for the answer from the tool results gathered so far.
            hit_tool_cap = bool(response.tool_calls)
            if hit_tool_cap:
                messages[-1] = HumanMessage(content=(
                    f"The tool-call limit ({max_iterations} rounds) is reached; no more tools can be called. "
                    "Write your final answer now, using only the tool results above, and state which "
                    "parts of the task could not be completed."))
                response = await self.llm_no_tool_calls.ainvoke(messages)
                messages.append(response)

            result_text = extract_text_from_llm_response(response.content)
            run_info = {"iterations": iteration, "tool_calls": n_tool_calls,
                        "tool_errors": len(tool_errors), "tool_calls_with_data": n_with_data,
                        "hit_tool_cap": hit_tool_cap, "tool_outputs": tool_outputs}

            # Report failure (which triggers retry/re-planning) instead of passing on an empty
            # answer or one written without any tool data (it could only come from model memory)
            if not result_text.strip():
                return {"success": False, "result": "", **run_info,
                        "error": f"{self.name} produced no answer after {iteration} tool rounds "
                                 f"({n_tool_calls} tool calls, {len(tool_errors)} failed)"}
            if n_tool_calls and n_with_data == 0:
                if tool_errors:
                    detail = f"first error: {tool_errors[0][:300]}"
                else:
                    detail = f"first output: {str(tool_outputs[0]['output'])[:300]}"
                return {"success": False, "result": result_text, **run_info,
                        "error": f"None of the {n_tool_calls} tool calls returned data "
                                 f"({len(tool_errors)} raised errors); {detail}"}

            return {"success": True, "result": result_text, **run_info}
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def _execute_tool(self, tool_name: str, tool_args: Dict) -> tuple:
        """Execute a specific tool by name. Returns (result, error message or None)."""
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    # Always use ainvoke for consistency with async execution
                    return await tool.ainvoke(tool_args), None
                except Exception as e:
                    # Fallback to sync invoke if async fails
                    try:
                        return tool.invoke(tool_args), None
                    except Exception as e2:
                        # Report the async error: for async-only tools e2 is just "sync not supported"
                        msg = f"Tool execution error: {type(e).__name__}: {e}"
                        return msg, msg
        msg = f"Tool {tool_name} not found"
        return msg, msg


# ==============================================================================
# SCIENTIST EXPERT ROUTING SYSTEM
# ==============================================================================

# Pydantic model for structured routing decisions
class ExpertRouting(BaseModel):
    """Structured routing decision from the ScientistsAgent"""
    reasoning: str = Field(description="Brief reasoning for why these experts were selected")
    selected_experts: PyList[str] = Field(
        description="List of expert names to delegate to. Choose from: GenomicsExpert, NeuroscienceExpert, LongevityBiostatsExpert, CancerExpert"
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
- Flag potential confounders explicitly (e.g., "cell-type composition or capture efficiency may confound scRNA-seq DE in this cohort").
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

    def __init__(self, tools: List = None, llm=None, model_name: str = DEFAULT_MODEL):
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

# Hard-coded case-study cohorts. When a query is about one of these diseases and
# names no cell type, the planner runs one OmicMiningAgent task per cell type, in
# this order. Names are the exact OmniCellTOSG disease / organ / CMT_name strings.
# A plain string uses that cell type for both groups. A dict gives the cancer-cell
# cohorts' separate control cell type and, for PDAC, the original label of the
# disease cells, as chosen in case_study_handoff_20260923. All nine handoff cohorts
# reproduce with identical metacells, except the LUAD cancer-cell controls: the handoff
# drew 1,000 of 2,134 type II pneumocyte metacells at random, the sampler draws its own 1,000.
CASE_STUDY_COHORTS = {
    "Alzheimer's Disease": {
        "organ": "brain",
        "cell_types": ["microglial cell", "astrocyte", "glutamatergic neuron"],
    },
    "lung adenocarcinoma": {
        "organ": "lung",
        "cell_types": [
            {"cell_type": "malignant cell",
             "control_cell_type": "type II pneumocyte;type II pneumocyte:proliferating"},
            "fibroblast;fibroblast:fibromyocyte",
            "macrophage;macrophage:monocyte-derived;macrophage:1;macrophage:2;macrophage:3",
        ],
    },
    # The PDAC disease metacells come from GSE202051, which records no donor IDs.
    # The fibroblast label covers the handoff's CAF cohort. omic_analysis_tool applies
    # keep_unknown_donor itself; the planner prompt does not show it.
    "pancreatic ductal adenocarcinoma": {
        "organ": "pancreas",
        "cell_types": [
            {"cell_type": "epithelial cell", "disease_cell_label": "Malignant epithelial cell",
             "control_cell_type": "pancreatic ductal cell"},
            "fibroblast;fibroblast:fibromyocyte",
            "macrophage;macrophage:monocyte-derived;macrophage:1;macrophage:2;macrophage:3",
        ],
        "keep_unknown_donor": True,
    },
}


def cohort_cell_types(cohort: Dict[str, Any]) -> List[Dict[str, str]]:
    """A cohort's cell-type entries as dicts; a plain string means one cell type for both groups."""
    return [entry if isinstance(entry, dict) else {"cell_type": entry}
            for entry in cohort.get("cell_types", [])]


def case_study_keep_unknown_cell_types(disease: Optional[str]) -> set:
    """Lower-cased cell types whose keep_unknown_donor is on: the listed cell types of the
    disease's CASE_STUDY_COHORTS entry when it sets the flag, otherwise none."""
    cohort = next((c for name, c in CASE_STUDY_COHORTS.items()
                   if disease and name.lower() == disease.strip().lower()), {})
    if not cohort.get("keep_unknown_donor"):
        return set()
    return {entry["cell_type"].lower() for entry in cohort_cell_types(cohort)}


class LangGraphOmniCellAgent:
    """
    LangGraph-based OmniCellAgent with planning, sub-agent execution, re-planning, and reporting.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL, log_dir: str = None, session_id: str = None, llm=None,
                 lead_model_name: str = LEAD_MODEL, lead_llm=None):
        """
        Initialize the LangGraph agent system.

        Args:
            model_name: LLM of the sub-agents (default: DEFAULT_MODEL)
            log_dir: Directory to save logs
            session_id: Unique session identifier
            llm: Optional pre-configured sub-agent LLM instance (any type)
            lead_model_name: LLM of the planner, replanner, reporter and report revision (default: LEAD_MODEL,
                at LEAD_REASONING_EFFORT)
            lead_llm: Optional pre-configured LLM instance for those steps
        """
        if log_dir is None:
            log_dir = get_path('logs.base', absolute=True, create=True)
        
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        self.session_id = session_id
        self.log_dir = log_dir
        self.model_name = model_name
        self.lead_model_name = lead_model_name

        # Create the session directory and set it globally for tools
        self.session_dir = create_session_dir(self.session_id)
        
        print(f"🔧 Initializing LangGraph OmniCellAgent")
        print(f"📁 Session ID: {self.session_id}")
        print(f"📂 Session Directory: {self.session_dir}")
        
        # Initialize LLMs - accept any LLM instance or create one
        self.llm = llm if llm is not None else create_llm(model_name)
        self.lead_llm = lead_llm if lead_llm is not None else create_lead_llm(lead_model_name)
        print(f"🤖 Models: {model_name} (sub-agents), {lead_model_name} (planner, replanner, reporter)")

        # Token/time usage of this session, saved to <session_dir>/usage_log.json. The callback sits on
        # both LLM instances, so it sees every call (bind_tools / structured output too).
        self.usage_tracker = UsageTracker(self.session_id, model=model_name, lead_model=lead_model_name)
        self._usage_callback = LLMUsageCallback(self.usage_tracker)
        for model in [self.llm] + ([self.lead_llm] if self.lead_llm is not self.llm else []):
            existing_callbacks = model.callbacks
            if hasattr(existing_callbacks, "add_handler"):
                existing_callbacks.add_handler(self._usage_callback)
            else:
                model.callbacks = list(existing_callbacks or []) + [self._usage_callback]
        set_current_tracker(self.usage_tracker)

        # Initialize sub-agents
        self._init_sub_agents()
        
        # Build the graph
        self.graph = self._build_graph()
        
    def _init_sub_agents(self):
        """Initialize all sub-agents"""
        
        self.sub_agents = {
            "OmicMiningAgent": SubAgent(
                name="OmicMiningAgent",
                description="Analyzes a cohort of single-cell RNA-seq (scRNA-seq) data from OmniCellTOSG for gene expression, biomarkers, and pathway enrichment. Inputs are scRNA-seq metacells (about 200 cells each), NOT bulk RNA-seq.",
                system_message="""You are an omics data specialist working with a COHORT of SINGLE-CELL RNA-seq (scRNA-seq) data from OmniCellTOSG. Samples are metacells (SEACells aggregates of about 200 cells of one cell type from one source file), NOT single cells, donors or bulk RNA-seq libraries. Any summary or report you write MUST describe the analysis as single-cell at metacell level — never call it bulk, and never call the samples individual cells.

Your task is to answer gene/biomarker questions using the omic_analysis tool.

## STEP 1: EXTRACT PARAMETERS FROM THE QUERY
From the user's question, identify:
- Disease name (MUST match the variations list below)
- Organ (REQUIRED for efficiency)
- Cell type (OPTIONAL, if mentioned):
  - If the TASK description gives a cell type in double quotes, pass that exact
    string as `cell_type`, even when the user's query names no cell type. Run one
    tool call for that cell type only.
  - Otherwise, if the TASK or the user's query names a cell type, pass it as
    `cell_type`. Never drop the named cell type.
  - If no cell type is named at all and label is "disease", do not choose one.
    Call omic_analysis_tool without `cell_type`: it runs no analysis and returns
    the cell types available for the disease and organ. Report the returned list
    in your answer and stop; the planner then chooses the cell types.
  - If omic_analysis_tool returns `reason: "disease_not_found"`, it ran nothing and
    lists the closest disease names. If one of them is the same disease, call it
    once more with that name copied verbatim and the other arguments unchanged;
    otherwise report the listed names and stop.
  - If omic_analysis_tool returns `reason: "cell_type_below_minimum"`, it ran nothing:
    the named cell type is not among the cell types with enough metacells for an
    analysis, which it lists. If one listed name is the same cell type under its
    database spelling, call it once more with that name copied verbatim and the other
    arguments unchanged; otherwise do not pick another cell type: report the list and stop.
  - For label "gender" or "cell_type", set no `cell_type` unless one is named.
- Control cell type (OPTIONAL). If the TASK description gives a control cell type
  in double quotes, pass that exact string as `control_cell_type`: the normal
  group then uses that cell type instead of `cell_type`.
- Original disease-cell label (OPTIONAL). If the TASK description gives an original
  label for the disease cells in double quotes, pass that exact string as
  `disease_cell_label`.
- Label column for the comparison (defaults to "disease"; set to "gender" if
  the question is specifically about female-vs-male differences within a cohort)

## DISEASE NAME VARIATIONS (use EXACT names — verified against OmniCellTOSG DB):
- Alzheimer's → "Alzheimer's disease"   (note the apostrophe; "Alzheimer disease" without apostrophe returns NO match)
- Lung cancer / NSCLC → "lung adenocarcinoma"
- Breast cancer → "breast cancer"
- Colon cancer → "colorectal cancer"
- Pancreatic cancer / PDAC → "pancreatic ductal adenocarcinoma"

## ORGAN MAPPING (use EXACT names):
- Alzheimer's → "brain"
- Lung cancer → "lung"
- Breast cancer → "breast"

## RETRY ON FAILURE (use the tool's own suggestions)
If the tool returns `success: false`, inspect the result for BOTH:
- `similar_terms`: dict of soft-matching adjustments (rarely present on hard miss)
- `suggestions`: a text block listing nearby valid terms from the DB,
  e.g. "disease: 'Alzheimer disease' -> try ['Alzheimer's disease', ...]"
On the next attempt, pick ONE alternative from `suggestions` (verbatim — keep
the apostrophe / capitalization exactly as listed) and retry. Do this at most
once per failed call to avoid loops.
Exception: a result with `reason: "cell_type_required"` is not an error to fix by
choosing a cell type; report its list and stop, as STEP 1 says. Results with
`reason: "disease_not_found"` or `reason: "cell_type_below_minimum"` are handled as
STEP 1 says.

## LABEL COLUMN (default "disease")
- label="disease" (default) → disease metacells vs normal metacells that pass the same
  cell-type (or control_cell_type) and organ filters, both drawn only from datasets that
  hold both groups; the normal metacells are not matched by protocol, age or sex. The DE
  model adjusts for source dataset and chrY expression. Requires `disease`.
- label="gender" → female vs male comparison within the queried subset.
- label="cell_type" → cell-type-stratified comparison.

## AUTOMATIC LABEL FALLBACK
If the requested label has only one non-empty class in the fetched cohort
(e.g., a disease subset that turned out to be all female), the tool will
automatically fall back to another viable label and run DE on that. Check
the result fields:
- `requested_label`: what you asked for
- `actual_label`: what was actually used for the DE comparison
- `label_fallback_message`: human-readable explanation (only set when fallback occurred)

After tool calls complete, write a concise summary of the gene list and pathway
data. Explicitly state that the data is single-cell RNA-seq (scRNA-seq) at metacell
level, give sample counts as metacells, and report `actual_label` (the column that
drove the comparison). If a fallback occurred, surface `label_fallback_message` in
your summary.
- Pathways: report only the terms in `enrichment_top_terms`, per direction and library,
  with their adjusted p-values. An empty list means no term of that library reached
  adjusted p < 0.05; say so. Never present one gene's known function as an enriched pathway.
- `qc_highest_mean_expression_genes` only confirms the cell type. Do not present these
  genes as differentially expressed, discriminating or disease-related.""",
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
3. Each tool call retrieves up to """ + str(NUM_PAPERS_TO_DOWNLOAD) + """ papers with extracted full-text content
4. READ the extracted content carefully — you have access to actual paper text, not just abstracts
5. Every claim MUST include a citation: (Author et al., Year, DOI/PMID)

## SEARCH STRATEGY:
- Queries use PubMed syntax. Words separated by spaces are ANDed. OR binds only its two neighbouring
  terms, so ALWAYS put OR-lists in parentheses:
  RIGHT: "(TREM2 OR TYROBP OR APOE) AND Alzheimer's disease"
  WRONG: "TREM2 OR TYROBP OR APOE Alzheimer" (matches any TREM2 or TYROBP paper on any disease)
- Group related genes into 2-4 focused queries for efficiency, each an OR-list in parentheses
  ANDed with the disease (and cell type if useful)
- Include disease context: "[GENE] [disease]" or "[GENE] [cell_type] [disease]"
- If a gene has no papers with the disease, say so in your answer. Papers on the gene in other
  diseases may describe its function, but label them as such: they are not evidence of a role
  in this disease
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
    
    @staticmethod
    def _with_usage_label(label: str, node_fn):
        """Wrap a graph node so its LLM calls are attributed to `label` in the usage log."""
        async def labeled_node(state: AgentState) -> Dict[str, Any]:
            with usage_label(label):
                return await node_fn(state)
        return labeled_node

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine"""
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes (LLM usage inside each node is attributed to the node name, or to the sub-agent
        # label set in _execution_node)
        workflow.add_node("planner", self._with_usage_label("planner", self._planning_node))
        workflow.add_node("executor", self._with_usage_label("executor", self._execution_node))
        workflow.add_node("replanner", self._with_usage_label("replanner", self._replanning_node))
        workflow.add_node("reporter", self._with_usage_label("reporter", self._reporting_node))
        
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

        def describe(entry):
            text = f'cell type "{entry["cell_type"]}"'
            if entry.get("disease_cell_label"):
                text += f', disease cells with original label "{entry["disease_cell_label"]}"'
            if entry.get("control_cell_type"):
                text += f', control cell type "{entry["control_cell_type"]}"'
            return text

        fixed_cohorts = "\n".join(
            f'- {disease} (organ "{c["organ"]}"):\n'
            + "\n".join(f"    {i}. {describe(entry)}" for i, entry in enumerate(cohort_cell_types(c), 1))
            for disease, c in CASE_STUDY_COHORTS.items()
        )

        planning_prompt = f"""Create a comprehensive research plan for hypothesis-driven discovery:

Query: {state['query']}

**Goal**: Generate thorough Gene→Pathway→Phenotype grounded hypotheses with literature validation.

**FIXED CELL-TYPE COHORTS (MANDATORY — overrides any other omics instruction):**
If the query is about one of the diseases below and does NOT name a cell type, the plan
MUST begin with one separate OmicMiningAgent task per listed cell type, in the listed
order, before any other agent. Each task description MUST contain the exact disease,
organ and cell-type strings in double quotes, e.g.
  'Differential expression for disease "Alzheimer's Disease", organ "brain",
   cell type "microglial cell": disease vs normal cells of the same cell type'
When a listed cell type also gives a control cell type and/or an original label for the
disease cells, the disease and control groups are selected by different names: copy
those strings into the same task description, exactly and in double quotes, e.g.
  'Differential expression for disease "lung adenocarcinoma", organ "lung",
   cell type "malignant cell", control cell type "type II pneumocyte;type II pneumocyte:proliferating":
   disease cells vs normal control cells'
Never merge the cell types into one task and never add or drop a cell type. All later
tasks (BioMarkerKGAgent, PubMedResearcher, ScientistsAgent, ...) must use the DEGs from
every cell type and compare the cell types.
{fixed_cohorts}

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
            structured_llm = self.lead_llm.with_structured_output(ResearchPlan)
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
                response = await self.lead_llm.ainvoke([
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
        
        # An unknown agent name fails the task at once (no retry), so the replanner reassigns it;
        # running it on some other agent would pass that agent's output off as the task's result
        if agent_name not in self.sub_agents:
            print(f"❌ Unknown agent: {agent_name}")
            current_task["status"] = "failed"
            current_task["attempts"] = max(current_task["attempts"], 2)
            current_task["error"] = (f"Unknown agent '{agent_name}'; valid agents: "
                                     f"{', '.join(self.sub_agents)}")
            return {
                "plan": plan,
                "current_task_index": current_idx,
                "process_log": state.get("process_log", []) + [{
                    "phase": "execution", "timestamp": datetime.now().isoformat(),
                    "task_id": current_task["id"], "agent": agent_name, "success": False,
                    "result_summary": current_task["error"]}],
            }

        sub_agent = self.sub_agents[agent_name]
        
        # Prepare context from previous task results, including structured shared_data
        shared_data = state.get("shared_data", {})
        # Gene lists go in once (top_genes, annotated with cell type), not again inside shared_data
        # and each omics run record: the context is resent on every LLM call of the sub-agent
        context_shared = {k: v for k, v in shared_data.items() if k != "top_genes"}
        if "omic_runs" in context_shared:
            context_shared["omic_runs"] = [{k: v for k, v in r.items() if k != "top_genes"}
                                           for r in context_shared["omic_runs"]]
        context = {
            "query": state["query"],
            "session_id": state["session_id"],
            "previous_results": state.get("agent_outputs", {}),
            # Structured data for programmatic access by downstream agents
            "shared_data": context_shared,
            "top_genes": shared_data.get("top_genes", []),  # Explicit gene list
            "paper_dois": shared_data.get("paper_dois", []),  # Explicit DOI list
        }
        
        # Execute the task
        task_start = time.perf_counter()
        with usage_label(agent_name):
            result = await sub_agent.execute(current_task["description"], context)
        task_seconds = time.perf_counter() - task_start
        self.usage_tracker.record_task(current_task["id"], agent_name, task_seconds,
                                       result.get("success", False))

        # Structured tool outputs are kept out of agent_outputs (they are passed to every later
        # agent as context); they are read here even if the task failed, since the tool data are real
        tool_outputs = result.pop("tool_outputs", [])
        shared_data = self._extract_structured_data(
            agent_name, result, dict(state.get("shared_data", {})), tool_outputs)

        # Process result
        process_entry = {
            "phase": "execution",
            "timestamp": datetime.now().isoformat(),
            "task_id": current_task["id"],
            "agent": agent_name,
            "task_description": current_task["description"],
            "seconds": round(task_seconds, 2),
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
            # No cell type was given, the disease was not found or the cell type is below the
            # analysis minimum: rerunning the task cannot help, so it goes straight to the
            # replanner, which receives the values to choose from
            options = _cell_type_options(tool_outputs) if agent_name == "OmicMiningAgent" else None
            if options is not None:
                current_task["cell_type_options"] = options
                current_task["attempts"] = max(current_task["attempts"], 2)
                current_task["error"] = {
                    "cell_type_required": "No cell type was given, so no analysis was run: a "
                                          "disease-vs-normal comparison needs one cell type",
                    "disease_not_found": "The disease name was not found, so no analysis was run",
                    "cell_type_below_minimum": "The cell type has too few metacells for an analysis, "
                                               "so no analysis was run",
                }[options["reason"]]

            return {
                "plan": plan,
                "current_task_index": current_idx,
                "shared_data": shared_data,
                "process_log": state.get("process_log", []) + [process_entry],
                "messages": [AIMessage(content=f"Task {current_task['id']} failed: {result.get('error', '')}")]
            }
    
    def _extract_structured_data(self, agent_name: str, result: Dict[str, Any], shared_data: Dict[str, Any],
                                 tool_outputs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Extract structured data from agent results for programmatic use by downstream agents.

        This enables cross-agent data sharing without relying on LLM parsing of free-text context.

        Args:
            agent_name: Name of the agent that produced the result
            result: The result dictionary from the agent
            shared_data: Current shared_data dictionary to update
            tool_outputs: Raw tool return values from SubAgent.execute ({"tool", "args", "output"})

        Returns:
            Updated shared_data dictionary
        """
        result_content = result.get("result", "")

        if agent_name == "OmicMiningAgent":
            # The agent's text answer is prose; the omics numbers come from the tool's result dict.
            # One record per cohort (CASE_STUDY_COHORTS runs one task per cell type).
            omic_runs = list(shared_data.get("omic_runs", []))
            for call in tool_outputs or []:
                out = call["output"]
                # A refused call ran nothing (it only listed cell types or disease names): not a
                # cohort; the failed task carries the refusal
                if (call["tool"] != "omic_analysis_tool" or not isinstance(out, dict)
                        or out.get("reason") in _OPTION_REASONS):
                    continue
                entities = out.get("extracted_entities") or {}
                run = {
                    "disease": entities.get("disease"),
                    "cell_type": entities.get("cell type"),
                    "control_cell_type": entities.get("control cell type"),
                    "disease_cell_label": entities.get("disease cell label"),
                    "organ": entities.get("organ"),
                    "label": out.get("actual_label"),
                    "contrast": out.get("contrast"),
                    "num_samples": out.get("num_samples"),
                    "de_success": out.get("de_success"),
                    "error": None if out.get("success", True) else out.get("message"),
                    "top_genes": (out.get("top_genes_by_fdr") or [])[:25],
                    "enrichment_top_terms": out.get("enrichment_top_terms") or {},
                }
                # A retried task replaces its earlier record for the same cohort
                fields = ("disease", "cell_type", "control_cell_type", "disease_cell_label",
                          "organ", "label")
                key = tuple(run[f] for f in fields)
                omic_runs = [r for r in omic_runs if tuple(r.get(f) for f in fields) != key]
                omic_runs.append(run)
                print(f"[SharedData] Omics run {run['cell_type'] or run['disease']}: "
                      f"{len(run['top_genes'])} top genes")
            if omic_runs:
                shared_data["omic_runs"] = omic_runs
                # Flat gene list for downstream agents: the top genes of every cohort, first occurrence kept
                seen, top_genes = set(), []
                for run in omic_runs:
                    for gene in run["top_genes"]:
                        if gene.get("gene_name") not in seen:
                            seen.add(gene.get("gene_name"))
                            top_genes.append({**gene, "cell_type": run["cell_type"]})
                shared_data["top_genes"] = top_genes
                shared_data["disease"] = omic_runs[-1]["disease"] or shared_data.get("disease", "")
                shared_data["cell_type"] = "; ".join(r["cell_type"] for r in omic_runs if r["cell_type"])

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
        
        # Failed tasks not yet replaced; those replaced in an earlier round stay in the plan (marked
        # "replaced") so the reporter can state which evidence is missing
        failed_tasks = [t for t in state["plan"] if t["status"] == "failed" and not t.get("replaced")]
        earlier_failed = [t for t in state["plan"] if t["status"] == "failed" and t.get("replaced")]
        completed_tasks = [t for t in state["plan"] if t["status"] == "completed"]
        # Tasks not yet run are kept as planned and run after the replacements
        pending_tasks = [t for t in state["plan"] if t["status"] == "pending"]
        agent_list = "\n".join(f"- {name}: {getattr(agent, 'description', '')}"
                               for name, agent in self.sub_agents.items())
        failed_info = [{"id": t["id"], "assigned_agent": t["assigned_agent"], "description": t["description"],
                        "error": t.get("error", ""),
                        **({"cell_type_options": t["cell_type_options"]} if t.get("cell_type_options") else {})}
                       for t in failed_tasks]

        replanning_prompt = f"""A task has failed. Write replacement tasks for the failed task(s).

Original Query: {state['query']}

Completed Tasks:
{json.dumps([{"id": t["id"], "description": t["description"], "result": t.get("result", "")[:200]} for t in completed_tasks], indent=2)}

Failed Tasks:
{json.dumps(failed_info, indent=2)}

Tasks still to run after your replacements (kept unchanged; do not repeat them):
{json.dumps([{"id": t["id"], "assigned_agent": t["assigned_agent"], "description": t["description"]} for t in pending_tasks], indent=2)}

Available agents (assigned_agent MUST be one of these exact names):
{agent_list}

Rules:
1. Replace only the failed task(s), using a different agent or a changed approach that avoids the error.
2. If a failure cannot be worked around (e.g. no literature exists, a service is down), return no
   replacement for it; the final report will state that it failed.
3. Do not add synthesis, summary or report-writing tasks; the reporting step does that.
4. A failed OmicMiningAgent task with `cell_type_options` ran no analysis:
   - `reason` "cell_type_required": it named no cell type, and a disease-vs-normal comparison
     needs one. Replace it with one OmicMiningAgent task per cell type you choose: up to 3 cell
     types from `cell_type_options.cell_types`, the ones most relevant to the original query. Each
     task description MUST contain the exact disease, organ and cell-type strings in double
     quotes, copied verbatim from `cell_type_options` (leave the organ out if it is null), e.g.
       'Differential expression for disease "<disease>", organ "<organ>", cell type "<cell type>":
        disease vs normal cells of the same cell type'
     If `cell_type_options.cell_types` is empty, return no replacement for that task.
   - `reason` "disease_not_found": no disease has the task's disease name. If one name in
     `cell_type_options.disease_candidates` is the same disease as the original query, replace the
     task with one OmicMiningAgent task that gives that name verbatim in double quotes, with the
     failed task's organ and cell type (if any); otherwise return no replacement for that task.
   - `reason` "cell_type_below_minimum": the task's cell type does not have enough metacells for an
     analysis; `cell_type_options.cell_types` lists the ones that do. If the original query names
     that cell type, replace the task only with a listed name for the same cells, in the format
     above; otherwise return no replacement, so the report states it could not be analysed. If the
     original query names no cell type, replace it as for "cell_type_required", choosing listed
     cell types that no other task already covers.

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
        
        response = await self.lead_llm.ainvoke([
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
            
            # Create new tasks (tasks naming an agent that does not exist are dropped)
            new_tasks = []
            dropped = []
            for task in replan_data.get("tasks", []):
                if task.get("assigned_agent") not in self.sub_agents:
                    dropped.append(task)
                    print(f"⚠️ Dropped replanned task {task.get('id')}: unknown agent {task.get('assigned_agent')}")
                    continue
                new_tasks.append(SubTask(
                    id=task["id"],
                    description=task["description"],
                    assigned_agent=task["assigned_agent"],
                    status="pending",
                    result=None,
                    error=None,
                    attempts=0
                ))
            
            # Done and failed tasks first (not run again), then the replacements, then the tasks not yet run
            for t in failed_tasks:
                t["replaced"] = True
            done = completed_tasks + earlier_failed + failed_tasks
            final_plan = done + new_tasks + pending_tasks

            print(f"📝 Revised plan with {len(new_tasks)} new tasks"
                  f" ({len(pending_tasks)} planned tasks kept, {len(dropped)} dropped)")

            return {
                "plan": final_plan,
                "current_task_index": len(done),
                "plan_revision_count": revision_count,
                "status": "executing",
                "process_log": state.get("process_log", []) + [{
                    "phase": "replanning",
                    "timestamp": datetime.now().isoformat(),
                    "revision_count": revision_count,
                    "reason": replan_data.get("revision_reason", ""),
                    "new_tasks_count": len(new_tasks),
                    "kept_pending_count": len(pending_tasks),
                    "dropped_unknown_agent": [t.get("assigned_agent") for t in dropped]
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

        # Per-cell-type omics runs write to omic_<cell type>/ subfolders (see
        # omic_analysis_tool); list their figures so the report embeds real files.
        session_dir = get_current_session_dir()
        omic_figures = []
        for sub in sorted(d for d in os.listdir(session_dir) if d.startswith("omic_")):
            for root, _, files in sorted(os.walk(os.path.join(session_dir, sub))):
                omic_figures += [os.path.relpath(os.path.join(root, f), session_dir)
                                 for f in sorted(files) if f.endswith(".png")]
        omic_figure_section = ""
        if omic_figures:
            omic_figure_section = (
                "\n## Per-Cell-Type Omics Runs\n"
                "The omics analysis ran once per cell type; each run's outputs are in its own "
                "omic_<cell type>/ folder. Report Step 1 and Step 4 separately for each cell type, "
                "then compare the cell types. Embed figures ONLY from this list (paths are relative "
                "to the report), not the default paths below:\n"
                + "\n".join(f"- {p}" for p in omic_figures) + "\n"
            )

        # Per-cohort facts come from the omics tool's result dict (shared_data), not from the
        # agents' prose, so the report states them as computed
        cohort_section = ""
        omic_runs = state.get("shared_data", {}).get("omic_runs", [])
        if omic_runs:
            lines = []
            for run in omic_runs:
                name = run.get("cell_type") or run.get("disease")
                if run.get("disease_cell_label"):
                    name += f' (original label "{run["disease_cell_label"]}")'
                if run.get("control_cell_type"):
                    name += f" vs normal {run['control_cell_type']}"
                lines.append(f"- {name} ({run.get('disease')}, {run.get('organ')}; "
                             f"{run.get('num_samples')} metacells)")
                if run.get("error"):
                    lines.append(f"  - analysis error: {run['error']}")
            cohort_section = "\n## Omics Cohorts (computed by the omics tool)\n" + "\n".join(lines) + "\n"
            # The Enrichr results themselves, so Step 4 reports computed terms
            enrichment_lines = [f"- {run.get('cell_type') or run.get('disease')}:\n"
                                + format_enrichment_terms(run.get("enrichment_top_terms"))
                                for run in omic_runs]
            cohort_section += (
                "\n## Pathway Enrichment (computed by Enrichr on each cohort's significant up- and "
                "down-regulated genes; adjusted p < 0.05, at most 5 terms per library)\n"
                + "".join(enrichment_lines))

        # Failed and never-run tasks: the report must say this evidence is missing instead of filling
        # the corresponding section from model knowledge
        missing_section = ""
        not_run = [t for t in state["plan"] if t["status"] in ("pending", "in_progress")]
        if failed_tasks or not_run:
            lines = [f"- {t['id']} ({t['assigned_agent']}): {t['description'][:150]} -- FAILED: "
                     f"{str(t.get('error', ''))[:200]}" for t in failed_tasks]
            lines += [f"- {t['id']} ({t['assigned_agent']}): {t['description'][:150]} -- NOT RUN"
                      for t in not_run]
            missing_section = (
                "\n## Failed Tasks (no results)\n"
                "These tasks produced no usable results. In the report, state for each one that this "
                "evidence is missing and why. Do NOT write the corresponding section (e.g. knowledge-graph, "
                "web or literature findings) from general knowledge as if the tool had returned it.\n"
                + "\n".join(lines) + "\n"
            )

        reporting_prompt = f"""Generate a comprehensive research report for a graduate-level biomedical audience.

## DATA PROVENANCE (DO NOT MISCHARACTERIZE)
The omics results come from a COHORT of SINGLE-CELL RNA-seq (scRNA-seq) data
drawn from OmniCellTOSG. Samples in the differential expression analysis are
METACELLS (SEACells aggregates of about 200 cells of one cell type from one
source file), not individual cells, donors or bulk RNA-seq libraries. Describe
the analysis as single-cell at metacell level throughout the report, and give
sample counts as metacells. Inspect the `data_type`, `sample_unit` and
`label_column` fields of the omics tool result to confirm.

## Research Question:
{state['query']}

## Task Results:
{json.dumps(task_results, indent=2, default=str)}
{cohort_section}{missing_section}{omic_figure_section}
---

## REPORT STRUCTURE (Follow this order):

### Step 1: scRNA-seq Cohort Analysis Summary
- State the data type explicitly as single-cell RNA-seq (scRNA-seq) cohort
- Report the label column used (e.g., "disease" → disease vs normal cells;
  "gender" → female vs male cells)
- Report sample sizes for each group (metacells per group)
- Genes listed as `qc_highest_mean_expression_genes` only confirm the cell type;
  do not present them as markers, discriminators or disease-related genes
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
- For each cell type and direction, report the enriched terms listed under
  "Pathway Enrichment (computed by Enrichr)" above, with their adjusted p-values.
  These are the only enrichment results; do not add terms that are not listed.
- Where a direction or library has no term with adjusted p < 0.05, say so.
- Never present the known function of a single gene as an enriched pathway.
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

## DRUG-TARGET NETWORK FIGURE (REQUIRED)
Near the end of the report — after the hypotheses section, before the validation
experiments — insert ONE fenced code block describing a small drug-target
network that visualises the top DEGs and any candidate therapeutics you have
literature evidence for. The block must use the exact fence
``` ```graph-spec``` ``` and contain a JSON object with the schema below.
A renderer will replace the block with an actual PNG figure when the report
is saved.

Schema:

```graph-spec
{{
  "title": "<short title; will be used as alt-text and caption>",
  "subtitle": "<optional one-line context>",
  "filename_hint": "<lowercase_snake_case_id>",
  "nodes": [
    {{"id": "GENE_SYMBOL", "type": "gene", "weight": 0.0-1.0, "role": "hub|regular"}},
    {{"id": "DRUG_NAME",   "type": "drug", "status": "approved|investigational|preclinical|unknown"}}
  ],
  "edges": [
    {{"source": "GENE_A", "target": "GENE_B", "type": "gene-gene", "weight": 0.0-1.0}},
    {{"source": "GENE_A", "target": "DRUG_X", "type": "target-drug", "evidence": "PMID:12345"}}
  ]
}}
```

Guidelines — favour DENSITY over conservatism:

- **Include EVERY gene** that appears in any of the task results above
  (DEGs from omic mining, knowledge-graph hits, pathway-enrichment hits,
  literature-validated targets, scientist-synthesis recommendations).
  Do NOT truncate to a "top-N" — include every gene you can identify
  from the prior task context. Mark the few highest-FDR / highest-impact
  ones as `"role": "hub"`.
- **Include EVERY drug** that was retrieved at any point — KG neighbour
  drugs, drugs mentioned in PubMed papers, drugs from the Google search,
  drugs suggested by the scientist agents. Use the real evidence PMID/DOI
  for `evidence`; if multiple sources support the same target-drug pair,
  pick the most specific PMID/DOI rather than dropping the edge.
- **Include EVERY interaction** you can recall from the prior task
  context, not just a sample. Sources you must mine for edges:
    * Knowledge-graph neighbours from `BioMarkerKGAgent` — every drug,
      pathway, GO term, or disease the KG returned as a first-neighbour
      of a DEG.
    * Pathway co-membership from your KEGG / Reactome / GO enrichment
      results — every gene pair that appears together in an enriched
      term should produce a gene-gene edge (re-use the `Genes` column
      of the enrichment CSVs you can see in the task results).
    * Co-mention in a single retrieved paper as interacting / co-regulated.
    * Well-established protein complexes or signalling cascades that
      were named in the report (OXPHOS, ribosomal subunit, IL-1 family,
      RAS/MAPK cascade, …).
- An EDGE you can justify from prior context is ALWAYS better than no
  edge — when in doubt, include it and set `evidence` to the source
  ("KEGG:hsa00190", "PrimeKG", "PMID:nnn"). The renderer can handle
  dense graphs; sparseness is the bigger readability problem.
- Do NOT invent novel interactions out of thin air. The rule is
  "include if I retrieved evidence", not "include any gene-pair that
  sounds biologically reasonable".
- Use the exact gene symbols from the DE results so the figure agrees
  with the tables above it. **Drop any orphan node** (no edges at all)
  before emitting the spec — orphan nodes only clutter the figure.
- It is fine — and encouraged — to emit 30, 50, or 100 nodes if that
  reflects what you actually pulled from prior task results. Do NOT
  cap the size of the network. The figure renderer is sized adaptively.
- Do NOT include this block if you have fewer than 3 drug-evidence pairs;
  omit silently rather than fabricate.
"""

        response, cut_off = await stream_long_reply(self.lead_llm, [
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
        ], "report")

        # Extract text from response using helper function
        report = extract_text_from_llm_response(response.content)
        if cut_off:
            report += REPORT_CUT_OFF_NOTE

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
                "report_cut_off": cut_off,
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

    @staticmethod
    def _appendix_table_cell(value: Any, max_chars: int = None) -> str:
        """Normalize appendix table text so pipe tables keep their shape."""
        text = " ".join(str(value if value is not None else "N/A").split())
        text = text.replace("|", "/")
        if max_chars and len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        return text

    @staticmethod
    def _display_agent_name(agent_name: str) -> str:
        """Add break opportunities for long agent identifiers in PDF tables."""
        display_names = {
            "OmicMiningAgent": "Omic-Mining Agent",
            "BioMarkerKGAgent": "BioMarker-KG Agent",
            "PubMedResearcher": "PubMed Researcher",
            "GoogleSearcher": "Google Searcher",
            "ScientistsAgent": "Scientists Agent",
        }
        return display_names.get(agent_name, agent_name)

    def _format_execution_plan_table(self, plan: List[Dict[str, Any]]) -> str:
        """Render A2 using the same pipe-table style as the main report tables."""
        table = [
            "| Step | Task ID | Agent | Description | Status |\n",
            "| :--- | :--- | :--- | :--- | :--- |\n",
        ]
        for i, task in enumerate(plan, 1):
            table.append(
                "| {step} | {task_id} | {agent} | {description} | {status} |\n".format(
                    step=i,
                    task_id=self._appendix_table_cell(task.get("id", f"task_{i}")),
                    agent=self._appendix_table_cell(
                        self._display_agent_name(task.get("assigned_agent", "Unknown"))
                    ),
                    description=self._appendix_table_cell(
                        task.get("description", "N/A"),
                        max_chars=220,
                    ),
                    status=self._appendix_table_cell(task.get("status", "pending")),
                )
            )
        table.append("\n")
        return "".join(table)
    
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
            appendix_parts.append(self._format_execution_plan_table(plan))
        
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
            
            # Top genes, one block per omics run (from the tool's result dict)
            for run in shared_data.get("omic_runs", []):
                name = run.get("cell_type") or run.get("disease")
                appendix_parts.append(f"### Top Differentially Expressed Genes: {name}\n\n")

                if not run.get("top_genes"):
                    appendix_parts.append("No differential expression results.\n\n")
                    continue
                appendix_parts.append("| # | Gene | Log2FC | FDR | Direction |\n")
                appendix_parts.append("|---|------|--------|-----|----------|\n")
                for i, gene in enumerate(run["top_genes"], 1):
                    log2fc = gene.get("log2_fold_change")
                    fdr = gene.get("FDR")
                    direction = "N/A" if log2fc is None else ("up" if log2fc > 0 else "down")
                    log2fc = f"{log2fc:.3f}" if isinstance(log2fc, float) else "N/A"
                    fdr = f"{fdr:.2e}" if isinstance(fdr, float) else "N/A"
                    appendix_parts.append(f"| {i} | {gene.get('gene_name')} | {log2fc} | {fdr} | {direction} |\n")
                appendix_parts.append("\n")
                appendix_parts.append(f"### Enriched Terms (Enrichr, adjusted p < 0.05): {name}\n\n")
                appendix_parts.append(format_enrichment_terms(run.get("enrichment_top_terms")) + "\n")

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
        tmp_md_path = None
        try:
            import subprocess
            import shutil
            import tempfile
            import textwrap
            import re

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

            def cleanup(*paths):
                for path in paths:
                    if not path:
                        continue
                    try:
                        os.remove(path)
                    except OSError:
                        pass

            def preprocess_markdown(md_path: str) -> str:
                """Use the report-bundle preprocessing for agent-generated PDFs."""
                agent_display_names = {
                    "OmicMiningAgent": "Omic-Mining Agent",
                    "BioMarkerKGAgent": "BioMarker-KG Agent",
                    "PubMedResearcher": "PubMed Researcher",
                    "GoogleSearcher": "Google Searcher",
                    "ScientistsAgent": "Scientists Agent",
                }

                with open(md_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                out = []
                in_code_block = False

                for i, line in enumerate(lines):
                    stripped = line.rstrip()

                    if stripped.startswith("```"):
                        in_code_block = not in_code_block
                        out.append(line)
                        continue

                    if in_code_block:
                        if len(stripped) > 100:
                            broken = stripped
                            broken = re.sub(r"('\s*,\s*')", r"',\n'", broken)
                            broken = re.sub(r'("\s*,\s*")', r'",\n"', broken)
                            broken = re.sub(r"(},\s*{)", r"},\n{", broken)
                            broken = re.sub(r"(\],\s*\[)", r"],\n[", broken)
                            broken = re.sub(r"(:\s*True,)", r": True,\n", broken)
                            broken = re.sub(r"(:\s*False,)", r": False,\n", broken)
                            broken = re.sub(r"(\.\s+)([A-Z])", r".\n\2", broken)
                            if any(len(segment) > 90 for segment in broken.split("\n")):
                                new_broken = []
                                for segment in broken.split("\n"):
                                    if len(segment) > 90:
                                        words = segment.split(" ")
                                        current_line = []
                                        current_len = 0
                                        for word in words:
                                            if current_len + len(word) + 1 > 80 and current_line:
                                                new_broken.append(" ".join(current_line))
                                                current_line = [word]
                                                current_len = len(word)
                                            else:
                                                current_line.append(word)
                                                current_len += len(word) + 1
                                        if current_line:
                                            new_broken.append(" ".join(current_line))
                                    else:
                                        new_broken.append(segment)
                                broken = "\n".join(new_broken)
                            out.append(broken + "\n")
                        else:
                            out.append(line)
                        continue

                    is_table_row = stripped.startswith("|") and stripped.endswith("|")
                    is_image = stripped.startswith("![") and "](" in stripped

                    if is_image:
                        if out and out[-1].strip() != "":
                            out.append("\n")
                        out.append(line)
                        if i + 1 < len(lines) and lines[i + 1].strip() != "":
                            out.append("\n")
                        continue

                    if is_table_row:
                        for raw, display in agent_display_names.items():
                            stripped = stripped.replace(raw, display)
                        padded_line = re.sub(r"\|([^|])", r"|  \1", stripped)
                        padded_line = re.sub(r"([^|])\|", r"\1  |", padded_line)
                        line = padded_line + "\n"
                        stripped = padded_line

                        if out and out[-1].strip() != "" and not (
                            out[-1].rstrip().startswith("|") and out[-1].rstrip().endswith("|")
                        ):
                            out.append("\n")
                    else:
                        if (
                            out
                            and out[-1].rstrip().startswith("|")
                            and out[-1].rstrip().endswith("|")
                            and stripped != ""
                        ):
                            out.append("\n")

                    out.append(line)

                return "".join(out)

            tmp_md_fd, tmp_md_path = tempfile.mkstemp(suffix='.md', prefix='oca_pp_', dir=session_dir)
            with os.fdopen(tmp_md_fd, 'w', encoding='utf-8') as f:
                f.write(preprocess_markdown(markdown_path))

            # ── Strategy 1: pandoc + xelatex with custom LaTeX header ──
            if has_xelatex:
                latex_header = r"""
% Fonts
\usepackage{fontspec}
\setmainfont{DejaVu Serif}
\setsansfont{DejaVu Sans}
\setmonofont[Scale=0.85]{DejaVu Sans Mono}

% Page layout
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\textit{OmniCellAgent Analysis Report}}
\fancyhead[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\fancyfoot{}

% Tables
\usepackage{makecell}
\usepackage{etoolbox}
\usepackage{array}
\usepackage{tabularx}
\renewcommand{\arraystretch}{1.5}
\AtBeginEnvironment{longtable}{\footnotesize\setlength{\tabcolsep}{10pt}}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}
\setlength{\arrayrulewidth}{0.4pt}
\AtBeginEnvironment{longtable}{%
  \setlength{\tabcolsep}{8pt}%
  \renewcommand{\arraystretch}{1.4}%
}
\newcommand{\thmark}{\rule[-0.5ex]{0.4pt}{2.5ex}}

% Images
\makeatletter
\def\maxwidth{\ifdim\Gin@nat@width>0.85\linewidth 0.85\linewidth\else\Gin@nat@width\fi}
\makeatother
\setkeys{Gin}{width=\maxwidth,keepaspectratio}
\usepackage{float}
\let\origfigure\figure
\let\endorigfigure\endfigure
\renewenvironment{figure}[1][htbp]{%
  \origfigure[H]%
  \centering
}{%
  \endorigfigure
}
% pandoc >= 3 wraps each image in \pandocbounded, which measures it inside a box where \par and
% center are not allowed (LaTeX Error: Not allowed in LR mode), so inside a box the image is
% set plainly and the spacing and centering go around \pandocbounded instead
\let\oldincludegraphics\includegraphics
\renewcommand{\includegraphics}[2][]{%
  \ifinner
    \oldincludegraphics[#1]{#2}%
  \else
    \par\vspace{12pt}%
    \begin{center}%
      \oldincludegraphics[#1]{#2}%
    \end{center}%
    \vspace{12pt}\par%
  \fi
}
\AtBeginDocument{%
  \ifdefined\pandocbounded
    \let\oldpandocbounded\pandocbounded
    \renewcommand*{\pandocbounded}[1]{%
      \par\vspace{12pt}%
      \begin{center}%
        \oldpandocbounded{#1}%
      \end{center}%
      \vspace{12pt}\par%
    }%
  \fi
}

% Code blocks
\usepackage{fvextra}
\fvset{breaklines,breakanywhere,fontsize=\scriptsize,breaksymbol=,breakanywheresymbolpre=,breakbeforesymbolpre=,breakaftersymbolpre=}
\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,breakanywhere,commandchars=\\\{\},fontsize=\scriptsize}

% Section spacing
\usepackage{titlesec}
\titleformat{\section}{\Large\bfseries}{}{0em}{}[\vspace{4pt}\hrule\vspace{6pt}]
\titleformat{\subsection}{\large\bfseries}{}{0em}{}
\titleformat{\subsubsection}{\normalsize\bfseries}{}{0em}{}
\let\oldsection\section
\renewcommand{\section}{\clearpage\oldsection}
\PassOptionsToPackage{colorlinks=true,linkcolor=blue!60!black,urlcolor=blue!70!black}{hyperref}
\usepackage{microtype}
\usepackage{seqsplit}
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
                    '-V', 'papersize=letter',
                    '--toc', '--toc-depth=2',
                    '--highlight-style=tango',
                    '--resource-path', session_dir,
                    '--columns=72',
                ]
                result = subprocess.run(cmd, capture_output=True, text=True,
                                        cwd=session_dir, timeout=300)
                cleanup(header_path)

                if result.returncode == 0:
                    print(f"📄 PDF generated (xelatex): {pdf_path}")
                    cleanup(tmp_md_path)
                    return pdf_path
                print(f"⚠️ xelatex PDF failed, trying HTML method...")
                if result.stderr:
                    for line in result.stderr.strip().split("\n")[-8:]:
                        print(f"   {line}")

            # ── Strategy 2: Pandoc → HTML → wkhtmltopdf ─────────────────
            if has_wkhtmltopdf:
                css_content = textwrap.dedent("""\
                body {
                    font-family: "DejaVu Sans", "Noto Sans", "Segoe UI", Roboto, Arial, sans-serif;
                    font-size: 11pt;
                    line-height: 1.55;
                    color: #1a1a1a;
                    max-width: 100%;
                }
                h1 { font-size: 1.6em; border-bottom: 2px solid #2c3e50; padding-bottom: 6px; margin-top: 30px; }
                h2 { font-size: 1.3em; border-bottom: 1px solid #bdc3c7; padding-bottom: 4px; margin-top: 24px; }
                h3 { font-size: 1.1em; margin-top: 18px; }
                h1, h2, h3 { page-break-after: avoid; }
                p > img, figure, .figure {
                    display: block !important;
                    margin-left: auto !important;
                    margin-right: auto !important;
                    text-align: center !important;
                    max-width: 85% !important;
                    max-height: 550px !important;
                    height: auto !important;
                    width: auto !important;
                    page-break-inside: avoid;
                }
                figure { text-align: center; }
                figcaption { text-align: center; font-style: italic; margin-top: 8px; }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 14px 0;
                    font-size: 9pt;
                    page-break-inside: avoid;
                    table-layout: auto;
                }
                th, td {
                    border: 1px solid #ccc;
                    padding: 6px 10px;
                    text-align: left;
                    vertical-align: top;
                    word-wrap: break-word;
                    overflow-wrap: break-word;
                    max-width: 200px;
                }
                th { background-color: #ecf0f1; font-weight: 600; white-space: nowrap; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                pre, code {
                    font-family: "DejaVu Sans Mono", "Consolas", monospace;
                    font-size: 7.5pt;
                    background-color: #f5f5f5;
                    border-radius: 3px;
                }
                pre {
                    padding: 10px 12px;
                    overflow-x: hidden;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    word-break: break-all;
                    border: 1px solid #e0e0e0;
                    max-width: 100%;
                    line-height: 1.4;
                }
                code {
                    word-break: break-all;
                }
                blockquote { border-left: 3px solid #3498db; padding-left: 12px; color: #555; }
                hr { border: none; border-top: 1px solid #bbb; margin: 20px 0; }
                """)
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
                        '--page-size', 'Letter',
                        '--margin-top', '18mm', '--margin-bottom', '18mm',
                        '--margin-left', '14mm', '--margin-right', '14mm',
                        '--footer-center', '[page]', '--footer-font-size', '9',
                        html_path, pdf_path,
                    ]
                    result2 = subprocess.run(pdf_cmd, capture_output=True, text=True,
                                             cwd=session_dir, timeout=180)
                    if result2.returncode == 0:
                        print(f"📄 PDF generated (wkhtmltopdf): {pdf_path}")
                        cleanup(html_path, css_path, tmp_md_path)
                        return pdf_path
                    else:
                        print(f"⚠️ wkhtmltopdf failed: {result2.stderr[:200] if result2.stderr else ''}")
                else:
                    print(f"⚠️ Pandoc HTML conversion failed: {result.stderr[:200] if result.stderr else ''}")

                cleanup(css_path, html_path)

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
                cleanup(tmp_md_path)
                return html_path

            print(f"⚠️ Could not generate PDF or HTML: {result.stderr[:300]}")
            cleanup(tmp_md_path)
            return None

        except subprocess.TimeoutExpired:
            print("⚠️ PDF generation timed out")
            try:
                os.remove(tmp_md_path)  # type: ignore[arg-type]
            except (OSError, TypeError):
                pass
            return None
        except Exception as e:
            print(f"⚠️ Error generating PDF: {e}")
            try:
                os.remove(tmp_md_path)  # type: ignore[arg-type]
            except (OSError, TypeError):
                pass
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

            # Render any ```graph-spec``` blocks the reporter emitted into PNGs
            # under <session>/network_plots/ and rewrite the markdown to embed
            # those PNGs via standard ![](…) syntax so pandoc picks them up.
            try:
                from agent.network_renderer import render_specs_in_markdown
                full_report, generated = render_specs_in_markdown(
                    full_report, session_dir, relative_image_dir="network_plots"
                )
                if generated:
                    print(f"🖼  Rendered {len(generated)} drug-target network figure(s):")
                    for p in generated:
                        print(f"   - {p}")
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️  network_renderer post-process failed: {exc}")

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
    
    async def revise_report_from_feedback(self, original_report: str, feedback: str, query: str = "") -> dict:
        """
        Revise an existing report based on user feedback WITHOUT re-running the
        full analysis pipeline.  The revised report is saved as
        ``report_revised_<timestamp>.{md,pdf}`` so it never overwrites the
        original.

        Args:
            original_report: The full text of the original report (markdown).
            feedback: User-provided feedback / revision instructions.
            query: The original research question (for context in the header).

        Returns:
            dict with keys ``revised_report`` (str) and ``report_path`` (str|None).
        """
        print("\n📝 REPORT REVISION PHASE (feedback-driven)")
        print("=" * 60)

        revision_prompt = f"""You are given an existing biomedical research report and user feedback.
Your job is to **edit the report according to the feedback** while preserving the overall structure, data, citations, and scientific accuracy.

## RULES
1. Only modify sections that the feedback explicitly asks to change.
2. Do NOT remove data, tables, or citations unless the feedback asks for it.
3. Do NOT fabricate new data or citations that were not in the original report.
4. Keep the same section numbering / heading hierarchy.
5. If the feedback asks for additions that require new analysis, note them as
   "⚠️ This addition would require re-running the analysis pipeline." and leave
   a placeholder — do NOT hallucinate results.
6. Return the COMPLETE revised report in Markdown (not just the diff).
7. **PRESERVE the drug-target network figure.** If the original report
   contains either an embedded image ``![…](network_plots/…)`` or a
   ```graph-spec``` fenced JSON block, keep an equivalent block in the
   revised report so the renderer can produce a figure for the revised
   PDF too. If the original embedded an image and you want the figure
   regenerated (e.g. to add genes the reviewer suggested), emit a fresh
   ```graph-spec``` block matching the schema described at the end of
   the original report — keep the same `filename_hint` value so the
   image filename is stable. Do NOT silently drop the network figure.

---

## Original Report
{original_report}

---

## User Feedback / Revision Instructions
{feedback}

---

Please return the **full revised report** in Markdown.
"""

        with usage_label("revision"):
            response, cut_off = await stream_long_reply(self.lead_llm, [
                SystemMessage(content=(
                    "You are a meticulous scientific editor. You revise biomedical "
                    "research reports based on reviewer feedback. Preserve all existing "
                    "data, citations, and structure unless the feedback explicitly asks "
                    "for changes. Return the complete revised markdown report."
                )),
                HumanMessage(content=revision_prompt),
            ], "revised report")
        self._save_usage_log(finish=False)

        revised_report = extract_text_from_llm_response(response.content)
        if cut_off:
            revised_report += REPORT_CUT_OFF_NOTE

        # ── Save the revised report (never overwrite original) ──────────
        report_path = self._save_revised_report(query, revised_report, feedback)

        print(f"📄 Revised report generated ({len(revised_report)} characters)")
        if report_path:
            print(f"💾 Saved to: {report_path}")

        return {
            "revised_report": revised_report,
            "report_path": report_path,
        }

    def _save_revised_report(self, query: str, report: str, feedback: str = "") -> Optional[str]:
        """Save the revised report with a ``_revised`` suffix so the original is untouched."""
        try:
            sessions_base = get_path('sessions.base', absolute=True, create=True)
            session_dir = os.path.join(sessions_base, self.session_id)
            os.makedirs(session_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"report_revised_{timestamp}.md"
            report_path = os.path.join(session_dir, report_filename)

            header = f"""---
title: OmniCellAgent Analysis Report — REVISED
session_id: {self.session_id}
generated_at: {datetime.now().isoformat()}
query: {query}
revision_feedback: |
  {feedback[:500]}
---

"""
            full_report = header + report

            # Same network-renderer post-process as the first-run save path:
            # if the revised report still contains a ```graph-spec``` block,
            # render it into network_plots/ and rewrite the markdown.
            try:
                from agent.network_renderer import render_specs_in_markdown
                full_report, generated = render_specs_in_markdown(
                    full_report, session_dir, relative_image_dir="network_plots"
                )
                if generated:
                    print(f"🖼  Rendered {len(generated)} drug-target network figure(s) (revised):")
                    for p in generated:
                        print(f"   - {p}")
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️  network_renderer post-process failed (revised): {exc}")

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(full_report)

            print(f"📝 Revised markdown report saved: {report_path}")

            # Compile to PDF (reuses existing _compile_pdf which derives the
            # .pdf path from the .md path, so it will be report_revised_*.pdf)
            pdf_path = self._compile_pdf(report_path)
            if pdf_path:
                print(f"✅ Revised report available as both MD and PDF")

            return report_path

        except Exception as e:
            print(f"Error saving revised report: {e}")
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
    
    def _save_usage_log(self, finish: bool = True):
        """Write the session's token/time usage to <session_dir>/usage_log.json.

        finish=False keeps total_seconds of the run (used when a revision is appended afterwards)."""
        try:
            if finish or self.usage_tracker.total_seconds is None:
                self.usage_tracker.finish()
            filepath = os.path.join(self.session_dir, "usage_log.json")
            self.usage_tracker.save(filepath)
            s = self.usage_tracker.summary()
            print(f"📊 Usage: {s['total_seconds']}s, {s['total_tokens_all_llm_calls']:,} tokens "
                  f"(agents {s['agent_llm_tokens']['total_tokens']:,}, "
                  f"PubMed summaries {s['pubmed_full_text']['summarizer_tokens']['total_token_count']:,}) "
                  f"-> {filepath}")
        except Exception as e:
            print(f"Error saving usage log: {e}")

    async def run(self, query: str) -> str:
        """Run the agent system on a query"""
        print(f"\n🔬 Processing query: {query}")
        print("=" * 80)

        # Fresh usage record per run
        self.usage_tracker = UsageTracker(self.session_id, model=self.model_name, lead_model=self.lead_model_name)
        self.usage_tracker.query = query
        self._usage_callback.tracker = self.usage_tracker
        set_current_tracker(self.usage_tracker)

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
                "omic_runs": [],      # One record per omic_analysis_tool run (OmicMiningAgent)
                "top_genes": [],      # Top genes across omic_runs, annotated with cell type
                "paper_dois": [],      # Populated by PubMed tools
                "pathways": [],        # Populated by KEGG analysis
                "disease": "",         # Extracted from query
                "cell_type": "",       # Extracted from query
            },
            "process_log": [],
            "final_report": None,
            "status": "planning"
        }
        
        # Run the graph (usage log is saved even if the run fails)
        try:
            # Each task attempt is one graph step; failed tasks are retried and re-planned (up to
            # max_plan_revisions), which can exceed LangGraph's default limit of 25 steps
            final_state = await self.graph.ainvoke(initial_state, config={"recursion_limit": 100})
        finally:
            self._save_usage_log()

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
        default=DEFAULT_MODEL,
        help=f"Model of the sub-agents (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--lead-model",
        type=str,
        default=LEAD_MODEL,
        help=f"Model of the planner, replanner and reporter, at {LEAD_REASONING_EFFORT} reasoning effort "
             f"(default: {LEAD_MODEL})"
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
        lead_model_name=args.lead_model,
        session_id=args.session_id
    )
    
    report = await agent.run(args.query)
    
    print("\n📄 FINAL REPORT:")
    print("=" * 80)
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
