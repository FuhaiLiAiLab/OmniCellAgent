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
# STATE DEFINITIONS
# ==============================================================================

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
    
    Args:
        query: Search query for PubMed
        
    Returns:
        Search results with paper summaries
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
            use_llm_processing=True,
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
                use_llm_processing=True,
                max_concurrent=10,
                session_id=session_id
            )
            if isinstance(papers2, list):
                all_papers.extend(papers2)
        
        # Deduplicate papers by title
        seen_titles = set()
        unique_papers = []
        for paper in all_papers:
            title = paper.get('title', '').lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_papers.append(paper)
        
        if not unique_papers:
            return f"No papers found for queries: {queries_run}"
        
        # Format the results
        result_lines = [
            f"Found {len(unique_papers)} unique medical research papers",
            f"Queries run: {queries_run}",
            "=" * 60
        ]
        
        for i, paper in enumerate(unique_papers, 1):
            paper_info = [
                f"\nPaper {i}:",
                f"  Title: {paper.get('title', 'N/A')}",
                f"  Has LLM Processing: {'Yes' if paper.get('llm_content') else 'No'}"
            ]
            
            # Add LLM-processed content summary if available
            if paper.get('llm_content'):
                llm_summary = paper['llm_content']
                paper_info.append(f"  Content: {llm_summary}")
            else:
                paper_info.append(f"  Content: Title only (no content processing)")
            
            result_lines.extend(paper_info)
        
        return "\n".join(result_lines)
        
    except Exception as e:
        return f"Error searching PubMed: {str(e)}"


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

def create_llm(model_name: str = "gemini-2.0-flash"):
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
    
    def __init__(self, name: str, description: str, system_message: str, tools: List, llm=None, model_name: str = "gemini-2.0-flash"):
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
                "result": response.content,
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
    
    def __init__(self, model_name: str = "gemini-2.0-flash", log_dir: str = None, session_id: str = None, llm=None):
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
                description="Searches biomedical literature using PubMed",
                system_message=PUBMED_AGENT_SYSTEM_MESSAGE_v1,
                tools=[pubmed_search_tool],
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
        """Create initial plan for the query"""
        print("\n📋 PLANNING PHASE")
        print("=" * 60)
        
        planning_prompt = f"""You are a biomedical research planner. Analyze the user's query and create a detailed plan.

User Query: {state['query']}

Available Agents:
1. OmicMiningAgent - For gene expression, biomarkers, differential expression, pathway analysis
2. PubMedResearcher - For searching biomedical literature
3. GoogleSearcher - For current web information and recent research
4. ScientistRAGExpert - For querying scientific knowledge bases
5. BioMarkerKGAgent - For querying biomedical knowledge graphs

Create a plan with sequential tasks. Each task should specify:
- What needs to be done
- Which agent should handle it
- Dependencies on previous tasks

For molecular/genetic queries, ALWAYS include OmicMiningAgent first, then literature validation.

Respond in JSON format:
{{
    "analysis": "Brief analysis of the query",
    "tasks": [
        {{
            "id": "task_1",
            "description": "Task description",
            "assigned_agent": "AgentName",
            "depends_on": []
        }}
    ]
}}
"""
        
        response = await self.llm.ainvoke([
            SystemMessage(content="You are a strategic research planner. Output valid JSON only."),
            HumanMessage(content=planning_prompt)
        ])
        
        # Parse the plan
        try:
            # Extract JSON from response
            content = response.content
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
            
            print(f"📝 Created plan with {len(tasks)} tasks:")
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
            
        except json.JSONDecodeError as e:
            print(f"⚠️ Error parsing plan: {e}")
            # Create a default plan
            default_task = SubTask(
                id="task_1",
                description=state['query'],
                assigned_agent="GoogleSearcher",
                status="pending",
                result=None,
                error=None,
                attempts=0
            )
            return {
                "plan": [default_task],
                "current_task_index": 0,
                "status": "executing",
                "process_log": state.get("process_log", []) + [{"phase": "planning", "error": str(e)}],
                "messages": [AIMessage(content="Created default plan due to parsing error")]
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
        
        # Prepare context from previous task results
        context = {
            "query": state["query"],
            "session_id": state["session_id"],
            "previous_results": state.get("agent_outputs", {})
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
            
            return {
                "plan": plan,
                "current_task_index": current_idx + 1,
                "agent_outputs": agent_outputs,
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
            content = response.content
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
        """Generate comprehensive final report"""
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
        
        reporting_prompt = f"""Generate a comprehensive research report based on the following analysis.

Original Query: {state['query']}

## Process Summary:
- Total tasks planned: {len(state['plan'])}
- Tasks completed: {len(completed_tasks)}
- Tasks failed: {len(failed_tasks)}
- Plan revisions: {state.get('plan_revision_count', 0)}

## Task Results:
{json.dumps(task_results, indent=2, default=str)}

## Process Log:
{json.dumps(process_log, indent=2, default=str)}

Generate a COMPREHENSIVE report that:
1. **Executive Summary**: Key findings and conclusions
2. **Methodology**: How the analysis was conducted (which agents, what tools)
3. **Detailed Findings**: For each task, explain what was discovered
4. **Data and Evidence**: Include specific numbers, gene names, pathways, statistics
5. **Integration**: How different findings connect and support each other
6. **Limitations**: What couldn't be accomplished and why
7. **Recommendations**: Next steps and further research directions

The report should be at least 1500 words and include all important details from the process.
Format the report in Markdown.
"""
        
        response = await self.llm.ainvoke([
            SystemMessage(content="""You are an expert scientific report writer. Generate detailed, comprehensive reports 
that capture all important information from the research process. Include specific data, names, and findings.
Always use Markdown formatting."""),
            HumanMessage(content=reporting_prompt)
        ])
        
        report = response.content
        
        # Save the report
        report_path = self._save_report(state["query"], report)
        
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
    
    def _save_report(self, query: str, report: str) -> Optional[str]:
        """Save the report to a file"""
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
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(header + report)
            
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
        default="gemini-2.0-flash",
        help="Model to use (default: gemini-2.0-flash)"
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
