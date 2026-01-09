#!/usr/bin/env python3
"""
Concise Magnetic Agent system for biomedical research.
"""

import asyncio
import argparse
import json
import os
import logging
import uuid
from functools import partial
from datetime import datetime
from typing import Optional, Dict, Any, Literal
from dotenv import load_dotenv
load_dotenv(".env")

import sys
# Add project root to Python path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import path config
from utils.path_config import get_path

# Suppress INFO logs from autogen
logging.getLogger("autogen_core").setLevel(logging.WARNING)
logging.getLogger("autogen_agentchat").setLevel(logging.WARNING)
logging.getLogger("autogen_ext").setLevel(logging.WARNING)
# Suppress HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
# Suppress paperscraper warnings
logging.getLogger("paperscraper").setLevel(logging.ERROR)

# Import core AutoGen components
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_core.tools import FunctionTool

# Import tools
from tools.pubmed_tools.query_pubmed_tool import medical_research_tool
from tools.google_search_tools.google_search_w3m import web_search_tool
from tools.scientist_rag_tools.scientist_tool import scientist_rag_tool_wrapper
from tools.gretriever_tools.gretriever_client import gretriever_tool

# Add omic_tools to path for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools', 'omic_tools'))
from omic_fetch_analysis_workflow import omic_fetch_analysis_workflow as _omic_workflow


# ==============================================================================
# OMIC ANALYSIS TOOL WRAPPER
# ==============================================================================

def create_omic_analysis_tool(session_id: str) -> FunctionTool:
    """
    Create an omic analysis FunctionTool with a partialized session directory.
    
    Args:
        session_id: Unique session identifier for this agent instance
        
    Returns:
        FunctionTool: A tool that can be used by AutoGen agents
    """
    # Create the session directory path
    sessions_base = get_path('sessions.base', absolute=True, create=True)
    session_dir = os.path.join(sessions_base, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
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
            
        Usage:
            - For lung cancer: disease="lung adenocarcinoma", organ="lung"
            - For Alzheimer's: disease="Alzheimer disease", organ="brain"  
            - For specific cells in disease: disease="...", cell_type="...", organ="..."
        """
        try:
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
            
            # Call the actual workflow
            result = _omic_workflow(**params)
            
            return result
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "message": f"Error in omic analysis: {str(e)}",
                "traceback": traceback.format_exc()
            }
    
    # Create and return the FunctionTool with explicit name
    return FunctionTool(
        omic_analysis_tool,
        name="omic_analysis",
        description="""ALWAYS USE THIS TOOL when asked about genes, biomarkers, or molecular mechanisms of diseases.

This tool retrieves real omics data and performs differential expression analysis with pathway enrichment.

REQUIRED PARAMETERS:
- disease: Disease name (e.g., "lung adenocarcinoma", "Alzheimer disease", "breast cancer")
- organ: Organ to filter data (e.g., "lung", "brain", "breast") - HIGHLY RECOMMENDED for efficiency

OPTIONAL:
- cell_type: Specific cell type (e.g., "microglial cell", "T cell")

EXAMPLES:
1. omic_analysis(disease="Alzheimer disease", organ="brain")
2. omic_analysis(disease="lung adenocarcinoma", organ="lung")
3. omic_analysis(disease="breast cancer", organ="breast", cell_type="epithelial cell")

RETURNS: Top differentially expressed genes, pathway enrichment results, and visualization plots.
"""
    )


from utils.prompt import MAGNETIC_ONE_ORCHESTRATOR_PROMPT, SEARCH_AGENT_SYSTEM_MESSAGE_v1, PUBMED_AGENT_SYSTEM_MESSAGE_v1

class SimpleMagneticAgentSystem:
    """
    Concise Magnetic Agent System for biomedical research.
    """
    
    def __init__(self, model_name="gemini-2.5-pro", temperature=0.0, log_dir=None, session_id=None):
        """
        Initialize the system with model configuration.
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Model temperature for generation
            log_dir: Directory to save conversation logs
            session_id: Unique session identifier. If None, a UUID will be generated.
                       This is used to create a dedicated directory for all analysis outputs.
        """
        # Use default log directory from config if not provided
        if log_dir is None:
            log_dir = get_path('logs.base', absolute=True, create=True)
        
        # Generate or use provided session ID
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.session_id = session_id
        
        print(f"🔧 Initializing with model: {model_name}")
        print(f"📁 Session ID: {self.session_id}")
        
        self.model_client = OpenAIChatCompletionClient(
            model=model_name,
            temperature=temperature,
            model_info=ModelInfo(vision=True, 
                                function_calling=True, 
                                json_output=True, 
                                family="GEMINI_2_5_PRO", 
                                structured_output=False),
        )

        self.model_client_reflection_compatible = OpenAIChatCompletionClient(
            model="o4-mini",
        )
        
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.team = None
        
    async def initialize_team(self):
        """Initialize the MagenticOneGroupChat team with agents."""
        print("🚀 Initializing agent team...")
        
        agents = []
        
        pubmed_agent = AssistantAgent(
            name='PubMedResearcher',
            description="""Specialized agent for searching biomedical literature using PubMed. 
            Provides peer-reviewed scientific articles and research papers on medical topics.""",
            model_client=self.model_client,
            tools=[medical_research_tool],
            system_message=PUBMED_AGENT_SYSTEM_MESSAGE_v1,
            # """ You are a biomedical research assistant. You must use your tool to search pubmed for relevant articles.
            # The tool will return a list of summaries of the papers found.
            # After receiving the search result summaries, your main task is to write a cohesive report.
            # Integrate Information: Combine facts and data from all relevant summaries into a single, well-structured answer. Do not simply copy or list the summaries.
            # Highlight Key Findings: Emphasize points of consensus across sources, unique or important details from a single source, and any notable discrepancies or conflicting perspectives.
            # Cite Sources: Reference your information using the source title or URL.""",

            reflect_on_tool_use=True,
            max_tool_iterations=5,
        )
        # agents.append(pubmed_agent)
        
        search_agent = AssistantAgent(
            name="GoogleSearcher",
            description="""General search agent for finding current information and broad context 
            on topics using Google search.""",
            model_client=self.model_client,
            tools=[web_search_tool],
            system_message=SEARCH_AGENT_SYSTEM_MESSAGE_v1,
            reflect_on_tool_use=True,
            #max_tool_iterations=3,
        )
        agents.append(search_agent)
        
        scientist_rag_agent = AssistantAgent(
            name="ScientistRAGExpert",
            description="""Specialist agent that uses RAG retrieval to access scientific knowledge base 
            for expert scientific information and analysis.""",
            model_client=self.model_client,
            tools=[scientist_rag_tool_wrapper],
            system_message="""You are a scientific knowledge expert with access to a comprehensive 
            scientific knowledge base. Your role is to:
            1. Query your tool (scientific knowledge base) using a concise phrase (3-5 words).
            2. Provide expert analysis and insights from scientific literature
            3. Offer detailed explanations of scientific concepts and mechanisms
            4. Synthesize information from multiple scientific sources
            
            Available scientists: Scientist 1 (Alzheimer's disease research, amyloid-beta, tau proteins, neurodegeneration)
            When querying, use anonymized scientist names (e.g., "Scientist 1", "Scientist 2") to protect privacy.
            Use precise scientific terminology and provide comprehensive analysis.""",
            reflect_on_tool_use=False,
            #max_tool_iterations=5,
        )
        agents.append(scientist_rag_agent)

        primekg_agent = AssistantAgent(
            name="BioMarkerKGAgent",
            model_client=self.model_client_reflection_compatible,
            description="""Start with using this agent if the query is not related to omic data and more general biomedical concepts. 
            This agent queries the PrimeKG knowledge graph to extract highly relevant and reliable data on biological processes and their relations. Its purpose is to build a broad, comprehensive understanding of biomedical concepts. This functionality is distinct from the OmicMiningAgent, which is optimized for granular queries of the Omic-specific knowledge graph.        """,
            tools=[gretriever_tool],
            system_message="""You are a PrimeKG agent that queries the PrimeKG knowledge graph to retrieve relevant biomedical information. \
            You will provide structured data and relationships from the knowledge graph to enhance understanding of biomedical concepts.
            """,
            reflect_on_tool_use=True,
            max_tool_iterations=5,
        )
        agents.append(primekg_agent)

        # Create session-aware omic analysis tool
        omic_tool = create_omic_analysis_tool(self.session_id)
        
        Omicxagent = AssistantAgent(
            name="OmicMiningAgent",
            model_client=self.model_client,
            tools=[omic_tool],
            description="""USE THIS AGENT for any query about genes, biomarkers, molecular mechanisms, or disease pathways.

TRIGGER KEYWORDS: genes, biomarkers, differentially expressed, gene expression, pathway, KEGG, molecular mechanism, 
drug targets, therapeutic targets, transcriptomics, genomics, proteomics, signaling, microglia, neurons, cell types in disease.

EXAMPLES OF QUERIES FOR THIS AGENT:
- "What genes are dysregulated in Alzheimer's disease?"
- "Find biomarkers for lung cancer"
- "What are the key signaling pathways in breast cancer?"
- "Identify drug targets for pancreatic cancer"
- "What genes are expressed in microglia in AD?"

This agent retrieves REAL omics data and performs differential expression + pathway enrichment analysis.""",
            system_message="""You are an omics data specialist. Your task is to answer gene/biomarker questions using the omic_analysis tool.

## STEP 1: EXTRACT PARAMETERS FROM THE QUERY
From the user's question, identify:
- Disease name (MUST match the variations list below)
- Organ (REQUIRED for efficiency)
- Cell type (OPTIONAL, if mentioned)

## STEP 2: CALL THE TOOL WITH YOUR BEST EXTRACTION
Call omic_analysis with the extracted parameters using your best judgment.

DISEASE NAME VARIATIONS (use EXACT names):
- Alzheimer's → "Alzheimer disease"
- Lung cancer → "lung adenocarcinoma"
- Breast cancer → "breast cancer"
- Colon cancer → "colorectal cancer"
- Pancreatic cancer → "pancreatic ductal adenocarcinoma"
- Leukemia → "leukemia"

ORGAN MAPPING (use EXACT names):
- Alzheimer's → "brain"
- Lung cancer → "lung"
- Breast cancer → "breast"
- Colon cancer → "colon"
- Pancreatic cancer → "pancreas"
- Blood cancer → "blood"

EXAMPLE CALLS:
✓ omic_analysis(disease="Alzheimer disease", organ="brain")
✓ omic_analysis(disease="lung adenocarcinoma", organ="lung")
✓ omic_analysis(disease="breast cancer", organ="breast", cell_type="epithelial cell")

## STEP 3: CHECK THE RESPONSE FOR "similar_terms"
The tool will return:
- "success": true → DATA FOUND, report results
- "success": false with "similar_terms" → NO EXACT MATCH, RETRY with one of the suggested terms
- "success": false with no suggestions → NO DATA AVAILABLE, report failure

## RETRY LOGIC - WHEN TO RETRY
✓ DO RETRY if response contains "similar_terms" → Try ONE alternative from the list
✓ DO RETRY if response suggests the disease/organ combination doesn't exist in the database
✗ DO NOT RETRY if you've already tried similar_terms and got another "no match" result
✗ DO NOT RETRY more than 2 times total

## CRITICAL INSTRUCTIONS
1. First attempt: Use your best disease/organ extraction
2. If "similar_terms" are provided: Try ONE alternative from the suggestions
3. After second attempt with similar_terms: Accept the result (success or failure)
4. Always report honest results - either successful data or "No data found for this disease/organ combination"
5. DO NOT attempt to answer from general knowledge - only report what the tool returns

## OUTPUT FORMAT
After tool calls complete, write a concise summary:
1. If successful: Present the gene list and pathway data
2. If no data after retry: "No data found. Similar terms available: [list]. Try searching with: [suggestion]"
3. STOP - analysis complete""",
            model_client_stream=False,
            reflect_on_tool_use=True,
            max_tool_iterations=2,
        )
        agents.append(Omicxagent)

        # test_agent = AssistantAgent(
        #     name="TestAgent",
        #     model_client=self.model_client,
        #     tools=[],
        #     description="""Greeting Agent""",
        #     system_message="""You are a friendly assistant. Greet the user warmly and offer help.""",
        #     reflect_on_tool_use=True,
        # )
        # agents.append(test_agent)

        self.team = MagenticOneGroupChat(
            agents, 
            model_client=self.model_client,
            final_answer_prompt=MAGNETIC_ONE_ORCHESTRATOR_PROMPT
        )
        
        print(f"✅ Team initialized with {len(agents)} agent(s)")
        return self.team

    def save_conversation_log(self, query: str, messages: list):
        """Save the conversation log to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_log_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "messages": messages,
            "final_message": messages[-1] if messages else None
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)
        
        return filepath

    def save_final_message(self, query: str, final_message: dict):
        """Save only the final message to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"final_message_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        final_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "final_message": final_message
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False, default=str)
        
        return filepath

    def save_report_markdown(self, query: str, report_content: str) -> str:
        """
        Save the final report as a markdown file in the session directory.
        
        Args:
            query: The original user query
            report_content: The markdown content of the report (from TaskResult)
            
        Returns:
            str: Path to the saved report file, or None if failed
        """
        try:
            # Get the session directory
            sessions_base = get_path('sessions.base', absolute=True, create=True)
            session_dir = os.path.join(sessions_base, self.session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            # Generate report filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"report_{timestamp}.md"
            report_path = os.path.join(session_dir, report_filename)
            
            # Build the markdown report with metadata header
            report_header = f"""---
title: OmniCellAgent Analysis Report
session_id: {self.session_id}
generated_at: {datetime.now().isoformat()}
query: {query}
---

"""
            
            full_report = report_header + report_content
            
            # Save the report
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(full_report)
            
            print(f"📄 Report saved to: {report_path}")
            return report_path
            
        except Exception as e:
            print(f"[ERROR] Failed to save report: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def run_query(self, query: str):
        """Run a query through the magnetic agent system."""
        if not self.team:
            await self.initialize_team()
        
        print(f"\n🔬 Processing query: {query}")
        print("=" * 80)
        
        captured_messages = []
        
        async for message in self.team.run_stream(task=query):
            message_dict = {
                "type": type(message).__name__,
                "content": str(message.content) if hasattr(message, 'content') else str(message),
                "source": getattr(message, 'source', 'unknown'),
                "timestamp": datetime.now().isoformat()
            }
            
            if hasattr(message, 'role'):
                message_dict["role"] = message.role
            
            captured_messages.append(message_dict)
            
            # Enhanced display based on message type
            message_type = message_dict['type']
            source = message_dict['source']
            content = message_dict['content']
            
            if message_type == "TextMessage":
                print(f"💬 [{source}]: {content}")
            elif message_type == "ToolCallRequestEvent":
                print(f"🔧 [{source} - Tool Call]: {content}")
                # Try to extract tool details if available
                if hasattr(message, 'content') and hasattr(message.content, '__iter__'):
                    for tool_call in message.content:
                        if hasattr(tool_call, 'name') and hasattr(tool_call, 'arguments'):
                            print(f"   └─ Tool: {tool_call.name}, Args: {tool_call.arguments}")
            elif message_type == "ToolCallExecutionEvent":
                print(f"📊 [{source} - Tool Results]: {content}")
                # Try to extract execution details if available
                if hasattr(message, 'content') and hasattr(message.content, '__iter__'):
                    for result in message.content:
                        if hasattr(result, 'call_id') and hasattr(result, 'content'):
                            print(f"   └─ Result for {result.call_id}: {result.content}")
            elif message_type == "ToolCallSummaryMessage":
                print(f"📋 [{source} - Tool Summary]: {content}")
            elif message_type == "TaskResult":
                # Extract the final report content from TaskResult
                if hasattr(message, 'messages') and message.messages:
                    final_content = message.messages[-1].content if hasattr(message.messages[-1], 'content') else str(message.messages[-1])
                    print(f"📄 [Final Report]: {final_content[:500]}...")  # Print first 500 chars
                    
                    # Save the report as markdown
                    report_path = self.save_report_markdown(query, final_content)
                    if report_path:
                        print(f"\n✅ Report saved to: {report_path}")
        
        if captured_messages:
            self.save_conversation_log(query, captured_messages)
            final_message = captured_messages[-1]
            self.save_final_message(query, final_message)
        else:
            print("\n⚠️  No messages were captured")
    
    async def close(self):
        """Clean up resources."""
        if self.model_client:
            await self.model_client.close()


async def main():
    """Main function to run the magnetic agent system."""
    parser = argparse.ArgumentParser(description="Magnetic Agent System for Biomedical Research")
    parser.add_argument(
        "--query", 
        type=str, 
        default="What are the key dysfunctional signaling targets in microglia of AD?",
        help="Query to process"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="gemini-2.5-pro",
        help="Model to use (default: gemini-2.5-pro)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.0,
        help="Model temperature (default: 0.0)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,  # Will use path from config
        help="Directory to save conversation logs (default: from config)"
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,  # Will auto-generate if not provided
        help="Session ID for this agent run. Used to organize analysis outputs. (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Use log directory from args or default from config
    if args.log_dir is None:
        args.log_dir = get_path('logs.base', absolute=True, create=True)
    else:
        os.makedirs(args.log_dir, exist_ok=True)
    
    system = SimpleMagneticAgentSystem(
        model_name=args.model, 
        temperature=args.temperature,
        log_dir=args.log_dir,
        session_id=args.session_id
    )

    await system.run_query(args.query)
    
    await system.close()


if __name__ == "__main__":
    asyncio.run(main())
