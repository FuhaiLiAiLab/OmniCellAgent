"""
Worker implementation for executing LangGraph agent tasks.

This module handles:
1. Running the LangGraph agent
2. Extracting structured outputs and artifacts
3. Converting agent results to A2A format
"""

import asyncio
import sys
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import base64
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.langgraph_agent import LangGraphOmniCellAgent, AgentState
from fasta2a_service.storage import Artifact, Task, TaskStatus, FileStorage


class ArtifactExtractor:
    """Extracts structured artifacts from LangGraph agent state"""
    
    @staticmethod
    def extract_final_report(state: AgentState) -> Optional[Artifact]:
        """Extract the final report as a TextPart artifact"""
        report = state.get("final_report")
        if report:
            return Artifact(
                part_type="TextPart",
                data=report,
                metadata={
                    "type": "final_report",
                    "format": "markdown"
                }
            )
        return None
    
    @staticmethod
    def extract_research_plan(state: AgentState) -> Optional[Artifact]:
        """Extract the research plan as a DataPart artifact"""
        plan = state.get("plan", [])
        if plan:
            return Artifact(
                part_type="DataPart",
                data={"result": plan},
                metadata={
                    "type": "research_plan",
                    "schema": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "description": {"type": "string"},
                                "assigned_agent": {"type": "string"},
                                "status": {"type": "string"},
                                "result": {"type": "string"},
                                "error": {"type": "string"},
                                "attempts": {"type": "integer"}
                            }
                        }
                    }
                }
            )
        return None
    
    @staticmethod
    def extract_agent_state(state: AgentState) -> Artifact:
        """Extract full agent state as a DataPart artifact"""
        # Convert state to serializable format
        serializable_state = {}
        for key, value in state.items():
            if key == "messages":
                # Convert messages to dict format
                serializable_state[key] = [
                    {
                        "type": msg.__class__.__name__,
                        "content": str(msg.content) if hasattr(msg, 'content') else str(msg)
                    }
                    for msg in value
                ]
            else:
                serializable_state[key] = value
        
        return Artifact(
            part_type="DataPart",
            data={"result": serializable_state},
            metadata={
                "type": "langgraph_state",
                "description": "Complete LangGraph agent state including all execution details"
            }
        )
    
    @staticmethod
    def extract_gene_data(state: AgentState) -> Optional[Artifact]:
        """Extract gene expression data as a DataPart artifact"""
        shared_data = state.get("shared_data", {})
        genes = shared_data.get("top_genes", [])
        
        if genes:
            return Artifact(
                part_type="DataPart",
                data={"result": genes},
                metadata={
                    "type": "gene_expression_data",
                    "count": len(genes),
                    "schema": {
                        "type": "array",
                        "description": "Top differentially expressed genes with statistics"
                    }
                }
            )
        return None
    
    @staticmethod
    def extract_pathway_data(state: AgentState) -> Optional[Artifact]:
        """Extract pathway enrichment data as a DataPart artifact"""
        shared_data = state.get("shared_data", {})
        pathways = shared_data.get("pathways", [])
        
        if pathways:
            return Artifact(
                part_type="DataPart",
                data={"result": pathways},
                metadata={
                    "type": "pathway_enrichment",
                    "count": len(pathways),
                    "schema": {
                        "type": "array",
                        "description": "Pathway enrichment analysis results"
                    }
                }
            )
        return None
    
    @staticmethod
    def extract_literature_data(state: AgentState) -> Optional[Artifact]:
        """Extract literature references as a DataPart artifact"""
        shared_data = state.get("shared_data", {})
        papers = shared_data.get("paper_dois", [])
        
        if papers:
            return Artifact(
                part_type="DataPart",
                data={"result": papers},
                metadata={
                    "type": "literature_references",
                    "count": len(papers),
                    "schema": {
                        "type": "array",
                        "description": "Literature references with DOIs and citations"
                    }
                }
            )
        return None
    
    @staticmethod
    def extract_process_log(state: AgentState) -> Optional[Artifact]:
        """Extract process execution log as a DataPart artifact"""
        process_log = state.get("process_log", [])
        
        if process_log:
            return Artifact(
                part_type="DataPart",
                data={"result": process_log},
                metadata={
                    "type": "process_log",
                    "count": len(process_log),
                    "description": "Detailed execution log of all agent actions"
                }
            )
        return None
    
    @staticmethod
    def extract_plots(state: AgentState, session_dir: str) -> List[Artifact]:
        """Extract plot files as base64-encoded DataPart artifacts"""
        artifacts = []
        
        # Look for plots in session directory
        session_path = Path(session_dir)
        if session_path.exists():
            for plot_file in session_path.glob("**/*.png"):
                try:
                    with open(plot_file, "rb") as f:
                        plot_data = base64.b64encode(f.read()).decode('utf-8')
                    
                    artifacts.append(Artifact(
                        part_type="DataPart",
                        data={
                            "result": {
                                "filename": plot_file.name,
                                "format": "png",
                                "data": plot_data,
                                "encoding": "base64"
                            }
                        },
                        metadata={
                            "type": "visualization",
                            "plot_type": "analysis_plot",
                            "filename": plot_file.name
                        }
                    ))
                except Exception as e:
                    print(f"Failed to encode plot {plot_file}: {e}")
        
        return artifacts
    
    @classmethod
    def extract_all_artifacts(cls, state: AgentState, session_dir: str) -> List[Artifact]:
        """Extract all artifacts from agent state"""
        artifacts = []
        
        # Extract in order of importance
        extractors = [
            cls.extract_final_report,
            cls.extract_research_plan,
            cls.extract_gene_data,
            cls.extract_pathway_data,
            cls.extract_literature_data,
            cls.extract_process_log,
            cls.extract_agent_state,
        ]
        
        for extractor in extractors:
            artifact = extractor(state)
            if artifact:
                artifacts.append(artifact)
        
        # Add plots
        artifacts.extend(cls.extract_plots(state, session_dir))
        
        return artifacts


class LangGraphWorker:
    """Worker that executes LangGraph agent tasks"""
    
    def __init__(self, storage: FileStorage, model_name: str = "gemini-2.0-flash-exp"):
        self.storage = storage
        self.model_name = model_name
    
    async def execute_task(self, task: Task, context_state: Optional[Dict[str, Any]] = None) -> Task:
        """
        Execute a task using the LangGraph agent.
        
        Args:
            task: The task to execute
            context_state: Previous LangGraph state for conversation continuity
        
        Returns:
            Updated task with results and artifacts
        """
        try:
            # Update task status
            task.status = TaskStatus.RUNNING
            self.storage.save_task(task)
            
            # Create agent instance
            agent = LangGraphOmniCellAgent(
                model_name=self.model_name,
                session_id=task.context_id  # Use context_id as session_id
            )
            
            # If we have previous context, we could restore some state here
            # For now, LangGraph agent creates fresh state each time
            # TODO: Implement state restoration for true conversation continuity
            
            # Run the agent
            report = await agent.run(task.prompt)
            
            # Get the final state from the agent
            # We need to modify the agent to return the state
            # For now, we'll create a minimal state
            # This is a limitation we should document
            
            # Create a basic final state
            final_state: AgentState = {
                "query": task.prompt,
                "session_id": task.context_id,
                "plan": [],
                "current_task_index": 0,
                "plan_revision_count": 0,
                "max_plan_revisions": 2,
                "messages": [],
                "agent_outputs": {},
                "shared_data": {},
                "process_log": [],
                "final_report": report,
                "status": "completed"
            }
            
            # Extract artifacts
            artifacts = ArtifactExtractor.extract_all_artifacts(
                final_state,
                agent.session_dir
            )
            
            # Update task
            task.status = TaskStatus.COMPLETED
            task.artifacts = artifacts
            task.completed_at = None  # Will be set by storage
            
            # Add final message to history
            task.message_history.append({
                "role": "assistant",
                "content": report,
                "timestamp": task.updated_at
            })
            
            # Update context
            self.storage.update_context_with_task(
                task.context_id,
                task.task_id,
                final_state
            )
            
            self.storage.save_task(task)
            return task
            
        except Exception as e:
            import traceback
            task.status = TaskStatus.FAILED
            task.error = f"{str(e)}\n{traceback.format_exc()}"
            self.storage.save_task(task)
            return task
