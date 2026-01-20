"""
LangGraph UI Adapter

This module provides a clean adapter interface between the LangGraph-based OmniCellAgent
and the Dash UI. It handles:
1. Streaming agent outputs to the UI in real-time
2. Converting LangGraph events to UI-compatible messages
3. Managing session directories and file outputs
4. Supporting HTML visualization embeds in the UI
5. Incremental conversation support
"""

import asyncio
import os
import json
import re
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

# Import the LangGraph agent
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.langgraph_agent import (
    LangGraphOmniCellAgent, 
    AgentState,
    create_session_dir,
    get_current_session_dir,
    extract_text_from_llm_response
)

# Sessions directory for UI access
SESSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sessions')


class UILangGraphAdapter:
    """
    Adapter class that connects the LangGraph OmniCellAgent to the Dash UI.
    
    This adapter:
    - Creates a LangGraph agent with a UI-specific session
    - Provides streaming execution that yields UI-compatible events
    - Converts agent phase transitions, tool calls, and results to UI messages
    - Handles multimedia outputs (plots, HTML files) for UI rendering
    - Supports incremental conversation across multiple rounds
    """
    
    def __init__(self, session_id: str, model_name: str = "gemini-3-pro-preview"):
        """
        Initialize the UI adapter.
        
        Args:
            session_id: Unique session identifier (from UI)
            model_name: LLM model to use
        """
        self.session_id = session_id
        self.model_name = model_name
        self.agent: Optional[LangGraphOmniCellAgent] = None
        self.session_dir: Optional[str] = None
        
        # Conversation state for incremental chat
        self.round_number = 1
        self.conversation_summaries: List[str] = []
        self.last_query: Optional[str] = None
        self.last_report: Optional[str] = None
        
        # Callbacks for UI updates
        self._process_step_callback: Optional[Callable] = None
        self._left_chat_callback: Optional[Callable] = None
        self._multimedia_callback: Optional[Callable] = None
        self._final_result_callback: Optional[Callable] = None
    
    def set_callbacks(self, 
                      process_step: Callable = None,
                      left_chat: Callable = None, 
                      multimedia: Callable = None,
                      final_result: Callable = None):
        """Set callbacks for UI updates."""
        self._process_step_callback = process_step
        self._left_chat_callback = left_chat
        self._multimedia_callback = multimedia
        self._final_result_callback = final_result
    
    async def initialize(self):
        """Initialize the LangGraph agent with session-specific directory."""
        self.session_dir = os.path.join(SESSIONS_DIR, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        
        self.agent = LangGraphOmniCellAgent(
            model_name=self.model_name,
            session_id=self.session_id
        )
        self._emit_process_step("System", f"🔧 LangGraph Agent initialized (model: {self.model_name})")
        
        self.agent.session_dir = self.session_dir
        import agent.langgraph_agent as lg_module
        lg_module._GLOBAL_SESSION_DIR = self.session_dir
        lg_module._GLOBAL_SESSION_ID = self.session_id
        
        print(f"[LangGraph Adapter] Initialized for session {self.session_id}")
        print(f"[LangGraph Adapter] Session directory: {self.session_dir}")
    
    def _emit_process_step(self, agent: str, content: str, is_tool_call: bool = False):
        if self._process_step_callback:
            self._process_step_callback(self.session_id, agent, content, is_tool_call=is_tool_call)
    
    def _emit_left_chat(self, sender: str, content: str, msg_type: str = "assistant"):
        if self._left_chat_callback:
            self._left_chat_callback(self.session_id, sender, content, msg_type)
    
    def _emit_multimedia(self, agent: str, content: str, media_type: str, media_url: str):
        if self._multimedia_callback:
            self._multimedia_callback(self.session_id, agent, content, 
                                      media_type=media_type, media_url=media_url)
    
    def _emit_final_result(self, content: str):
        if self._final_result_callback:
            self._final_result_callback(self.session_id, content)
    
    def _convert_path_to_web_url(self, file_path: str) -> str:
        sessions_dir_str = str(SESSIONS_DIR)
        if file_path.startswith(sessions_dir_str):
            relative_path = file_path[len(sessions_dir_str):].lstrip('/')
            return f"/assets/sessions/{relative_path}"
        elif '/sessions/' in file_path:
            idx = file_path.find('/sessions/')
            relative_part = file_path[idx + len('/sessions/'):].lstrip('/')
            return f"/assets/sessions/{relative_part}"
        return file_path
    
    def _get_media_type(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.html':
            return "html"
        elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg']:
            return "image"
        return "file"
    
    def _process_omic_result(self, result_content: Any):
        try:
            if isinstance(result_content, str):
                try:
                    result_dict = json.loads(result_content)
                except json.JSONDecodeError:
                    try:
                        import ast
                        result_dict = ast.literal_eval(result_content)
                    except (ValueError, SyntaxError):
                        return
            elif isinstance(result_content, dict):
                result_dict = result_content
            else:
                return
            
            plot_paths = result_dict.get('plot_paths', [])
            if not isinstance(plot_paths, list):
                return
            
            for plot_path in plot_paths:
                if not plot_path or not os.path.exists(plot_path):
                    continue
                web_url = self._convert_path_to_web_url(plot_path)
                media_type = self._get_media_type(plot_path)
                filename = os.path.basename(plot_path)
                self._emit_multimedia("OmicMiningAgent", f"🧬 Generated visualization: {filename}",
                                      media_type=media_type, media_url=web_url)
        except Exception as e:
            print(f"[LangGraph Adapter] Error processing omic result: {e}")
    
    def _scan_session_for_visualizations(self):
        if not self.session_dir or not os.path.exists(self.session_dir):
            return
        
        viz_dirs = [
            ("volcano_plots", "Volcano Plot"),
            ("enrichment_plots", "Enrichment Plot"),
            ("enrichment_results", "Enrichment Results"),
            ("plots", "Analysis Plot")
        ]
        emitted_files = set()
        
        for subdir, label_prefix in viz_dirs:
            dir_path = os.path.join(self.session_dir, subdir)
            if not os.path.exists(dir_path):
                continue
            for root, dirs, files in os.walk(dir_path):
                for file in sorted(files):
                    if file.startswith('.'):
                        continue
                    ext = os.path.splitext(file)[1].lower()
                    if ext not in ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.html']:
                        continue
                    file_path = os.path.join(root, file)
                    if file_path in emitted_files:
                        continue
                    emitted_files.add(file_path)
                    web_url = self._convert_path_to_web_url(file_path)
                    media_type = self._get_media_type(file_path)
                    self._emit_multimedia("Visualization", f"📊 {label_prefix}: {file}",
                                          media_type=media_type, media_url=web_url)
    
    async def run_stream(self, task: str, stop_event=None) -> str:
        if not self.agent:
            await self.initialize()
        
        try:
            self._emit_process_step("Orchestrator", f"📋 Starting analysis: {task[:100]}...")
            
            initial_state: AgentState = {
                "query": task,
                "session_id": self.session_id,
                "plan": [],
                "current_task_index": 0,
                "plan_revision_count": 0,
                "max_plan_revisions": 2,
                "messages": [],
                "agent_outputs": {},
                "shared_data": {
                    "top_genes": [], "paper_dois": [], "pathways": [],
                    "disease": "", "cell_type": "",
                },
                "process_log": [],
                "final_report": None,
                "status": "planning"
            }
            
            final_state = None
            step_count = 0
            
            async for event in self.agent.graph.astream(initial_state):
                if stop_event and stop_event.is_set():
                    self._emit_process_step("System", "🛑 Processing cancelled by user")
                    raise asyncio.CancelledError("Processing stopped by user")
                
                for node_name, state_update in event.items():
                    step_count += 1
                    if final_state is None:
                        final_state = {**initial_state}
                    final_state.update(state_update)
                    await self._process_graph_event(node_name, state_update, final_state, step_count)
            
            self._scan_session_for_visualizations()
            final_report = final_state.get("final_report", "No report generated") if final_state else "No report generated"
            
            self.last_query = task
            self.last_report = final_report
            
            self._emit_process_step("Final", "✅ Analysis completed successfully!")
            self._emit_final_result(final_report)
            
            try:
                from webapp.UI.dash_UI import save_session_report
                report_path = save_session_report(self.session_id, final_report, task)
                if report_path:
                    self._emit_process_step("System", f"📄 Report saved: {os.path.basename(report_path)}")
            except Exception as e:
                print(f"[LangGraph Adapter] Error saving report via UI: {e}")
            
            if final_state:
                self.agent._save_conversation_log(task, final_state)
            
            return final_report
            
        except asyncio.CancelledError:
            self._emit_process_step("System", "🛑 Processing was cancelled")
            raise
        except Exception as e:
            import traceback
            self._emit_process_step("System", f"❌ Error during analysis: {str(e)}")
            print(f"[LangGraph Adapter] Error: {traceback.format_exc()}")
            raise
    
    async def _process_graph_event(self, node_name: str, state_update: Dict, 
                                    full_state: Dict, step_count: int):
        if node_name == "planner":
            plan = state_update.get("plan", [])
            if plan:
                plan_summary = "\n".join([
                    f"  {i+1}. [{t['assigned_agent']}] {t['description'][:60]}..."
                    for i, t in enumerate(plan)
                ])
                self._emit_process_step("Orchestrator", 
                    f"📋 **Research Plan Created** ({len(plan)} tasks):\n{plan_summary}")
        
        elif node_name == "executor":
            current_idx = full_state.get("current_task_index", 0)
            plan = full_state.get("plan", [])
            
            if current_idx > 0 and current_idx <= len(plan):
                completed_task = plan[current_idx - 1]
                agent_name = completed_task.get("assigned_agent", "Agent")
                task_desc = completed_task.get("description", "")[:80]
                task_status = completed_task.get("status", "")
                
                if task_status == "completed":
                    result = completed_task.get("result", "")
                    self._emit_process_step(agent_name,
                        f"✅ **Task Completed**: {task_desc}...\n\n{self._truncate_result(result)}")
                    
                    if agent_name == "OmicMiningAgent":
                        agent_outputs = full_state.get("agent_outputs", {})
                        task_id = completed_task.get("id")
                        if task_id and task_id in agent_outputs:
                            output = agent_outputs[task_id]
                            if isinstance(output, dict) and "result" in output:
                                self._process_omic_result(output.get("result"))
                
                elif task_status == "failed":
                    error = completed_task.get("error", "Unknown error")
                    self._emit_process_step(agent_name,
                        f"❌ **Task Failed**: {task_desc}...\nError: {error}")
            
            if current_idx < len(plan):
                next_task = plan[current_idx]
                self._emit_process_step(next_task.get("assigned_agent", "Agent"),
                    f"🔄 **Starting Task {current_idx + 1}/{len(plan)}**: {next_task.get('description', '')[:80]}...")
        
        elif node_name == "replanner":
            revision_count = state_update.get("plan_revision_count", 0)
            self._emit_process_step("Orchestrator",
                f"🔄 **Re-planning** (attempt {revision_count}): Adjusting strategy based on results...")
        
        elif node_name == "reporter":
            self._emit_process_step("Reporter", "📄 **Generating Final Report**...")
    
    def _truncate_result(self, result: str, max_length: int = 1500) -> str:
        if not result:
            return ""
        result_str = str(result)
        if len(result_str) > max_length:
            return result_str[:max_length] + f"\n\n[... truncated ({len(result_str)} chars total)]"
        return result_str
    
    async def continue_conversation(self, new_query: str, stop_event=None) -> str:
        """Continue conversation with a new query, building on previous rounds."""
        if not self.agent:
            await self.initialize()
        
        self.round_number += 1
        self._emit_process_step("System",
            f"🔄 **Continuing Conversation** (Round {self.round_number})\n"
            f"Building on {len(self.conversation_summaries)} previous rounds...")
        
        context_prefix = ""
        if self.conversation_summaries:
            context_prefix = "## Previous Analysis Context\n\n"
            for i, summary in enumerate(self.conversation_summaries, 1):
                context_prefix += f"### Round {i}:\n{summary}\n\n"
            context_prefix += "---\n\n## New Query\n\n"
        
        enhanced_query = context_prefix + new_query
        result = await self.run_stream(enhanced_query, stop_event)
        
        summary = self._generate_round_summary(new_query, result)
        self.conversation_summaries.append(summary)
        self.last_query = new_query
        self.last_report = result
        
        return result
    
    def _generate_round_summary(self, query: str, report: str) -> str:
        summary_parts = [f"Query: {query[:200]}"]
        if "Key Findings" in report:
            start = report.find("Key Findings")
            end = report.find("\n#", start + 1)
            if end == -1:
                end = start + 1000
            summary_parts.append(f"Findings: {report[start:end][:500]}")
        if "Hypothesis" in report:
            start = report.find("Hypothesis")
            end = report.find("\n#", start + 1)
            if end == -1:
                end = start + 500
            summary_parts.append(f"Hypotheses: {report[start:end][:300]}")
        return "\n".join(summary_parts)


def create_langgraph_adapter(session_id: str, 
                              model_name: str = "gemini-3-pro-preview") -> UILangGraphAdapter:
    """Factory function to create a configured LangGraph adapter for the UI."""
    return UILangGraphAdapter(session_id=session_id, model_name=model_name)
