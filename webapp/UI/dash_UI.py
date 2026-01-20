import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import uuid
import asyncio
import time
import threading
import argparse
import os
from datetime import datetime, timedelta
from autogen_core.models import ModelInfo

# === GLOBAL CONFIGURATION ===
PROCESSING_TIMEOUT = 3000  # Processing timeout in seconds (default: 50 minutes)

# Ensure project root is on sys.path so 'agent' is importable when running this script directly
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Go up 2 levels: UI -> webapp -> bioRAG
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Sessions directory for serving static plot files (defined early for use in classes)
# Sessions are stored inside webapp/sessions/ for proper Flask serving
SESSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sessions')

from dotenv import load_dotenv
load_dotenv()
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import MagenticOneGroupChat, RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_core import CancellationToken

# Import the SimpleMagneticAgentSystem from agent (AutoGen-based)
from agent.simple_magentic_agent import SimpleMagneticAgentSystem

# Import the LangGraph adapter for the new LangGraph-based agent
# Try both import paths for flexibility (webapp.UI or relative)
try:
    from webapp.UI.langgraph_adapter import UILangGraphAdapter, create_langgraph_adapter
except ImportError:
    from .langgraph_adapter import UILangGraphAdapter, create_langgraph_adapter


class UIMagneticAgentSystem(SimpleMagneticAgentSystem):
    """UI-specific extension of SimpleMagneticAgentSystem for Dash integration."""
    
    def __init__(self, *args, **kwargs):
        # Set default log directory for UI
        if 'log_dir' not in kwargs:
            kwargs['log_dir'] = "./logs"
        super().__init__(*args, **kwargs)
    
    async def run_stream_for_ui(self, task: str, session_id: str, stop_event=None):
        """Run the agent system and yield messages formatted for UI with cancellation support."""
        if not self.team:
            await self.initialize_team()
        
        captured_messages = []
        
        try:
            async for message in self.team.run_stream(task=task):
                # Check for cancellation at the start of each iteration
                if stop_event and stop_event.is_set():
                    add_process_step(session_id, "System", "🛑 Processing cancelled by user")
                    raise asyncio.CancelledError("Processing stopped by user")
                
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
                
                if source == "user":
                    continue
                
                if message_type == "TextMessage":
                    formatted_content = f"💬 [{source}]: {content}"
                    add_process_step(session_id, source, content)
                elif message_type == "ToolCallRequestEvent":
                    details = ""
                    if hasattr(message, 'content') and hasattr(message.content, '__iter__'):
                        for tool_call in message.content:
                            if hasattr(tool_call, 'name') and hasattr(tool_call, 'arguments'):
                                details += f"\n🔧 Tool: {tool_call.name}, Args: {tool_call.arguments}"
                    formatted_content = f"🔧 [{source} - Tool Call]: {content}{details}"
                    add_process_step(session_id, source, formatted_content, is_tool_call=True)
                elif message_type == "ToolCallExecutionEvent":
                    details = ""
                    if hasattr(message, 'content') and hasattr(message.content, '__iter__'):
                        for result in message.content:
                            if hasattr(result, 'call_id') and hasattr(result, 'content'):
                                details += f"\n📊 Result for {result.call_id}: {str(result.content)[:500]}..."
                    formatted_content = f"📊 [{source} - Tool Results]: {details}"
                    add_process_step(session_id, source, formatted_content, is_tool_call=True)

                    # Check if this is an OmicMiningAgent result (by source name or by detecting omic_analysis in call_id)
                    is_omic_result = source == "OmicMiningAgent"
                    
                    # Also check if any result is from omic_analysis tool
                    if not is_omic_result and hasattr(message, 'content'):
                        for result in message.content:
                            if hasattr(result, 'call_id') and 'omic' in str(result.call_id).lower():
                                is_omic_result = True
                                break
                    
                    print(f"[DEBUG] ToolCallExecutionEvent - source: {source}, is_omic_result: {is_omic_result}")

                    if is_omic_result:
                        try:
                            # breakpoint()
                            for result in message.content:
                                if hasattr(result, 'content'):
                                    # Parse the string content as JSON/dict
                                    import ast
                                    import json
                                    
                                    content_str = result.content
                                    if isinstance(content_str, str):
                                        try:
                                            # Try parsing as JSON first
                                            result_dict = json.loads(content_str)
                                        except json.JSONDecodeError:
                                            try:
                                                # If JSON fails, try literal_eval for Python dict strings
                                                result_dict = ast.literal_eval(content_str)
                                            except (ValueError, SyntaxError):
                                                print(f"[DEBUG] Could not parse OmicMiningAgent result: {content_str[:100]}...")
                                                continue
                                    elif isinstance(content_str, dict):
                                        result_dict = content_str
                                    else:
                                        continue
                                    
                                    # Extract plot_paths
                                    plot_paths = result_dict.get('plot_paths', [])
                                    print(f"[DEBUG] Found plot_paths in result: {plot_paths}")
                                    print(f"[DEBUG] SESSIONS_DIR = {SESSIONS_DIR}")
                                    
                                    if isinstance(plot_paths, list) and len(plot_paths) > 0:
                                        for plot_path in plot_paths:
                                            print(f"[DEBUG] Processing plot_path: {plot_path}")
                                            
                                            # Convert absolute paths to web-accessible URLs
                                            # Use /assets/sessions/ because Dash serves assets folder automatically
                                            # We have a symlink: assets/sessions -> ../sessions
                                            web_url = plot_path
                                            sessions_dir_str = str(SESSIONS_DIR)
                                            
                                            if plot_path.startswith(sessions_dir_str):
                                                # Convert absolute path to relative web URL via assets
                                                relative_path = plot_path[len(sessions_dir_str):].lstrip('/')
                                                web_url = f"/assets/sessions/{relative_path}"
                                                print(f"[DEBUG] Converted to web_url: {web_url}")
                                            elif '/sessions/' in plot_path:
                                                # Already contains sessions path, convert to assets path
                                                idx = plot_path.find('/sessions/')
                                                relative_part = plot_path[idx + len('/sessions/'):].lstrip('/')
                                                web_url = f"/assets/sessions/{relative_part}"
                                                print(f"[DEBUG] Converted to web_url (method 2): {web_url}")
                                            else:
                                                print(f"[DEBUG] Could not convert path, using as-is: {web_url}")
                                            
                                            # Determine media type based on file extension
                                            file_ext = os.path.splitext(plot_path)[1].lower()
                                            if file_ext == '.html':
                                                media_type = "html"
                                            elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg']:
                                                media_type = "image"
                                            else:
                                                media_type = "file"
                                            
                                            print(f"[DEBUG] Adding multimedia step: media_type={media_type}, web_url={web_url}")
                                            add_multimedia_process_step(
                                                session_id, 
                                                source, 
                                                f"🧬 OmicMiningAgent plot: {os.path.basename(plot_path)}",
                                                media_type=media_type,
                                                media_url=web_url
                                            )
                                    else:
                                        print(f"[DEBUG] plot_paths is not a list: {plot_paths}")
                        except Exception as e:
                            print(f"[DEBUG] Error processing OmicMiningAgent results: {e}")
                            import traceback
                            traceback.print_exc()

                elif message_type == "ToolCallSummaryMessage":
                    formatted_content = f"📋 [{source} - Tool Summary]: {content}"
                    add_process_step(session_id, source, formatted_content)
                elif message_type == "TaskResult":
                    formatted_content = message.messages[-1].content#.replace("\\'","'")
                    # Add final result to chat
                    append_to_final_result(session_id, formatted_content)
                    add_process_step(session_id, "Final", "✅ Task completed successfully!")
                    
                    # Save the report as markdown
                    try:
                        report_path = save_session_report(session_id, formatted_content, task)
                        if report_path:
                            report_filename = os.path.basename(report_path)
                            add_process_step(session_id, "System", f"📄 Report saved: {report_filename}")
                    except Exception as e:
                        print(f"[DEBUG] Error saving report: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    formatted_content = f"ℹ️ [{source} - {message_type}]: {content}"
                    add_process_step(session_id, source, formatted_content)
                
                # Check for cancellation again after processing the message
                if stop_event and stop_event.is_set():
                    add_process_step(session_id, "System", "🛑 Processing cancelled by user")
                    raise asyncio.CancelledError("Processing stopped by user")
            
            if captured_messages:
                self.save_conversation_log(task, captured_messages)
                final_message = captured_messages[-1]
                self.save_final_message(task, final_message)
                
                # Add final result if we have a final message and it's not already added
                if final_message and final_message.get('type') != 'TaskResult':
                    append_to_final_result(session_id, final_message.get('content', 'Processing completed.'))
            else:
                add_process_step(session_id, "System", "⚠️ No messages were captured")
                append_to_final_result(session_id, "No results were generated.")
                
        except asyncio.CancelledError:
            # Handle cancellation gracefully
            add_process_step(session_id, "System", "🛑 Processing was cancelled")
            add_left_chat_message(session_id, "System", "🛑 Processing was cancelled by user", "system")
            raise
        except Exception as e:
            add_process_step(session_id, "System", f"❌ Error during processing: {str(e)}")
            add_left_chat_message(session_id, "System", f"❌ Error occurred: {str(e)}", "system")


# ==============================================================================
# REPORT SAVING UTILITIES
# ==============================================================================

def collect_visualization_htmls(session_dir: str) -> dict:
    """
    Scan session directory for visualization HTML files.
    
    Returns:
        dict: {"volcano": [...], "enrichment": [...], "kegg": [...]}
              Each entry is a tuple of (relative_path, absolute_path)
    """
    html_files = {
        "volcano": [],
        "enrichment": [],
        "kegg": []
    }
    
    # Collect volcano plots
    volcano_dir = os.path.join(session_dir, "volcano_plots")
    if os.path.exists(volcano_dir):
        for f in sorted(os.listdir(volcano_dir)):
            if f.endswith('.html'):
                rel_path = f"volcano_plots/{f}"
                abs_path = os.path.join(session_dir, rel_path)
                html_files["volcano"].append((rel_path, abs_path))
    
    # Collect enrichment plots (lollipop, bar charts)
    enrichment_dir = os.path.join(session_dir, "enrichment_plots")
    if os.path.exists(enrichment_dir):
        for f in sorted(os.listdir(enrichment_dir)):
            if f.endswith('.html'):
                rel_path = f"enrichment_plots/{f}"
                abs_path = os.path.join(session_dir, rel_path)
                html_files["enrichment"].append((rel_path, abs_path))
    
    # Collect KEGG pathway plots
    plots_dir = os.path.join(session_dir, "plots")
    if os.path.exists(plots_dir):
        for f in sorted(os.listdir(plots_dir)):
            if f.endswith('.html'):
                rel_path = f"plots/{f}"
                abs_path = os.path.join(session_dir, rel_path)
                html_files["kegg"].append((rel_path, abs_path))
    
    return html_files


def read_and_embed_html(html_path: str, title: str = None) -> str:
    """
    Read HTML file and embed its content directly in markdown.
    Only strips the <html> and </html> tags to avoid conflicts when embedding.
    
    Args:
        html_path: Absolute path to HTML file
        title: Optional title for the visualization
        
    Returns:
        str: Markdown with embedded HTML content
    """
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Only remove <html> and </html> tags - user confirmed this is all that's needed
        import re
        html_content = re.sub(r'<html[^>]*>', '', html_content, flags=re.IGNORECASE)
        html_content = re.sub(r'</html>', '', html_content, flags=re.IGNORECASE)
        html_content = html_content.strip()
        
        # Create a collapsible section with embedded HTML
        filename = os.path.basename(html_path)
        section_title = title if title else filename
        
        # Embed the cleaned HTML in a details/summary block for collapsibility
        embedded = f"""
<details>
<summary><strong>{section_title}</strong> (click to expand)</summary>

{html_content}

</details>

"""
        return embedded
        
    except Exception as e:
        print(f"[WARNING] Could not read HTML file {html_path}: {e}")
        return f"- [{os.path.basename(html_path)}]({html_path}) *(could not embed)*\n"


def generate_visualization_appendix(html_files: dict, session_dir: str) -> str:
    """
    Generate markdown appendix section with embedded HTML visualizations.
    
    Args:
        html_files: dict from collect_visualization_htmls()
        session_dir: Session directory path for reading files
        
    Returns:
        str: Markdown formatted appendix section with embedded HTML
    """
    # Check if there are any HTML files
    total_files = sum(len(files) for files in html_files.values())
    if total_files == 0:
        return ""
    
    appendix = """

---

## Appendix: Interactive Visualizations

The following interactive visualizations were generated during the analysis.

"""
    
    # Volcano plots section
    if html_files["volcano"]:
        appendix += "### Volcano Plots\n\n"
        appendix += "Volcano plots showing differential expression analysis results (log2 fold change vs significance).\n\n"
        for rel_path, abs_path in html_files["volcano"]:
            appendix += read_and_embed_html(abs_path, os.path.basename(rel_path))
    
    # Enrichment plots section
    if html_files["enrichment"]:
        appendix += "### Enrichment Analysis Plots\n\n"
        appendix += "Lollipop and bar charts showing pathway/gene set enrichment results.\n\n"
        for rel_path, abs_path in html_files["enrichment"]:
            appendix += read_and_embed_html(abs_path, os.path.basename(rel_path))
    
    # KEGG plots section
    if html_files["kegg"]:
        appendix += "### KEGG Pathway Visualizations\n\n"
        appendix += "KEGG pathway analysis dot plots and combined visualizations.\n\n"
        for rel_path, abs_path in html_files["kegg"]:
            appendix += read_and_embed_html(abs_path, os.path.basename(rel_path))
    
    return appendix


def save_session_report(session_id: str, report_content: str, query: str = None) -> str:
    """
    Save the final report as a markdown file in the session directory.
    Automatically appends visualization links if HTML files are present.
    
    Args:
        session_id: Unique session identifier
        report_content: The markdown content of the report (from TaskResult)
        query: Optional original query for context
        
    Returns:
        str: Path to the saved report file, or None if failed
    """
    try:
        # Create session directory if it doesn't exist
        session_dir = os.path.join(SESSIONS_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Generate report filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report_{timestamp}.md"
        report_path = os.path.join(session_dir, report_filename)
        
        # Build the markdown report with metadata header
        report_header = f"""---
title: OmniCellAgent Analysis Report
session_id: {session_id}
generated_at: {datetime.now().isoformat()}
query: {query if query else 'N/A'}
---

"""
        
        # Check for visualization HTML files and generate appendix
        html_files = collect_visualization_htmls(session_dir)
        visualization_appendix = generate_visualization_appendix(html_files, session_dir)
        
        # Combine: header + content + appendix
        full_report = report_header + report_content + visualization_appendix
        
        # Save the report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(full_report)
        
        # Log what was included
        total_htmls = sum(len(files) for files in html_files.values())
        if total_htmls > 0:
            print(f"[DEBUG] Report saved to: {report_path} (with {total_htmls} visualization links)")
        else:
            print(f"[DEBUG] Report saved to: {report_path}")
        
        # Also generate static versions (PNG-based MD and PDF)
        try:
            static_md_path = save_static_report(session_id, report_content, query, session_dir)
            if static_md_path:
                # Convert to HTML
                html_path = convert_md_to_html(static_md_path)
                # Convert to PDF
                pdf_path = convert_md_to_pdf(static_md_path)
        except Exception as e:
            print(f"[DEBUG] Static report generation skipped: {e}")
        
        return report_path
        
    except Exception as e:
        print(f"[ERROR] Failed to save report: {e}")
        import traceback
        traceback.print_exc()
        return None


def collect_visualization_pngs(session_dir: str) -> dict:
    """
    Scan session directory for visualization PNG files.
    
    Returns:
        dict: {"volcano": [...], "enrichment": [...], "kegg": [...]}
              Each entry is a tuple of (relative_path, absolute_path)
    """
    png_files = {
        "volcano": [],
        "enrichment": [],
        "kegg": []
    }
    
    # Collect volcano plots (PNG)
    volcano_dir = os.path.join(session_dir, "volcano_plots")
    if os.path.exists(volcano_dir):
        for f in sorted(os.listdir(volcano_dir)):
            if f.endswith('.png'):
                rel_path = f"volcano_plots/{f}"
                abs_path = os.path.join(session_dir, rel_path)
                png_files["volcano"].append((rel_path, abs_path))
    
    # Collect enrichment plots (PNG) - check both enrichment_plots and enrichment_results subdirs
    enrichment_dir = os.path.join(session_dir, "enrichment_plots")
    if os.path.exists(enrichment_dir):
        for f in sorted(os.listdir(enrichment_dir)):
            if f.endswith('.png'):
                rel_path = f"enrichment_plots/{f}"
                abs_path = os.path.join(session_dir, rel_path)
                png_files["enrichment"].append((rel_path, abs_path))
    
    # Also check enrichment_results subdirectories for PNGs
    enrichment_results_dir = os.path.join(session_dir, "enrichment_results")
    if os.path.exists(enrichment_results_dir):
        for subdir in os.listdir(enrichment_results_dir):
            subdir_path = os.path.join(enrichment_results_dir, subdir)
            if os.path.isdir(subdir_path):
                for f in sorted(os.listdir(subdir_path)):
                    if f.endswith('.png'):
                        rel_path = f"enrichment_results/{subdir}/{f}"
                        abs_path = os.path.join(session_dir, rel_path)
                        png_files["enrichment"].append((rel_path, abs_path))
    
    # Collect KEGG pathway plots (PNG)
    plots_dir = os.path.join(session_dir, "plots")
    if os.path.exists(plots_dir):
        for f in sorted(os.listdir(plots_dir)):
            if f.endswith('.png'):
                rel_path = f"plots/{f}"
                abs_path = os.path.join(session_dir, rel_path)
                png_files["kegg"].append((rel_path, abs_path))
    
    return png_files


def generate_static_visualization_appendix(png_files: dict) -> str:
    """
    Generate markdown appendix section with embedded PNG images (for PDF conversion).
    
    Args:
        png_files: dict from collect_visualization_pngs()
        
    Returns:
        str: Markdown formatted appendix section with embedded images
    """
    # Check if there are any PNG files
    total_files = sum(len(files) for files in png_files.values())
    if total_files == 0:
        return ""
    
    appendix = """

---

## Appendix: Visualizations

"""
    
    # Volcano plots section
    if png_files["volcano"]:
        appendix += "### Volcano Plots\n\n"
        appendix += "Volcano plots showing differential expression analysis results (log2 fold change vs significance).\n\n"
        for rel_path, abs_path in png_files["volcano"]:
            filename = os.path.basename(rel_path)
            appendix += f"**{filename}**\n\n"
            appendix += f"![{filename}]({rel_path})\n\n"
    
    # Enrichment plots section
    if png_files["enrichment"]:
        appendix += "### Enrichment Analysis Plots\n\n"
        appendix += "Lollipop and bar charts showing pathway/gene set enrichment results.\n\n"
        for rel_path, abs_path in png_files["enrichment"]:
            filename = os.path.basename(rel_path)
            appendix += f"**{filename}**\n\n"
            appendix += f"![{filename}]({rel_path})\n\n"
    
    # KEGG plots section
    if png_files["kegg"]:
        appendix += "### KEGG Pathway Visualizations\n\n"
        appendix += "KEGG pathway analysis dot plots and combined visualizations.\n\n"
        for rel_path, abs_path in png_files["kegg"]:
            filename = os.path.basename(rel_path)
            appendix += f"**{filename}**\n\n"
            appendix += f"![{filename}]({rel_path})\n\n"
    
    return appendix


def save_static_report(session_id: str, report_content: str, query: str, session_dir: str) -> str:
    """
    Save a static version of the report with PNG images (suitable for PDF conversion).
    
    Args:
        session_id: Unique session identifier
        report_content: The markdown content of the report
        query: Original query for context
        session_dir: Session directory path
        
    Returns:
        str: Path to the saved static report file, or None if failed
    """
    try:
        # Generate report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report_{timestamp}_static.md"
        report_path = os.path.join(session_dir, report_filename)
        
        # Build the markdown report with metadata header
        report_header = f"""---
title: OmniCellAgent Analysis Report
session_id: {session_id}
generated_at: {datetime.now().isoformat()}
query: {query if query else 'N/A'}
---

"""
        
        # Check for visualization PNG files and generate appendix
        png_files = collect_visualization_pngs(session_dir)
        visualization_appendix = generate_static_visualization_appendix(png_files)
        
        # Combine: header + content + appendix
        full_report = report_header + report_content + visualization_appendix
        
        # Save the report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(full_report)
        
        total_pngs = sum(len(files) for files in png_files.values())
        print(f"[DEBUG] Static report saved to: {report_path} (with {total_pngs} PNG images)")
        
        return report_path
        
    except Exception as e:
        print(f"[ERROR] Failed to save static report: {e}")
        return None


def convert_md_to_html(md_path: str) -> str:
    """
    Convert markdown file to HTML.
    
    Args:
        md_path: Path to the markdown file
        
    Returns:
        str: Path to the generated HTML file, or None if failed
    """
    try:
        import markdown
        
        # Read markdown content
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert to HTML with extensions for better formatting
        html_content = markdown.markdown(
            md_content,
            extensions=['tables', 'fenced_code', 'toc', 'meta']
        )
        
        # Get the directory for resolving relative image paths
        md_dir = os.path.dirname(md_path)
        
        # Wrap in a complete HTML document with styling
        html_template = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OmniCellAgent Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{ color: #4b225c; }}
        h1 {{ border-bottom: 2px solid #ffb74d; padding-bottom: 10px; }}
        h2 {{ border-bottom: 1px solid #e0e0e0; padding-bottom: 5px; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin: 10px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4b225c; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        code {{ background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background-color: #f4f4f4; padding: 15px; overflow-x: auto; border-radius: 5px; }}
        blockquote {{ border-left: 4px solid #ffb74d; margin: 15px 0; padding-left: 15px; color: #666; }}
        hr {{ border: none; border-top: 1px solid #e0e0e0; margin: 30px 0; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
        
        # Save HTML file
        html_path = md_path.replace('.md', '.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f"[DEBUG] HTML report saved to: {html_path}")
        return html_path
        
    except ImportError:
        print("[WARNING] markdown package not installed. Run: pip install markdown")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to convert MD to HTML: {e}")
        return None


def convert_md_to_pdf(md_path: str) -> str:
    """
    Convert markdown file to PDF using md2pdf.
    
    Args:
        md_path: Path to the markdown file
        
    Returns:
        str: Path to the generated PDF file, or None if failed
    """
    try:
        from md2pdf.core import md2pdf
        
        # Output PDF path
        pdf_path = md_path.replace('.md', '.pdf')
        
        # Get the directory for resolving relative image paths
        md_dir = os.path.dirname(md_path)
        
        # Read markdown content
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # CSS styling for PDF - images fit by width to stay within frame
        css_content = """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 40px;
        }
        h1, h2, h3 { color: #4b225c; }
        h1 { border-bottom: 2px solid #ffb74d; padding-bottom: 10px; }
        h2 { border-bottom: 1px solid #e0e0e0; padding-bottom: 5px; }
        img { 
            width: 100%; 
            max-width: 100%; 
            height: auto; 
            display: block;
            margin: 10px 0;
        }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4b225c; color: white; }
        """
        
        # Write CSS to a temp file for md2pdf
        import tempfile
        css_file = tempfile.NamedTemporaryFile(mode='w', suffix='.css', delete=False)
        css_file.write(css_content)
        css_file.close()
        
        try:
            # Convert to PDF with CSS styling
            md2pdf(
                pdf_path,
                md_content=md_content,
                md_file_path=md_path,
                css_file_path=css_file.name,
                base_url=md_dir
            )
        finally:
            # Clean up temp CSS file
            os.unlink(css_file.name)
        
        print(f"[DEBUG] PDF report saved to: {pdf_path}")
        return pdf_path
        
    except ImportError:
        print("[WARNING] md2pdf package not installed. Run: pip install md2pdf")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to convert MD to PDF: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Server-Side Session Storage ---
# WARNING: In-memory storage is not persistent. Use Redis for production.
SESSIONS = {}

# Left panel chat messages (user questions + final results)
LEFT_CHAT_MESSAGES = {}  # session_id -> list of chat messages

# Processing status
PROCESSING_STATUS = {}  # session_id -> {"status": "idle/processing/complete", "task_id": None}

# Step-wise process details for right panel
PROCESS_DETAILS = {}  # session_id -> list of process steps

# Track individual step collapse states to preserve user interactions
STEP_STATES = {}  # session_id -> {step_index: {"is_open": bool, "user_interacted": bool}}

# Track background threads and cancellation tokens for stopping
BACKGROUND_THREADS = {}  # session_id -> {"thread": thread_object, "stop_event": threading.Event(), "cancellation_token": CancellationToken}

# Session state for incremental conversation (simplified from HITL)
SESSION_STATE = {}  # session_id -> {"adapter": adapter, "can_continue": bool, "loop": event_loop, "history": []}

# Dedicated event loops per session to avoid "attached to different loop" errors
SESSION_LOOPS = {}  # session_id -> {"loop": asyncio.AbstractEventLoop, "thread": threading.Thread}

# --- Backward Compatibility Placeholders ---
# These are maintained for compatibility with other UI components (like page2.py)
# They are not used in the main dash_UI.py but may be referenced by legacy code
PENDING_USER_INPUTS = {}  # Placeholder for backward compatibility
MESSAGE_QUEUES = {}       # Placeholder for backward compatibility  
FINAL_RESULTS = {}        # Placeholder for backward compatibility
COLLAPSE_STATE = {}       # Placeholder for backward compatibility


# --- Session Helper Functions ---
def _mark_can_continue(session_id: str):
    """Mark that the session can continue with a follow-up query."""
    if session_id in SESSION_STATE:
        SESSION_STATE[session_id]["can_continue"] = True


def _on_final_result(session_id: str, content: str):
    """
    Handle final result: add to chat and mark session as ready for continuation.
    """
    append_to_final_result(session_id, content)
    _mark_can_continue(session_id)
    print(f"[HITL] Session {session_id} ready for continuation")


# --- Helper Function to Create an Autogen Team ---
def create_autogen_team(session_id, use_simple_team=False, team_type='magentic'):
    """
    Initializes and returns a new agent system with session-specific storage.
    
    Args:
        session_id: Unique session identifier
        use_simple_team: If True, use simple model (legacy parameter)
        team_type: 'simple', 'magentic', or 'langgraph'
        
    Returns:
        dict with system and metadata
    """
    print(f"Creating a new biomedical agent system (team_type={team_type}) for session: {session_id}")
    
    # Handle langgraph team type
    if team_type == 'langgraph':
        # Create LangGraph adapter with UI callbacks configured
        adapter = create_langgraph_adapter(
            session_id=session_id,
            model_name="gemini-3-pro-preview"
        )
        
        # Set up callbacks for UI updates
        adapter.set_callbacks(
            process_step=add_process_step,
            left_chat=add_left_chat_message,
            multimedia=add_multimedia_process_step,
            final_result=lambda sid, content: _on_final_result(sid, content)
        )
        
        # Store adapter reference for continuation support with history
        SESSION_STATE[session_id] = {
            "adapter": adapter,
            "can_continue": False,  # Set to True after report is generated
            "history": []  # Track conversation history for continuation
        }
        
        print(f"[DEBUG] Created LangGraph adapter for session {session_id}")
        
        return {
            "system": adapter,
            "adapter": adapter,  # Reference to adapter for type checking
            "conversation_started": False,
            "task_result": None,
            "team_type": "langgraph",
            "session_id": session_id
        }
    
    # Legacy handling for simple/magentic
    if use_simple_team or team_type == 'simple':
        # For simple mode, use a basic model configuration
        model_name = "gpt-4o-mini"
        print("[DEBUG] Created simple agent configuration")
    else:
        # For full mode, use the advanced model
        model_name = "gemini-2.5-pro"
        print("[DEBUG] Created full biomedical research team configuration")
    
    # Create the UI-enabled agent system with session_id for file storage
    system = UIMagneticAgentSystem(
        model_name=model_name,
        temperature=0.0,
        log_dir="./logs",
        session_id=session_id  # Pass session_id to store outputs in sessions/<session_id>/
    )
    print(f"[DEBUG] Successfully created UIMagneticAgentSystem with {model_name} for session {session_id}")
    
    return {
        "system": system, 
        "conversation_started": False,
        "task_result": None,
        "team_type": "simple" if use_simple_team else "magentic",
        "session_id": session_id
    }


# --- Dash App Initialization ---
# Configure assets folder to use the parent webapp directory
from flask import send_from_directory, Response

assets_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')

# Sessions directory - we'll serve this as a static folder
sessions_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sessions')

app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP], 
    assets_folder=assets_folder
)

# Get the Flask server for adding custom routes
server = app.server

# Add sessions folder as a static folder for Flask
# This allows serving files via /sessions/<filepath>
server.static_folder = None  # Clear default
server.static_url_path = None

# Note: SESSIONS_DIR is defined at the top of the file
print(f"[STARTUP] SESSIONS_DIR configured as: {SESSIONS_DIR}")
print(f"[STARTUP] SESSIONS_DIR exists: {os.path.exists(SESSIONS_DIR)}")
print(f"[STARTUP] sessions_folder: {sessions_folder}")

# Register a blueprint for serving session files
from flask import Blueprint, send_from_directory as flask_send

sessions_bp = Blueprint('sessions', __name__, static_folder=sessions_folder, static_url_path='/sessions')

@sessions_bp.route('/<path:filepath>')
def serve_session_files(filepath):
    """Serve files from the sessions directory."""
    print(f"[SESSIONS BP] Serving: {filepath}")
    return flask_send(sessions_folder, filepath)

server.register_blueprint(sessions_bp, url_prefix='/sessions')

# Custom CSS for the scholarly, mountain-inspired look
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: "Times New Roman", Times, serif;
                color: #4b225c;
                background: #fefefe;
                margin: 0;
                padding: 0;
            }
            .main-title {
                color: #ffb74d;
                font-size: 30px;
                font-weight: 600;
                font-family: "Times New Roman", serif;
                text-align: left;
                margin: 20px 0 15px 0;
            }
            .title-links {
                color: #4b225c;
                text-decoration: none;
                font-size: 22px;
                font-style: italic;
                font-family: "Times New Roman", serif;
                float: right;
                margin-top: 20px;
            }
            .title-links a {
                color: #4b225c;
                text-decoration: none;
            }
            .title-links a:hover {
                text-decoration: underline;
            }
            .results-panel {
                background: linear-gradient(145deg, #fefefe 0%, #f8f8f8 100%);
                border: 1px solid #e0e0e0;
                border-radius: 0;
                padding: 15px;
                min-height: 450px;
                max-height: 450px;
                overflow-y: auto;
                box-shadow: 0 2px 8px rgba(75, 34, 92, 0.08);
                font-family: "Times New Roman", Times, serif;
            }
            .chat-message {
                margin-bottom: 10px;
                padding: 8px 15px;
                border-radius: 0;
                max-width: 100%;
                word-wrap: break-word;
                font-family: "Times New Roman", Times, serif;
                line-height: 1.4;
            }
            .chat-message.user {
                background: linear-gradient(145deg, #e8eaf6 0%, #c5cae9 100%);
                border-left: 4px solid #4b225c;
                margin-left: 0;
                margin-right: 0;
            }
            .chat-message.assistant {
                background: linear-gradient(145deg, #f3e5f5 0%, #e1bee7 100%);
                border-left: 4px solid #7e57c2;
                margin-left: 0;
                margin-right: 0;
            }
            .chat-message.system {
                background: linear-gradient(145deg, #fff3e0 0%, #ffcc80 100%);
                border-left: 4px solid #ff9800;
                margin-left: 0;
                margin-right: 0;
                font-style: italic;
            }
            .chat-sender {
                font-weight: bold;
                color: #3a1a47;
                margin-bottom: 4px;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .chat-content {
                color: #4b225c;
                font-size: 14px;
            }
            .chat-timestamp {
                font-size: 11px;
                color: #8e8e8e;
                text-align: right;
                margin-top: 4px;
                font-style: italic;
            }
            .process-panel {
                background: linear-gradient(135deg, #f3e5f5 0%, #e3f2fd 45%, #ffe8d5 100%);
                border: 1px solid #e0e0e0;
                border-radius: 0;
                padding: 20px;
                min-height: 450px;
                max-height: 450px;
                overflow-y: auto;
                box-shadow: 0 2px 8px rgba(75, 34, 92, 0.08);
                font-family: "Times New Roman", Times, serif;
            }
            .step-card {
                border: 1px solid #d0d0d0;
                border-radius: 0;
                margin-bottom: 12px;
                background: rgba(255, 255, 255, 0.85);
                box-shadow: 0 1px 4px rgba(75, 34, 92, 0.06);
                backdrop-filter: blur(5px);
            }
            .step-header {
                background: rgba(255, 255, 255, 0.9);
                padding: 12px 18px;
                border-bottom: 1px solid #d0d0d0;
                cursor: pointer;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .step-content {
                padding: 18px;
                display: none;
            }
            .step-content.expanded {
                display: block;
            }
            .markdown-content {
                line-height: 1.8;
                font-family: "Times New Roman", Times, serif;
                color: #4b225c;
                font-size: 14px;
            }
            .markdown-content h1, .markdown-content h2, .markdown-content h3 {
                color: #3a1a47;
                margin-top: 24px;
                margin-bottom: 12px;
                font-family: "Times New Roman", Times, serif;
                font-weight: 500;
            }
            .markdown-content h1 {
                font-size: 18px;
            }
            .markdown-content h2 {
                font-size: 16px;
            }
            .markdown-content h3 {
                font-size: 15px;
            }
            .markdown-content h4 {
                font-size: 14.5px;
                color: #3a1a47;
                margin-top: 20px;
                margin-bottom: 10px;
                font-family: "Times New Roman", Times, serif;
                font-weight: 500;
            }
            .markdown-content h5 {
                font-size: 14px;
                color: #3a1a47;
                margin-top: 16px;
                margin-bottom: 8px;
                font-family: "Times New Roman", Times, serif;
                font-weight: 500;
            }
            .markdown-content h6 {
                font-size: 13.5px;
                color: #3a1a47;
                margin-top: 12px;
                margin-bottom: 6px;
                font-family: "Times New Roman", Times, serif;
                font-weight: 500;
            }
            .input-section {
                background: linear-gradient(145deg, #fefefe 0%, #f8f8f8 100%);
                padding: 20px;
                border-radius: 0;
                border: 1px solid #e0e0e0;
                margin-top: 20px;
                box-shadow: 0 2px 8px rgba(75, 34, 92, 0.08);
            }
            .team-config {
                background: rgba(255, 255, 255, 0.9);
                padding: 15px;
                border: 1px solid #e0e0e0;
                border-radius: 0;
                margin-bottom: 10px;
            }
            .btn {
                border-radius: 0 !important;
                font-family: "Times New Roman", Times, serif;
                font-weight: 500;
                letter-spacing: 0.3px;
                text-transform: none;
                border: none;
                transition: all 0.3s ease;
            }
            .btn-primary {
                background: linear-gradient(145deg, #4b225c 0%, #3a2a47 100%);
                color: #fefefe;
                box-shadow: 0 2px 6px rgba(75, 34, 92, 0.2), inset 0 1px 0 rgba(139, 123, 184, 0.3);
                border: 1px solid rgba(75, 34, 92, 0.8);
            }
            .btn-primary:hover {
                background: linear-gradient(145deg, #5b326c 0%, #4a2a57 100%);
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(75, 34, 92, 0.3), inset 0 1px 0 rgba(159, 143, 204, 0.4);
                border: 1px solid rgba(91, 50, 108, 0.9);
            }
            .btn-secondary {
                background: linear-gradient(145deg, #7e57c2 0%, #673ab7 100%);
                color: #fefefe;
                box-shadow: 0 2px 6px rgba(126, 87, 194, 0.2), inset 0 1px 0 rgba(159, 143, 204, 0.3);
                border: 1px solid rgba(126, 87, 194, 0.8);
            }
            .btn-secondary:hover {
                background: linear-gradient(145deg, #8e67d2 0%, #7749c7 100%);
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(126, 87, 194, 0.3), inset 0 1px 0 rgba(179, 163, 224, 0.4);
                border: 1px solid rgba(142, 103, 210, 0.9);
            }
            .btn-light {
                background: linear-gradient(145deg, #fefefe 0%, #f0f0f0 100%);
                color: #4b225c;
                border: 1px solid #e0e0e0;
                box-shadow: 0 1px 3px rgba(75, 34, 92, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.7);
            }
            .btn-light:hover {
                background: linear-gradient(145deg, #f0f0f0 0%, #e8e8e8 100%);
                color: #3a1d47;
                transform: translateY(-1px);
                box-shadow: 0 3px 8px rgba(75, 34, 92, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.8);
                border: 1px solid #d0d0d0;
            }
            .btn-danger {
                background: linear-gradient(145deg, #dc3545 0%, #c82333 100%);
                color: #ffffff;
                box-shadow: 0 2px 6px rgba(220, 53, 69, 0.2), inset 0 1px 0 rgba(240, 140, 150, 0.3);
                border: 1px solid rgba(220, 53, 69, 0.8);
            }
            .btn-danger:hover {
                background: linear-gradient(145deg, #e04555 0%, #d23343 100%);
                color: #ffffff;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(220, 53, 69, 0.3), inset 0 1px 0 rgba(250, 160, 170, 0.4);
                border: 1px solid rgba(224, 69, 85, 0.9);
            }
            textarea {
                border-radius: 0 !important;
                border: 1px solid #e0e0e0 !important;
                font-family: "Times New Roman", Times, serif !important;
                color: #4b225c !important;
                background: linear-gradient(145deg, #fefefe 0%, #f8f8f8 100%) !important;
                box-shadow: inset 0 1px 3px rgba(75, 34, 92, 0.05) !important;
            }
            textarea:focus {
                border-color: #4b225c !important;
                box-shadow: 0 0 0 2px rgba(75, 34, 92, 0.1) !important;
            }
            label {
                font-family: "Times New Roman", Times, serif;
                color: #4b225c;
                font-weight: 500;
            }
            .form-check-label {
                font-family: "Times New Roman", Times, serif;
                color: #4b225c;
                margin-left: 8px;
            }
            h4 {
                font-family: "Times New Roman", Times, serif;
                color: #4b225c;
                font-weight: 500;
                letter-spacing: 0.2px;
                font-size: 12px;
            }
            h5 {
                font-family: "Times New Roman", Times, serif;
                color: #4b225c;
                font-weight: 500;
            }
            /* Custom scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
            }
            ::-webkit-scrollbar-track {
                background: rgba(255, 255, 255, 0.5);
            }
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(145deg, #7e57c2 0%, #673ab7 100%);
                border-radius: 0;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: linear-gradient(145deg, #8e67d2 0%, #7749c7 100%);
            }
            /* Radio button styling */
            .form-check-input {
                border-radius: 0;
                border: 1px solid #e0e0e0;
            }
            .form-check-input:checked {
                background-color: #4b225c;
                border-color: #4b225c;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = dbc.Container([
    dcc.Store(id='session-id', storage_type='session'),
    dcc.Interval(id='message-updater', interval=1000, n_intervals=0),
    
    # Title Section with Icon
    html.Div([
        html.Div([
            html.Img(src='/assets/dash-logo-stripe.svg', 
                     style={'height': '60px', 'width': 'auto', 'margin-right': '20px', 'vertical-align': 'middle'}),
            html.Div([
                html.H1("agent.omni-cell.com", 
                        style={'color': '#ffb74d', 'font-size': '36px', 'font-weight': '600', 'font-family': 'Times New Roman, serif', 'margin': '0'}),
                html.H2("Multi-Agentic System for Bio-medical Research", 
                        style={'color': '#4b225c', 'font-size': '24px', 'font-weight': '400', 'font-family': 'Times New Roman, serif', 'margin': '5px 0 0 0'}),
                html.H3("Powered by Advanced AI", 
                        style={'color': '#7e57c2', 'font-size': '18px', 'font-weight': '300', 'font-family': 'Times New Roman, serif', 'margin': '5px 0 0 0', 'font-style': 'italic'})
            ], style={'display': 'inline-block', 'vertical-align': 'middle'})
        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})
    ], style={'text-align': 'center', 'margin': '20px 0 30px 0'}),

    # Main Content Area
    dbc.Row([
        # Left Panel - Final Results (equal width)
        dbc.Col([
            html.Div([
                html.H4("📝 Final Results", style={'margin-bottom': '15px', 'color': '#4b225c'}),
                html.Div(id='final-results-content', children=[
                    html.P("Results will appear here after processing...", 
                          style={'color': '#6c757d', 'font-style': 'italic'})
                ])
            ], className='results-panel')
        ], width=6),  # Changed back from 4 to 6
        
        # Right Panel - Step-wise Process Details (equal width)
        dbc.Col([
            html.Div([
                html.Div([
                    html.H4("🔧 Step-wise Process Details", style={'margin-bottom': '15px', 'color': '#4b225c', 'display': 'inline-block'}),
                    html.Div([
                        dbc.Button("Expand All", id='expand-all-button', n_clicks=0, 
                                  color='light', size='sm', 
                                  style={'margin-right': '5px', 'border-radius': '0', 'font-size': '0.8rem'}),
                        dbc.Button("Collapse All", id='collapse-all-button', n_clicks=0, 
                                  color='light', size='sm', 
                                  style={'border-radius': '0', 'font-size': '0.8rem'})
                    ], style={'float': 'right', 'margin-top': '10px'})
                ], style={'overflow': 'hidden', 'margin-bottom': '15px'}),
                html.Div(id='process-details-content', children=[
                    html.P("Process steps will appear here during execution...", 
                          style={'color': '#6c757d', 'font-style': 'italic'})
                ])
            ], className='process-panel')
        ], width=6)  # Changed back from 8 to 6
    ], style={'margin-bottom': '20px'}),
    
    # Input Section with Team Config
    html.Div([
        dbc.Row([
            # Team Configuration (moved to bottom left)
            dbc.Col([
                html.Div([
                    html.Label("Team Configuration:", style={'font-weight': 'bold', 'margin-bottom': '8px', 'font-size': '0.9rem'}),
                    dcc.RadioItems(
                        id='team-type-selector',
                        options=[
                            # {'label': ' Simple Assistant', 'value': 'simple'},
                            {'label': ' AutoGen Research Team', 'value': 'magentic'},
                            {'label': ' LangGraph Research Team', 'value': 'langgraph'}
                        ],
                        value='langgraph',
                        inline=False,
                        style={'font-size': '0.85rem'}
                    )
                ], className='team-config')
            ], width=3),
            
            # Text Input
            dbc.Col([
                dcc.Textarea(
                    id='user-input',
                    placeholder='Enter your bio-medical research question or task...',
                    style={
                        'width': '100%', 
                        'height': '120px',  # Height of 3 buttons
                        'resize': 'vertical',
                        'border': '1px solid #e0e0e0',
                        'border-radius': '0',
                        'padding': '10px',
                        'font-family': 'Times New Roman, Times, serif',
                        'color': '#4b225c'
                    }
                )
            ], width=5),
            
            # Action Buttons (primary)
            dbc.Col([
                dbc.Button('Send', id='send-button', n_clicks=0, 
                          color='primary', size='lg', 
                          style={'width': '100%', 'height': '40px', 'margin-bottom': '5px', 'border-radius': '0'}),
                dbc.Button('Stop', id='stop-button', n_clicks=0, 
                          color='danger', size='sm', 
                          style={'width': '100%', 'height': '35px', 'margin-bottom': '5px', 'border-radius': '0'}),
                dbc.Button('New Session', id='new-chat-button', n_clicks=0, 
                          color='secondary', size='sm', 
                          style={'width': '100%', 'height': '35px', 'margin-bottom': '5px', 'border-radius': '0'}),
                dbc.Button('📥 Download', id='download-session-button', n_clicks=0, 
                          color='success', size='sm', 
                          style={'width': '100%', 'height': '35px', 'border-radius': '0'}),
                dcc.Download(id='download-session-zip')
            ], width=2),
            
            # Continue Button (appears after report completes)
            dbc.Col([
                dbc.Button('🔄 Continue Chat', id='hitl-continue-button', n_clicks=0, 
                          color='info', size='lg', 
                          style={'width': '100%', 'height': '120px', 'border-radius': '0', 'font-size': '1rem'},
                          title='Continue conversation with follow-up question')
            ], id='hitl-controls-col', width=2, style={'display': 'none'})
        ])
    ], className='input-section'),
    
    # Footer with links
    html.Div([
        html.A("[lab]", 
               href="https://fuhailiailab.github.io", 
               target="_blank",
               style={'color': '#4b225c', 'text-decoration': 'none', 'font-family': 'Times New Roman, serif', 'font-size': '16px', 'margin-right': '15px'}),
        html.A("[github]", 
               href="https://github.com/fuhailiailab", 
               target="_blank",
               style={'color': '#4b225c', 'text-decoration': 'none', 'font-family': 'Times New Roman, serif', 'font-size': '16px', 'margin-right': '15px'}),
        html.A("[paper]", 
               href="https://www.biorxiv.org/content/10.1101/2025.07.31.667797v1", 
               target="_blank",
               style={'color': '#4b225c', 'text-decoration': 'none', 'font-family': 'Times New Roman, serif', 'font-size': '16px'})
    ], style={'text-align': 'center', 'margin': '30px 0 20px 0', 'padding': '20px 0', 'border-top': '1px solid #e0e0e0'})
    
], fluid=True, style={'max-width': '1400px'})

# --- Process Management Functions ---
def add_process_step(session_id, agent_name, content, step_number=None, is_tool_call=False):
    """Add a process step for the right panel."""
    if session_id not in PROCESS_DETAILS:
        PROCESS_DETAILS[session_id] = []
    
    if session_id not in STEP_STATES:
        STEP_STATES[session_id] = {}
    
    step = {
        "agent": agent_name,
        "content": content,  # Store raw content, will render with dcc.Markdown
        "timestamp": time.time(),
        "step_number": step_number or len(PROCESS_DETAILS[session_id]) + 1,
        "is_tool_call": is_tool_call
    }
    
    # Store the number of existing steps before adding the new one
    existing_step_count = len(PROCESS_DETAILS[session_id])
    
    PROCESS_DETAILS[session_id].append(step)
    
    # If this is the first step, just open it
    if existing_step_count == 0:
        new_step_index = f"{session_id}-0"
        STEP_STATES[session_id][new_step_index] = {
            "is_open": True,
            "user_interacted": False
        }
        return
    
    # For existing steps: DO NOT auto-collapse any steps that user has interacted with
    # This preserves exactly what the user is viewing
    for i in range(existing_step_count):
        existing_step_index = f"{session_id}-{i}"
        
        # Only initialize state for steps that don't have state yet
        # Do NOT modify existing states - this preserves user's current view
        if existing_step_index not in STEP_STATES[session_id]:
            # New step that hasn't been tracked yet - default to closed
            STEP_STATES[session_id][existing_step_index] = {
                "is_open": False,
                "user_interacted": False
            }
        # For existing tracked steps: DO NOTHING - preserve their exact state
    
    # Initialize the new (latest) step as open
    new_step_index = f"{session_id}-{existing_step_count}"
    STEP_STATES[session_id][new_step_index] = {
        "is_open": True,  # Always expand the latest step
        "user_interacted": False
    }

def add_left_chat_message(session_id, sender, content, message_type="user"):
    """Add a message to the left panel chat."""
    if session_id not in LEFT_CHAT_MESSAGES:
        LEFT_CHAT_MESSAGES[session_id] = []
    
    message = {
        "sender": sender,
        "content": content,
        "type": message_type,  # "user", "assistant", "system"
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }
    LEFT_CHAT_MESSAGES[session_id].append(message)

def set_user_question(session_id, question):
    """Set the user's initial question for display in the left panel."""
    add_left_chat_message(session_id, "You", question, "user")

def append_to_final_result(session_id, additional_content):
    """Append content to left panel as assistant response."""
    add_left_chat_message(session_id, "Assistant", additional_content, "assistant")

def add_multimedia_message(session_id, sender, content, media_type="text", media_url=None, message_type="assistant"):
    """Add a multimedia message to the left panel chat.
    
    Args:
        session_id: Session identifier
        sender: Who sent the message
        content: Text content of the message
        media_type: Type of media ('image', 'html', 'text')
        media_url: URL or path to the media file
        message_type: Message type ("user", "assistant", "system")
    """
    if session_id not in LEFT_CHAT_MESSAGES:
        LEFT_CHAT_MESSAGES[session_id] = []
    
    message = {
        "sender": sender,
        "content": content,
        "type": message_type,
        "media_type": media_type,
        "media_url": media_url,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }
    LEFT_CHAT_MESSAGES[session_id].append(message)

def add_multimedia_process_step(session_id, agent_name, content, media_type="text", media_url=None, step_number=None):
    """Add a multimedia process step for the right panel.
    
    Args:
        session_id: Session identifier
        agent_name: Name of the agent
        content: Text content of the step
        media_type: Type of media ('image', 'html', 'text')
        media_url: URL or path to the media file
        step_number: Step number (optional)
    """
    if session_id not in PROCESS_DETAILS:
        PROCESS_DETAILS[session_id] = []
    
    if session_id not in STEP_STATES:
        STEP_STATES[session_id] = {}
    
    step = {
        "agent": agent_name,
        "content": content,
        "media_type": media_type,
        "media_url": media_url,
        "timestamp": time.time(),
        "step_number": step_number or len(PROCESS_DETAILS[session_id]) + 1
    }
    
    # Store the number of existing steps before adding the new one
    existing_step_count = len(PROCESS_DETAILS[session_id])
    
    PROCESS_DETAILS[session_id].append(step)
    
    # If this is the first step, just open it
    if existing_step_count == 0:
        new_step_index = f"{session_id}-0"
        STEP_STATES[session_id][new_step_index] = {
            "is_open": True,
            "user_interacted": False
        }
        return
    
    # For existing steps: DO NOT auto-collapse any steps that user has interacted with
    # This preserves exactly what the user is viewing
    for i in range(existing_step_count):
        existing_step_index = f"{session_id}-{i}"
        
        # Only initialize state for steps that don't have state yet
        # Do NOT modify existing states - this preserves user's current view
        if existing_step_index not in STEP_STATES[session_id]:
            # New step that hasn't been tracked yet - default to closed
            STEP_STATES[session_id][existing_step_index] = {
                "is_open": False,
                "user_interacted": False
            }
        # For existing tracked steps: DO NOTHING - preserve their exact state
    
    # Initialize the new (latest) step as open
    new_step_index = f"{session_id}-{existing_step_count}"
    STEP_STATES[session_id][new_step_index] = {
        "is_open": True,  # Always expand the latest step
        "user_interacted": False
    }

def get_left_chat_display(session_id):
    """Get the left panel chat messages as HTML elements with multimedia support."""
    if session_id not in LEFT_CHAT_MESSAGES or not LEFT_CHAT_MESSAGES[session_id]:
        return [html.P("Your conversation will appear here...", 
                      style={'color': '#6c757d', 'font-style': 'italic', 'text-align': 'center', 'margin-top': '50px'})]
    
    chat_elements = []
    for msg in LEFT_CHAT_MESSAGES[session_id]:
        # Start with text content
        content_elements = []
        
        if msg["content"]:
            content_elements.append(
                dcc.Markdown(
                    children=msg["content"],
                    className='markdown-content'
                )
            )
        
        # Add multimedia content if present
        media_type = msg.get("media_type", "text")
        media_url = msg.get("media_url")
        
        if media_type == "image" and media_url:
            content_elements.append(
                html.Img(
                    src=media_url,
                    style={
                        'max-width': '100%',
                        'height': 'auto',
                        'border-radius': '8px',
                        'margin-top': '10px',
                        'box-shadow': '0 2px 8px rgba(0,0,0,0.1)'
                    }
                )
            )
        elif media_type == "html" and media_url:
            # For HTML files, create an iframe
            content_elements.append(
                html.Iframe(
                    src=media_url,
                    style={
                        'width': '100%',
                        'height': '400px',
                        'border': '1px solid #e0e0e0',
                        'border-radius': '8px',
                        'margin-top': '10px'
                    }
                )
            )
        
        chat_bubble = html.Div([
            html.Div(msg["sender"], className="chat-sender"),
            html.Div(content_elements, className="chat-content"),
            html.Div(msg["timestamp"], className="chat-timestamp")
        ], className=f"chat-message {msg['type']}")
        
        chat_elements.append(chat_bubble)
    
    return chat_elements

# --- Backward Compatibility Functions ---
# These functions are maintained for compatibility with other UI components
# They are deprecated and should not be used in new code

def add_message_to_queue(session_id, message_content, sender="Team", style=None, message_type="process"):
    """DEPRECATED: Add a message to the session's message queue for backward compatibility."""
    if session_id not in MESSAGE_QUEUES:
        MESSAGE_QUEUES[session_id] = []
    
    message = {
        "content": message_content,
        "sender": sender,
        "style": style or {},
        "timestamp": time.time(),
        "type": message_type
    }
    MESSAGE_QUEUES[session_id].append(message)
    print(f"[DEBUG] Added message to queue for session {session_id}: {sender}: {message_content[:50]}...")

def set_final_result(session_id, result_content):
    """DEPRECATED: Set the final result for backward compatibility."""
    FINAL_RESULTS[session_id] = result_content

def get_final_results_display(session_id):
    """DEPRECATED: Get the final results for backward compatibility."""
    return get_left_chat_display(session_id)

def get_process_details_display(session_id):
    """Get the process details as collapsible HTML elements for the right panel."""
    if session_id not in PROCESS_DETAILS or not PROCESS_DETAILS[session_id]:
        return [html.P("Process steps will appear here during execution...", 
                      style={'color': '#6c757d', 'font-style': 'italic'})]
    
    # Debug: print all steps with media
    for i, step in enumerate(PROCESS_DETAILS[session_id]):
        if step.get('media_type') in ['image', 'html'] and step.get('media_url'):
            print(f"[DEBUG RENDER] Step {i}: media_type={step.get('media_type')}, media_url={step.get('media_url')}")
    
    if session_id not in STEP_STATES:
        STEP_STATES[session_id] = {}
    
    steps = []
    for i, step in enumerate(PROCESS_DETAILS[session_id]):
        step_index = f"{session_id}-{i}"
        is_latest = i == len(PROCESS_DETAILS[session_id]) - 1
        is_tool_call = step.get('is_tool_call', False)
        
        # Get the current state or initialize it
        if step_index not in STEP_STATES[session_id]:
            # Initialize state - collapse tool call steps by default, expand others only if latest
            should_be_open = (not is_tool_call) and is_latest
            STEP_STATES[session_id][step_index] = {
                "is_open": should_be_open,
                "user_interacted": False
            }
        
        current_state = STEP_STATES[session_id][step_index]
        
        # Use the preserved state (don't override auto-collapse behavior)
        is_open = current_state["is_open"]
        
        step_card = dbc.Card([
            dbc.CardHeader([
                dbc.Button([
                    html.Span("▼ " if is_open else "▶ ", 
                             id={'type': 'icon', 'index': step_index}, 
                             style={'margin-right': '5px'}),
                    html.Span(f"{step['agent']}", style={'font-weight': 'bold'}),
                    html.Span(f" - Step {step['step_number']}", style={'color': '#6c757d', 'font-size': '0.9em', 'margin-left': '10px'})
                ], 
                id={'type': 'toggle', 'index': step_index},
                color="light",
                style={
                    'width': '100%', 
                    'text-align': 'left', 
                    'border': 'none',
                    'background': 'transparent',
                    'padding': '10px 15px'
                })
            ], 
            style={'padding': '0', 'background': '#f1f3f4', 'border-bottom': '1px solid #e3e6ea'}),
            
            dbc.Collapse([
                dbc.CardBody([
                    # Start with text content
                    dcc.Markdown(
                        children=step['content'],
                        className='markdown-content',
                        style={'margin': '0'}
                    ),
                    # Add multimedia content if present
                    html.Div([
                        # Debug: show the URL as a link
                        html.A(
                            f"Open {step.get('media_type', 'file')}: {step.get('media_url', 'N/A')}",
                            href=step.get('media_url', '#'),
                            target="_blank",
                            style={
                                'display': 'block' if step.get('media_url') else 'none',
                                'margin-bottom': '10px',
                                'color': '#007bff'
                            }
                        ) if step.get('media_url') else None,
                        # Image content
                        html.Img(
                            src=step.get('media_url', ''),
                            style={
                                'max-width': '100%',
                                'height': 'auto',
                                'border-radius': '8px',
                                'margin-top': '10px',
                                'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                                'display': 'block' if step.get('media_type') == 'image' and step.get('media_url') else 'none'
                            }
                        ) if step.get('media_type') == 'image' and step.get('media_url') else None,
                        # HTML iframe content
                        html.Iframe(
                            src=step.get('media_url', ''),
                            style={
                                'width': '100%',
                                'height': '600px',
                                'border': '1px solid #e0e0e0',
                                'border-radius': '8px',
                                'margin-top': '10px',
                                'display': 'block' if step.get('media_type') == 'html' and step.get('media_url') else 'none'
                            }
                        ) if step.get('media_type') == 'html' and step.get('media_url') else None
                    ], style={'margin-top': '10px'}) if step.get('media_type') in ['image', 'html'] and step.get('media_url') else None
                ], style={'padding': '15px'})
            ], 
            id={'type': 'collapse', 'index': step_index},
            is_open=is_open)  # Use preserved state
        ], style={'margin-bottom': '10px', 'border': '1px solid #e3e6ea'})
        
        steps.append(step_card)
    
    return steps

def start_processing_in_background(session_id, user_input, team_type):
    """Start agent processing in a background thread with proper cancellation support.
    
    Supports both AutoGen-based (magentic/simple) and LangGraph-based agents.
    Uses a dedicated event loop per session to avoid "attached to different loop" errors.
    
    Args:
        session_id: Unique session identifier
        user_input: The user's query/task
        team_type: 'simple', 'magentic', or 'langgraph'
    """
    
    # Check if there's already processing happening - stop it first
    if session_id in BACKGROUND_THREADS:
        print(f"[DEBUG] Stopping existing processing for session {session_id} before starting new one")
        stop_processing(session_id)
        # Give it a moment to clean up
        time.sleep(0.3)
    
    def process_agent_response():
        try:
            print(f"[DEBUG] Starting background processing for session {session_id}, team_type={team_type}")
            PROCESSING_STATUS[session_id] = {"status": "processing", "task_id": str(uuid.uuid4())}
            
            # Get the stop event for this session
            stop_event = BACKGROUND_THREADS[session_id]["stop_event"]
            
            # Check if stopped before starting
            if stop_event.is_set():
                add_process_step(session_id, "System", "🛑 Processing cancelled before starting")
                return
            
            # Get or create session
            if session_id not in SESSIONS or SESSIONS[session_id] is None:
                print(f"[DEBUG] Creating new agent system for session {session_id}, team_type={team_type}")
                SESSIONS[session_id] = create_autogen_team(session_id, team_type=team_type)
            
            if stop_event.is_set():
                return
            
            # Get session data
            session_data = SESSIONS[session_id]
            
            if isinstance(session_data, dict) and 'system' in session_data:
                agent_system = session_data['system']
                actual_team_type = session_data.get('team_type', 'magentic')
                
                # Create a new event loop for this thread - this is critical
                # Each thread needs its own event loop for asyncio to work properly
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Store the loop in session state for potential reuse
                SESSION_LOOPS[session_id] = {"loop": loop, "thread": threading.current_thread()}
                
                try:
                    async def run_with_cancellation():
                        """Run the agent system with proper cancellation monitoring."""
                        # Check for cancellation before starting
                        if stop_event.is_set():
                            raise asyncio.CancelledError("Processing stopped by user")
                        
                        # Store the current task for cancellation
                        current_task = asyncio.current_task()
                        BACKGROUND_THREADS[session_id]["current_task"] = current_task
                        
                        try:
                            # Different execution paths for different agent types
                            if actual_team_type == 'langgraph':
                                # LangGraph adapter - check if this is a continuation
                                print(f"[DEBUG] Running LangGraph adapter for session {session_id}")
                                
                                # Check if we have previous state to continue from
                                session_info = SESSION_STATE.get(session_id, {})
                                adapter = session_info.get("adapter")
                                history = session_info.get("history", [])
                                
                                if adapter and history:
                                    # This is a continuation - use continue_conversation
                                    print(f"[DEBUG] Continuing conversation with {len(history)} previous rounds")
                                    add_process_step(session_id, "System", f"🔄 Continuing from previous analysis ({len(history)} rounds)")
                                    await adapter.continue_conversation(user_input, stop_event=stop_event)
                                else:
                                    # First run - use run_stream
                                    await agent_system.run_stream(user_input, stop_event=stop_event)
                                
                                # Store query in history for future continuation
                                if session_id in SESSION_STATE:
                                    if "history" not in SESSION_STATE[session_id]:
                                        SESSION_STATE[session_id]["history"] = []
                                    SESSION_STATE[session_id]["history"].append({
                                        "query": user_input,
                                        "timestamp": datetime.now().isoformat()
                                    })
                            else:
                                # AutoGen-based system - use run_stream_for_ui
                                print(f"[DEBUG] Running AutoGen system for session {session_id}")
                                await agent_system.run_stream_for_ui(user_input, session_id, stop_event=stop_event)
                        except asyncio.CancelledError:
                            print(f"[DEBUG] Agent system cancelled for session {session_id}")
                            raise
                        except Exception as e:
                            if stop_event.is_set():
                                print(f"[DEBUG] Agent system stopped during execution for session {session_id}")
                                raise asyncio.CancelledError("Processing stopped by user")
                            else:
                                raise e
                    
                    # Run with global timeout and cancellation check
                    try:
                        loop.run_until_complete(asyncio.wait_for(run_with_cancellation(), timeout=PROCESSING_TIMEOUT))
                    except asyncio.TimeoutError:
                        add_process_step(session_id, "System", f"⏰ Processing timed out after {PROCESSING_TIMEOUT//60} minutes")
                        raise asyncio.CancelledError("Processing timed out")
                    except asyncio.CancelledError:
                        add_process_step(session_id, "System", "🛑 Processing cancelled by user")
                        return
                    
                    # Check if stopped after completion
                    if stop_event.is_set():
                        add_process_step(session_id, "System", "🛑 Processing stopped after completion")
                        return
                        
                except Exception as e:
                    if stop_event.is_set():
                        add_process_step(session_id, "System", "🛑 Processing stopped")
                        return
                    raise e
                finally:
                    # Clean up the event loop
                    try:
                        # Cancel all pending tasks
                        pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
                        if pending_tasks:
                            for task in pending_tasks:
                                task.cancel()
                            
                            # Wait for tasks to be cancelled
                            loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
                    except Exception as cleanup_error:
                        print(f"[DEBUG] Error during loop cleanup: {cleanup_error}")
                    finally:
                        try:
                            loop.close()
                        except Exception as close_error:
                            print(f"[DEBUG] Error closing loop: {close_error}")
                        # Remove from session loops
                        SESSION_LOOPS.pop(session_id, None)
            else:
                add_process_step(session_id, "System", "❌ Error: Invalid session data")
                    
        except Exception as e:
            if session_id in BACKGROUND_THREADS and BACKGROUND_THREADS[session_id]["stop_event"].is_set():
                add_process_step(session_id, "System", "🛑 Processing stopped")
                return
            
            error_message = str(e)
            if "Rate limit" in error_message or "quota" in error_message:
                add_process_step(session_id, "System", f"⚠️ {error_message}")
            else:
                add_process_step(session_id, "System", f"❌ Error: {error_message}")
        
        finally:
            PROCESSING_STATUS[session_id] = {"status": "complete", "task_id": None}
            # Clean up background thread tracking
            if session_id in BACKGROUND_THREADS:
                BACKGROUND_THREADS[session_id].pop("current_task", None)
            print(f"[DEBUG] Background processing completed for session {session_id}")
    
    def monitor_cancellation():
        """Monitor for cancellation from main thread and force stop if needed."""
        stop_event = BACKGROUND_THREADS[session_id]["stop_event"]
        processing_thread = BACKGROUND_THREADS[session_id]["thread"]
        
        while processing_thread.is_alive() and not stop_event.is_set():
            time.sleep(0.1)  # Check every 100ms
        
        if stop_event.is_set():
            print(f"[DEBUG] Monitor detected stop signal for session {session_id}")
            
            # Cancel the current async task if it exists
            if "current_task" in BACKGROUND_THREADS[session_id]:
                current_task = BACKGROUND_THREADS[session_id]["current_task"]
                if current_task and not current_task.done():
                    print(f"[DEBUG] Cancelling current async task for session {session_id}")
                    current_task.cancel()
            
            # Wait for thread to finish gracefully using global timeout
            thread_timeout = min(20.0, PROCESSING_TIMEOUT / 150)  # Proportional to main timeout, max 20s
            processing_thread.join(timeout=thread_timeout)
            if processing_thread.is_alive():
                print(f"[DEBUG] Thread still alive after {thread_timeout}s, forcing cleanup for session {session_id}")
                # Force cleanup session data to prevent further execution
                if session_id in SESSIONS:
                    SESSIONS[session_id] = None
    
    # Create stop event and cancellation token
    stop_event = threading.Event()
    cancellation_token = CancellationToken()
    
    # Start background thread with daemon mode for easier cleanup
    processing_thread = threading.Thread(target=process_agent_response, daemon=True)
    
    # Store thread and stop controls
    BACKGROUND_THREADS[session_id] = {
        "thread": processing_thread,
        "stop_event": stop_event,
        "cancellation_token": cancellation_token
    }
    
    processing_thread.start()
    
    # Start monitoring thread (also daemon)
    monitor_thread = threading.Thread(target=monitor_cancellation, daemon=True)
    BACKGROUND_THREADS[session_id]["monitor_thread"] = monitor_thread
    monitor_thread.start()

def stop_processing(session_id):
    """Stop the background processing for a session with proper cleanup."""
    if session_id in BACKGROUND_THREADS:
        print(f"[DEBUG] Stopping background processing for session {session_id}")
        
        # Signal the thread to stop
        BACKGROUND_THREADS[session_id]["stop_event"].set()
        
        # Cancel the async task directly if available
        if "current_task" in BACKGROUND_THREADS[session_id]:
            current_task = BACKGROUND_THREADS[session_id]["current_task"]
            if current_task and not current_task.done():
                print(f"[DEBUG] Directly cancelling async task for session {session_id}")
                current_task.cancel()
        
        # Cancel the autogen token
        try:
            cancellation_token = BACKGROUND_THREADS[session_id]["cancellation_token"]
            cancellation_token.cancel()
        except Exception as e:
            print(f"[DEBUG] Error cancelling token: {e}")
        
        # Clean up the event loop for this session
        if session_id in SESSION_LOOPS:
            try:
                loop_info = SESSION_LOOPS[session_id]
                loop = loop_info.get("loop")
                if loop and loop.is_running():
                    print(f"[DEBUG] Stopping event loop for session {session_id}")
                    loop.call_soon_threadsafe(loop.stop)
            except Exception as e:
                print(f"[DEBUG] Error stopping event loop: {e}")
        
        # Clean up the agent system immediately but keep adapter for history
        try:
            if session_id in SESSIONS and isinstance(SESSIONS[session_id], dict):
                agent_system = SESSIONS[session_id].get('system')
                if agent_system:
                    print(f"[DEBUG] Cleaning up agent system for session {session_id}")
                    # Forcefully clear the system
                    if hasattr(agent_system, 'team'):
                        agent_system.team = None
                
                # Don't clear SESSIONS[session_id] completely - keep it for history/continuation
                # SESSIONS[session_id] = None
        except Exception as e:
            print(f"[DEBUG] Error during agent system cleanup: {e}")
        
        # Wait for threads to finish with proportional timeout
        thread_timeout = min(5.0, PROCESSING_TIMEOUT / 600)  # Shorter timeout for faster interrupt
        processing_thread = BACKGROUND_THREADS[session_id]["thread"]
        monitor_thread = BACKGROUND_THREADS[session_id].get("monitor_thread")
        
        processing_thread.join(timeout=thread_timeout)
        if monitor_thread:
            monitor_thread.join(timeout=thread_timeout / 2)
        
        # Update status and clean up tracking
        PROCESSING_STATUS[session_id] = {"status": "stopped", "task_id": None}
        BACKGROUND_THREADS.pop(session_id, None)
        SESSION_LOOPS.pop(session_id, None)
        
        add_process_step(session_id, "System", "🛑 Processing stopped by user")

# --- Message Update Callback (Real-time) ---
@app.callback(
    [Output('final-results-content', 'children'),
     Output('process-details-content', 'children')],
    Input('message-updater', 'n_intervals'),
    State('session-id', 'data'),
    State('process-details-content', 'children'),  # Add current state
    prevent_initial_call=True
)
def update_display_panels(n_intervals, session_data, current_process_content):
    """Update both chat messages and process details panels."""
    if not session_data or 'id' not in session_data:
        return ([html.P("Your conversation will appear here...", 
                       style={'color': '#6c757d', 'font-style': 'italic', 'text-align': 'center', 'margin-top': '50px'})],
                [html.P("Process steps will appear here during execution...", 
                       style={'color': '#6c757d', 'font-style': 'italic'})])
    
    session_id = session_data['id']
    
    # Get chat messages for left panel
    chat_messages = get_left_chat_display(session_id)
    
    # Check if we have new process steps
    current_step_count = len(PROCESS_DETAILS.get(session_id, []))
    
    # If we have current content and the step count hasn't changed, don't update process details
    if (current_process_content and 
        current_process_content != [html.P("Process steps will appear here during execution...", 
                                         style={'color': '#6c757d', 'font-style': 'italic'})] and
        hasattr(current_process_content, '__len__') and
        len(current_process_content) == current_step_count):
        return chat_messages, dash.no_update
    
    # Get process details for right panel only if there are new steps
    process_details = get_process_details_display(session_id)
    
    return chat_messages, process_details

# --- Combined Callback for Session and Chat Logic ---
@app.callback(
    [Output('session-id', 'data'),
     Output('user-input', 'value'),
     Output('stop-button', 'style'),
     Output('stop-button', 'disabled')],
    [Input('send-button', 'n_clicks'),
     Input('new-chat-button', 'n_clicks'),
     Input('stop-button', 'n_clicks')],
    [State('user-input', 'value'),
     State('session-id', 'data'),
     State('team-type-selector', 'value')],
    prevent_initial_call=True
)
def handle_user_actions(send_clicks, new_chat_clicks, stop_clicks, user_input, session_data, team_type):
    """Handle user actions (send message, new chat, stop processing) - non-blocking."""
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    session_id = session_data.get('id') if session_data else None

    # Default stop button style (disabled and grayed out)
    stop_button_style = {'width': '100%', 'height': '35px', 'margin-bottom': '5px', 'border-radius': '0', 'opacity': '0.5'}
    stop_button_disabled = True

    # --- ACTION 1: Stop Processing ---
    if triggered_id == 'stop-button' and stop_clicks > 0 and session_id:
        stop_processing(session_id)
        return ({'id': session_id}, dash.no_update, stop_button_style, stop_button_disabled)

    # --- ACTION 2: Start a New Chat ---
    if triggered_id == 'new-chat-button' and new_chat_clicks > 0:
        if session_id:
            # Stop any running processing first
            if session_id in BACKGROUND_THREADS:
                stop_processing(session_id)
            
            # Clean up old session
            print(f"Destroying session: {session_id}")
            SESSIONS.pop(session_id, None)
            PROCESSING_STATUS.pop(session_id, None)
            LEFT_CHAT_MESSAGES.pop(session_id, None)
            PROCESS_DETAILS.pop(session_id, None)
            STEP_STATES.pop(session_id, None)
            BACKGROUND_THREADS.pop(session_id, None)
            SESSION_STATE.pop(session_id, None)  # Clean up session state
            SESSION_LOOPS.pop(session_id, None)  # Clean up event loops

        # Generate unique session ID with timestamp for better organization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_session_id = f"session_{timestamp}_{uuid.uuid4().hex[:8]}"
        print(f"Creating new session: {new_session_id}")
        
        # Initialize new session
        PROCESSING_STATUS[new_session_id] = {"status": "idle", "task_id": None}
        LEFT_CHAT_MESSAGES[new_session_id] = []
        PROCESS_DETAILS[new_session_id] = []
        STEP_STATES[new_session_id] = {}
        
        # Add welcome message to process details
        add_process_step(new_session_id, "System", "New chat session started. Ready to assist with your bio-medical research questions.")
        
        # Add welcome message to left chat
        add_left_chat_message(new_session_id, "System", "Welcome! I'm ready to help with your biomedical research questions.", "system")

        return ({'id': new_session_id}, '', stop_button_style, stop_button_disabled)

    # --- ACTION 3: Send a Message ---
    if triggered_id == 'send-button' and send_clicks > 0 and user_input:
        if not session_id:
            # Create new session if none exists with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            session_id = f"session_{timestamp}_{uuid.uuid4().hex[:8]}"
            PROCESSING_STATUS[session_id] = {"status": "idle", "task_id": None}
            LEFT_CHAT_MESSAGES[session_id] = []
            PROCESS_DETAILS[session_id] = []
            STEP_STATES[session_id] = {}
        
        # Add user question to left chat immediately
        set_user_question(session_id, user_input)
        
        # Add user input to process details
        # add_process_step(session_id, "User", f"**Question:** {user_input}")
        
        # Start background processing
        start_processing_in_background(session_id, user_input, team_type)
        
        # Enable stop button
        stop_button_style = {'width': '100%', 'height': '35px', 'margin-bottom': '5px', 'border-radius': '0', 'opacity': '1.0'}
        stop_button_disabled = False

        return ({'id': session_id}, '', stop_button_style, stop_button_disabled)

    # --- Default/Initial State ---
    if not session_id:
        session_id = str(uuid.uuid4())
        PROCESSING_STATUS[session_id] = {"status": "idle", "task_id": None}
        LEFT_CHAT_MESSAGES[session_id] = []
        PROCESS_DETAILS[session_id] = []
        STEP_STATES[session_id] = {}
        
        # Add welcome message
        add_process_step(session_id, "System", "Welcome to the Multi-Agentic System for Bio-medical Research! Your session is ready.")
        add_left_chat_message(session_id, "System", "Welcome! I'm ready to help with your biomedical research questions.", "system")
        
        return ({'id': session_id}, '', stop_button_style, stop_button_disabled)

    return (dash.no_update, dash.no_update, dash.no_update, dash.no_update)

# --- Update Stop Button Visibility Based on Processing Status ---
@app.callback(
    [Output('stop-button', 'style', allow_duplicate=True),
     Output('stop-button', 'disabled', allow_duplicate=True)],
    Input('message-updater', 'n_intervals'),
    State('session-id', 'data'),
    prevent_initial_call=True
)
def update_stop_button_visibility(n_intervals, session_data):
    """Enable/disable stop button based on processing status."""
    if not session_data or 'id' not in session_data:
        return ({'width': '100%', 'height': '35px', 'margin-bottom': '5px', 'border-radius': '0', 'opacity': '0.5'}, True)
    
    session_id = session_data['id']
    
    # Check if processing
    is_processing = (
        session_id in PROCESSING_STATUS and 
        PROCESSING_STATUS[session_id].get("status") == "processing" and
        session_id in BACKGROUND_THREADS
    )
    
    if is_processing:
        return ({'width': '100%', 'height': '35px', 'margin-bottom': '5px', 'border-radius': '0', 'opacity': '1.0'}, False)
    else:
        return ({'width': '100%', 'height': '35px', 'margin-bottom': '5px', 'border-radius': '0', 'opacity': '0.5'}, True)

# --- Expand All / Collapse All Callbacks ---
@app.callback(
    Output('process-details-content', 'children', allow_duplicate=True),
    [Input('expand-all-button', 'n_clicks'),
     Input('collapse-all-button', 'n_clicks')],
    State('session-id', 'data'),
    prevent_initial_call=True
)
def handle_expand_collapse_all(expand_clicks, collapse_clicks, session_data):
    """Handle expand all and collapse all button clicks."""
    if not session_data or 'id' not in session_data:
        return dash.no_update
    
    session_id = session_data['id']
    ctx = callback_context
    
    if not ctx.triggered:
        return dash.no_update
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if session_id not in PROCESS_DETAILS or not PROCESS_DETAILS[session_id]:
        return dash.no_update
    
    if session_id not in STEP_STATES:
        STEP_STATES[session_id] = {}
    
    # Create new step cards with all expanded or all collapsed
    steps = []
    expand_all = triggered_id == 'expand-all-button' and expand_clicks > 0
    
    for i, step in enumerate(PROCESS_DETAILS[session_id]):
        step_index = f"{session_id}-{i}"
        is_expanded = expand_all  # Expand all or collapse all
        
        # Update STEP_STATES to reflect the new state
        STEP_STATES[session_id][step_index] = {
            "is_open": is_expanded,
            "user_interacted": True  # Mark as user interacted since they used expand/collapse all
        }
        
        step_card = dbc.Card([
            dbc.CardHeader([
                dbc.Button([
                    html.Span("▼ " if is_expanded else "▶ ", 
                             id={'type': 'icon', 'index': step_index}, 
                             style={'margin-right': '5px'}),
                    html.Span(f"{step['agent']}", style={'font-weight': 'bold'}),
                    html.Span(f" - Step {step['step_number']}", style={'color': '#6c757d', 'font-size': '0.9em', 'margin-left': '10px'})
                ], 
                id={'type': 'toggle', 'index': step_index},
                color="light",
                style={
                    'width': '100%', 
                    'text-align': 'left', 
                    'border': 'none',
                    'background': 'transparent',
                    'padding': '10px 15px'
                })
            ], 
            style={'padding': '0', 'background': '#f1f3f4', 'border-bottom': '1px solid #e3e6ea'}),
            
            dbc.Collapse([
                dbc.CardBody([
                    dcc.Markdown(
                        children=step['content'],
                        className='markdown-content',
                        style={'margin': '0'}
                    )
                ], style={'padding': '15px'})
            ], 
            id={'type': 'collapse', 'index': step_index},
            is_open=is_expanded)
        ], style={'margin-bottom': '10px', 'border': '1px solid #e3e6ea'})
        
        steps.append(step_card)
    
    return steps

# --- Toggle Collapsible Process Steps ---
@app.callback(
    [Output({'type': 'collapse', 'index': dash.dependencies.MATCH}, 'is_open'),
     Output({'type': 'icon', 'index': dash.dependencies.MATCH}, 'children')],
    Input({'type': 'toggle', 'index': dash.dependencies.MATCH}, 'n_clicks'),
    State({'type': 'collapse', 'index': dash.dependencies.MATCH}, 'is_open'),
    State('session-id', 'data'),
    prevent_initial_call=True
)
def toggle_collapse(n_clicks, is_open, session_data):
    """Toggle collapsible sections in the process details panel and preserve state."""
    if n_clicks and session_data and 'id' in session_data:
        session_id = session_data['id']
        step_index = dash.callback_context.triggered[0]['prop_id'].split('.')[0].split('"index":"')[1].split('"')[0]
        
        # Update the step state to track user interaction
        if session_id not in STEP_STATES:
            STEP_STATES[session_id] = {}
        
        new_state = not is_open
        
        # Simply update this step's state without affecting others
        # This allows multiple steps to be open simultaneously
        STEP_STATES[session_id][step_index] = {
            "is_open": new_state,
            "user_interacted": True  # Mark this step as user-controlled
        }
        
        icon = "▼ " if new_state else "▶ "
        return new_state, icon
    return is_open, "▶ "


# --- Download Session Storage ---
# Track zip preparation status per session
DOWNLOAD_STATUS = {}  # session_id -> {"status": "idle/preparing/ready/error", "zip_path": None, "error": None}

def prepare_zip_in_background(session_id, session_dir):
    """Prepare zip file in background thread."""
    try:
        DOWNLOAD_STATUS[session_id] = {"status": "preparing", "zip_path": None, "error": None}
        print(f"[DEBUG] Starting zip preparation for session: {session_id}")
        
        import zipfile
        
        # Define the zip file path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"{session_id}_{timestamp}.zip"
        zip_path = os.path.join(session_dir, zip_filename)
        
        # Count and zip files
        file_count = 0
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(session_dir):
                for file in files:
                    # Skip session download zip files (but include .npy.zip compressed data files)
                    if file.endswith('.zip') and not file.endswith('.npy.zip'):
                        continue
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, session_dir)
                    zip_file.write(file_path, arcname)
                    file_count += 1
                    print(f"[DEBUG] Added to zip: {arcname}")
        
        print(f"[DEBUG] Zip ready: {zip_path} with {file_count} files")
        DOWNLOAD_STATUS[session_id] = {"status": "ready", "zip_path": zip_path, "zip_filename": zip_filename, "error": None}
        
    except Exception as e:
        print(f"[ERROR] Failed to create zip: {e}")
        import traceback
        traceback.print_exc()
        DOWNLOAD_STATUS[session_id] = {"status": "error", "zip_path": None, "error": str(e)}


# --- Download Session Zip Callback ---
@app.callback(
    Output('download-session-zip', 'data'),
    Output('download-session-button', 'children'),
    Input('download-session-button', 'n_clicks'),
    Input('message-updater', 'n_intervals'),
    State('session-id', 'data'),
    prevent_initial_call=True
)
def download_session_zip(n_clicks, n_intervals, session_data):
    """Handle download button click and zip preparation status. Auto-downloads when ready."""
    ctx = callback_context
    if not ctx.triggered:
        return None, "📥 Download Session"
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if not session_data or 'id' not in session_data:
        return None, "📥 Download Session"
    
    session_id = session_data['id']
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    
    # Check current download status
    current_status = DOWNLOAD_STATUS.get(session_id, {"status": "idle"})
    
    # Auto-download when zip is ready (triggered by interval check)
    if current_status.get("status") == "ready" and current_status.get("zip_path"):
        zip_path = current_status["zip_path"]
        zip_filename = current_status.get("zip_filename", os.path.basename(zip_path))
        if os.path.exists(zip_path):
            # Reset status for next download
            DOWNLOAD_STATUS[session_id] = {"status": "idle", "zip_path": None, "error": None}
            print(f"[DEBUG] Auto-downloading zip: {zip_filename}")
            return dcc.send_file(zip_path, filename=zip_filename), "📥 Download Session"
    
    # If button was clicked, start preparation
    if triggered_id == 'download-session-button' and n_clicks:
        # If not already preparing, start preparation
        if current_status.get("status") not in ["preparing", "ready"]:
            # Ensure session directory exists
            if not os.path.exists(session_dir):
                os.makedirs(session_dir, exist_ok=True)
            
            # Start background thread to prepare zip
            thread = threading.Thread(
                target=prepare_zip_in_background,
                args=(session_id, session_dir),
                daemon=True
            )
            thread.start()
            return None, "⏳ Preparing..."
    
    # Update button text based on status
    if current_status.get("status") == "preparing":
        return None, "⏳ Preparing..."
    elif current_status.get("status") == "error":
        return None, "❌ Error - Try Again"
    
    return None, "📥 Download"


# --- Continue Chat Callbacks ---

@app.callback(
    Output('hitl-controls-col', 'style'),
    Input('message-updater', 'n_intervals'),
    State('session-id', 'data'),
    prevent_initial_call=True
)
def update_continue_button_visibility(n_intervals, session_data):
    """Show Continue button after report is complete."""
    if not session_data or 'id' not in session_data:
        return {'display': 'none'}
    
    session_id = session_data['id']
    session_info = SESSION_STATE.get(session_id, {})
    
    # Show when can_continue is True (report completed)
    if session_info.get("can_continue", False):
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('user-input', 'value', allow_duplicate=True),
    Input('hitl-continue-button', 'n_clicks'),
    State('session-id', 'data'),
    State('user-input', 'value'),
    prevent_initial_call=True
)
def handle_continue_chat(continue_clicks, session_data, user_input):
    """Handle Continue button - start new conversation round."""
    ctx = callback_context
    if not ctx.triggered or not session_data:
        return dash.no_update
    
    session_id = session_data.get('id')
    
    if not session_id or session_id not in SESSION_STATE:
        return dash.no_update
    
    session_info = SESSION_STATE[session_id]
    adapter = session_info.get("adapter")
    
    if not adapter:
        return dash.no_update
    
    if not session_info.get("can_continue"):
        add_process_step(session_id, "System", "⚠️ Cannot continue - wait for report to complete")
        return dash.no_update
    
    if not user_input or not user_input.strip():
        add_process_step(session_id, "System", "⚠️ Please enter a follow-up question first")
        return dash.no_update
    
    # Reset continue state
    SESSION_STATE[session_id]["can_continue"] = False
    
    # Add user message to chat
    add_left_chat_message(session_id, "User", user_input, "user")
    add_process_step(session_id, "User", f"🔄 **Follow-up**: {user_input[:100]}...")
    
    # Start continuation in background
    def continue_conversation_bg():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                adapter.continue_conversation(user_input)
            )
            # Mark can_continue again after completion
            _mark_can_continue(session_id)
            loop.close()
        except Exception as e:
            add_process_step(session_id, "System", f"❌ Error continuing: {e}")
    
    thread = threading.Thread(target=continue_conversation_bg, daemon=True)
    thread.start()
    
    return ''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Agentic System for Bio-medical Research UI')
    parser.add_argument('--share', action='store_true', help='Create a shareable public URL using ngrok')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the server on (default: 8050)')
    args = parser.parse_args()
    
    if args.share:
        from pyngrok import ngrok
        
        print("Starting ngrok tunnel...")
        public_tunnel = ngrok.connect(args.port)
        print(f"Local:  http://127.0.0.1:{args.port}")
        print(f"Public: {public_tunnel.public_url}")
        
        try:
            app.run(debug=False, port=args.port, host='0.0.0.0')
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            ngrok.disconnect(public_tunnel.public_url)
            ngrok.kill()
            print("Tunnel closed")
    else:
        print(f"Running locally: http://127.0.0.1:{args.port}")
        app.run(debug=True, port=args.port)