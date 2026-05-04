#!/usr/bin/env python3
"""
Export all agent prompts, tool descriptions, and documentation to a formatted
text file suitable for inclusion in a paper appendix.

Outputs to: logs/appendix/appendix_prompts.txt

Usage:
    python scripts/export_appendix_prompts.py
"""
import inspect
import io
import os
import re
import sys
import textwrap
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def heading(title: str, level: int = 1) -> str:
    """Generate a formatted heading."""
    if level == 1:
        return f"\n{'=' * 80}\n{title}\n{'=' * 80}\n"
    elif level == 2:
        return f"\n{'-' * 60}\n{title}\n{'-' * 60}\n"
    else:
        return f"\n{title}\n{'-' * len(title)}\n"


def fmt(text: str, indent: int = 4) -> str:
    """Format and indent text."""
    prefix = " " * indent
    return textwrap.indent(textwrap.dedent(text).strip(), prefix)


def main():
    # ---- Setup output ----
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    export_dir = os.path.join(project_root, 'logs', 'appendix')
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, 'appendix_prompts.txt')
    
    output_lines = []
    def out(s: str = ""):
        output_lines.append(s)

    # ---- Silent import & instantiate ----
    import logging
    import warnings
    devnull = open(os.devnull, "w")
    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = devnull, devnull
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    try:
        import agent.langgraph_agent as mod
        from agent.langgraph_agent import LangGraphOmniCellAgent
        agent = LangGraphOmniCellAgent(session_id="_introspect")
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err
        logging.disable(logging.NOTSET)
        devnull.close()

    # ==================================================================
    # Collect data from live objects
    # ==================================================================

    # A) Module docstring
    module_doc = (mod.__doc__ or "").strip()

    # B) Orchestrator prompts
    def _extract_prompts_from_method(cls, method_name):
        """Extract SystemMessage content and large prompt strings from method source."""
        meth = getattr(cls, method_name, None)
        if not meth:
            return []
        src = inspect.getsource(meth)
        prompts = []
        # SystemMessage(content="""...""")
        for m in re.finditer(
            r'SystemMessage\(content="""(.*?)"""\)', src, re.DOTALL
        ):
            prompts.append((f"{method_name} System Message", textwrap.dedent(m.group(1)).strip()))
        # _prompt = f"""..."""  or  _prompt = """..."""
        for m in re.finditer(
            r'(\w+_prompt)\s*=\s*f?"""(.*?)"""', src, re.DOTALL
        ):
            prompts.append((f"{method_name} {m.group(1)}", textwrap.dedent(m.group(2)).strip()))
        return prompts

    orchestrator_prompts = []
    for mname in ['_planning_node', '_replanning_node', '_reporting_node']:
        orchestrator_prompts.extend(_extract_prompts_from_method(LangGraphOmniCellAgent, mname))

    # C) Imported prompt constants
    imported_prompts = {}
    for attr in sorted(dir(mod)):
        if not attr.isupper():
            continue
        val = getattr(mod, attr, None)
        if isinstance(val, str) and len(val.strip()) > 100:
            imported_prompts[attr] = val.strip()

    # D) Sub-agents
    sub_agents_data = []
    for name, sa in agent.sub_agents.items():
        tools_data = []
        for t in sa.tools:
            tools_data.append({
                "name": t.name,
                "description": t.description,
                "args": {k: str(v) for k, v in (t.args or {}).items()},
            })
        sub_agents_data.append({
            "name": sa.name,
            "description": sa.description,
            "system_message": sa.system_message,
            "tools": tools_data,
        })

    # E) All tools
    all_tools_data = []
    seen = {t["name"] for sa in sub_agents_data for t in sa["tools"]}
    all_tools_data = [t for sa in sub_agents_data for t in sa["tools"]]
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if hasattr(obj, "name") and hasattr(obj, "description") and hasattr(obj, "invoke"):
            if obj.name not in seen:
                seen.add(obj.name)
                all_tools_data.append({
                    "name": obj.name,
                    "description": obj.description,
                    "args": {k: str(v) for k, v in (getattr(obj, "args", {}) or {}).items()},
                })

    # ==================================================================
    # Generate Appendix Output
    # ==================================================================

    out("APPENDIX: OmniCellAgent System Prompts and Tool Documentation")
    out(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out("=" * 80)

    # Section A: System Overview
    out(heading("A. System Overview", 1))
    out(fmt(module_doc) if module_doc else "    (No module docstring)")

    # Section B: Orchestrator Prompts
    out(heading("B. Orchestrator Prompts", 1))
    out("These prompts guide the agent's planning, replanning, and reporting phases.")
    
    for i, (label, content) in enumerate(orchestrator_prompts, 1):
        out(heading(f"B.{i} {label}", 2))
        out(fmt(content))

    # Section C: Sub-Agent System Prompts
    out(heading("C. Sub-Agent System Prompts", 1))
    out("Each sub-agent has a specialized system prompt that defines its capabilities and behavior.")
    
    for i, sa in enumerate(sub_agents_data, 1):
        out(heading(f"C.{i} {sa['name']}", 2))
        out(f"    Description: {sa['description']}")
        out("")
        out("    System Prompt:")
        out(fmt(sa["system_message"], 8))
        
        if sa["tools"]:
            out("")
            out(f"    Available Tools ({len(sa['tools'])}):")
            for t in sa["tools"]:
                out(f"      - {t['name']}")

    # Section D: Tool Descriptions
    out(heading("D. Tool Descriptions", 1))
    out("Complete documentation for all available tools in the system.")
    
    for i, t in enumerate(all_tools_data, 1):
        out(heading(f"D.{i} {t['name']}", 3))
        out(fmt(t["description"]))
        if t["args"]:
            out("")
            out("    Arguments:")
            for k, v in t["args"].items():
                out(f"      - {k}: {v}")

    # Section E: Prompt Constants
    if imported_prompts:
        out(heading("E. Imported Prompt Constants", 1))
        out("These are reusable prompt templates imported from utility modules.")
        
        for i, (name, val) in enumerate(imported_prompts.items(), 1):
            out(heading(f"E.{i} {name}", 2))
            out(fmt(val))

    # Summary statistics
    out(heading("Summary Statistics", 1))
    out(f"    Orchestrator prompts: {len(orchestrator_prompts)}")
    out(f"    Sub-agents:           {len(sub_agents_data)}")
    out(f"    Tools:                {len(all_tools_data)}")
    out(f"    Prompt constants:     {len(imported_prompts)}")
    total_chars = (
        sum(len(c) for _, c in orchestrator_prompts)
        + sum(len(sa["system_message"]) for sa in sub_agents_data)
        + sum(len(v) for v in imported_prompts.values())
    )
    out(f"    Total prompt chars:   {total_chars:,}")

    # ---- Write output ----
    with open(export_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))
    
    print(f"✅ Appendix exported to: {export_path}")
    print(f"   - {len(orchestrator_prompts)} orchestrator prompts")
    print(f"   - {len(sub_agents_data)} sub-agent prompts")
    print(f"   - {len(all_tools_data)} tool descriptions")
    print(f"   - {total_chars:,} total prompt characters")


if __name__ == "__main__":
    main()
