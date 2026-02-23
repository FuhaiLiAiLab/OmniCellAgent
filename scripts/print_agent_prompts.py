#!/usr/bin/env python3
"""
Print all docstrings, sub-agent system prompts, imported prompt constants,
and tool descriptions from the LangGraphOmniCellAgent.

Outputs are auto-saved to logs/exported_prompts/.

Usage:
    python scripts/print_agent_prompts.py
    python scripts/print_agent_prompts.py --json
    python scripts/print_agent_prompts.py --prompts-only
    python scripts/print_agent_prompts.py --tools-only
"""
import argparse
import inspect
import io
import json
import os
import re
import sys
import textwrap
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

SEP = "=" * 88
THIN = "-" * 88


def heading(title: str, char: str = "=") -> str:
    bar = char * 88
    return f"\n{bar}\n  {title}\n{bar}"


def fmt(text: str, prefix: str = "    ") -> str:
    return textwrap.indent(textwrap.dedent(text).strip(), prefix)


def help_text(obj) -> str:
    buf = io.StringIO()
    old_stdout = sys.__stdout__
    sys.stdout = buf
    try:
        help(obj)
    finally:
        sys.stdout = old_stdout
    return buf.getvalue()


class TeeWriter:
    """Write to both the real terminal and a log file."""
    def __init__(self, stream, fileobj):
        self.stream = stream
        self.fileobj = fileobj

    def write(self, data):
        self.stream.write(data)
        self.fileobj.write(data)

    def flush(self):
        self.stream.flush()
        self.fileobj.flush()


def main():
    parser = argparse.ArgumentParser(description="Print agent prompts & docstrings")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    parser.add_argument("--prompts-only", action="store_true", help="Only system prompts")
    parser.add_argument("--tools-only", action="store_true", help="Only tool descriptions")
    args = parser.parse_args()

    # ---- auto-save setup ----
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    export_dir = os.path.join(project_root, 'logs', 'exported_prompts')
    os.makedirs(export_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    ext = '.json' if args.json else '.txt'
    export_path = os.path.join(export_dir, f'prompts_{ts}{ext}')
    export_file = open(export_path, 'w', encoding='utf-8')
    tee = TeeWriter(sys.__stdout__, export_file)

    def out(s: str = ""):
        tee.write(s + "\n")

    def finish():
        export_file.close()
        print(f"💾 Saved to: {export_path}", file=sys.__stderr__)

    # ---- silent import & instantiate ----
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

    # A) Module & class docstrings
    module_doc = (mod.__doc__ or "").strip()
    agent_class_doc = (LangGraphOmniCellAgent.__doc__ or "").strip()
    agent_init_doc = (LangGraphOmniCellAgent.__init__.__doc__ or "").strip()

    # B) Orchestrator prompts (extracted from method source via inspect)
    def _extract_prompts_from_method(cls, method_name):
        """Extract SystemMessage content and large prompt strings from method source."""
        meth = getattr(cls, method_name, None)
        if not meth:
            return []
        src = inspect.getsource(meth)
        prompts = []
        # 1) SystemMessage(content="""...""")
        for m in re.finditer(
            r'SystemMessage\(content="""(.*?)"""\)', src, re.DOTALL
        ):
            prompts.append((f"{method_name} → SystemMessage", textwrap.dedent(m.group(1)).strip()))
        # 2) _prompt = f"""..."""  or  _prompt = """..."""
        for m in re.finditer(
            r'(\w+_prompt)\s*=\s*f?"""(.*?)"""', src, re.DOTALL
        ):
            prompts.append((f"{method_name} → {m.group(1)}", textwrap.dedent(m.group(2)).strip()))
        return prompts

    orchestrator_prompts = []
    for mname in ['_planning_node', '_replanning_node', '_reporting_node']:
        orchestrator_prompts.extend(_extract_prompts_from_method(LangGraphOmniCellAgent, mname))

    # Also collect imported prompt constants actually used by sub-agents
    imported_prompts = {}
    for attr in sorted(dir(mod)):
        if not attr.isupper():
            continue
        val = getattr(mod, attr, None)
        if isinstance(val, str) and len(val.strip()) > 100:
            imported_prompts[attr] = val.strip()

    # C) Sub-agents
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

    # D) All @tool objects (including ones not bound to any sub-agent)
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

    # E) Key method docstrings
    method_docs = {}
    for mname in [
        "__init__", "_init_sub_agents", "_build_graph",
        "_planning_node", "_execution_node",
        "_replanning_node", "_reporting_node",
        "_extract_structured_data",
        "_should_execute_or_report", "_should_continue_or_replan",
        "_generate_appendix", "_save_report", "_compile_pdf",
        "run",
    ]:
        meth = getattr(LangGraphOmniCellAgent, mname, None)
        if meth and meth.__doc__:
            method_docs[mname] = meth.__doc__.strip()

    # ==================================================================
    # OUTPUT
    # ==================================================================

    # ---- JSON mode ----
    if args.json:
        blob = {
            "module_docstring": module_doc,
            "agent_class_docstring": agent_class_doc,
            "agent_init_docstring": agent_init_doc,
            "orchestrator_prompts": [{"label": l, "content": c} for l, c in orchestrator_prompts],
            "imported_prompt_constants": imported_prompts,
            "sub_agents": sub_agents_data,
            "all_tools": all_tools_data,
            "method_docstrings": method_docs,
        }
        out(json.dumps(blob, indent=2, ensure_ascii=False))
        finish()
        return

    # ---- --prompts-only ----
    if args.prompts_only:
        out(heading("ORCHESTRATOR PROMPTS (from _planning_node / _reporting_node / _replanning_node)"))
        for label, content in orchestrator_prompts:
            out(f"\n{THIN}")
            out(f"  {label}  ({len(content):,} chars)")
            out(THIN)
            out(fmt(content, "    | "))
        out(heading("IMPORTED PROMPT CONSTANTS"))
        for name, val in imported_prompts.items():
            out(f"\n{THIN}")
            out(f"  {name}  ({len(val):,} chars)")
            out(THIN)
            out(fmt(val, "    | "))
        out(heading("SUB-AGENT SYSTEM PROMPTS"))
        for sa in sub_agents_data:
            out(f"\n{THIN}")
            out(f"  {sa['name']}")
            out(THIN)
            out(fmt(sa["system_message"], "    | "))
        finish()
        return

    # ---- --tools-only ----
    if args.tools_only:
        out(heading("ALL TOOLS"))
        for t in all_tools_data:
            out(f"\n  ▸ {t['name']}")
            out(fmt(t["description"], "    "))
            if t["args"]:
                out(f"    Args: {t['args']}")
        finish()
        return

    # ---- Full output ----
    out(heading("OMNICELL AGENT — PROMPT & DOCSTRING INVENTORY", "█"))

    # 1. Module docstring
    out(heading("1. MODULE DOCSTRING"))
    out(fmt(module_doc) if module_doc else "    (none)")

    # 2. Agent class & __init__ docstrings
    out(heading("2. AGENT CLASS"))
    out(f"\n  class LangGraphOmniCellAgent:")
    out(fmt(agent_class_doc) if agent_class_doc else "    (none)")
    out(f"\n  __init__():")
    out(fmt(agent_init_doc) if agent_init_doc else "    (none)")

    # 3. help() on agent class
    out(heading("3. help(LangGraphOmniCellAgent)"))
    h = help_text(LangGraphOmniCellAgent)
    lines = h.splitlines()
    for line in lines[:80]:
        out(f"    {line}")
    if len(lines) > 80:
        out(f"    ... ({len(lines) - 80} more lines, use --json for full)")

    # 4. Orchestrator prompts (the actual planning/reporting/replanning prompts)
    out(heading("4. ORCHESTRATOR PROMPTS"))
    if orchestrator_prompts:
        for label, content in orchestrator_prompts:
            out(f"\n{THIN}")
            out(f"  {label}  ({len(content):,} chars)")
            out(THIN)
            out(fmt(content, "    | "))
    else:
        out("    (none found)")

    # 4b. Imported prompt constants (from utils.prompt etc.)
    out(heading("4b. IMPORTED PROMPT CONSTANTS"))
    if imported_prompts:
        for name, val in imported_prompts.items():
            out(f"\n{THIN}")
            out(f"  {name}  ({len(val):,} chars)")
            out(THIN)
            out(fmt(val, "    | "))
    else:
        out("    (none found)")

    # 5. Sub-agent system prompts & tools
    out(heading("5. SUB-AGENT SYSTEM PROMPTS & TOOLS"))
    for sa in sub_agents_data:
        out(f"\n{THIN}")
        out(f"  Agent:       {sa['name']}")
        out(f"  Description: {sa['description']}")
        out(THIN)
        out("\n  System Message:")
        out(fmt(sa["system_message"], "    | "))
        out()
        if sa["tools"]:
            out(f"  Tools ({len(sa['tools'])}):")
            for t in sa["tools"]:
                out(f"\n    ▸ {t['name']}")
                out(fmt(t["description"], "      "))
                if t["args"]:
                    out(f"      Args: {t['args']}")
        else:
            out("  Tools: (none)")

    # 6. All tools
    out(heading("6. ALL REGISTERED TOOLS"))
    for t in all_tools_data:
        out(f"\n  ▸ {t['name']}")
        out(fmt(t["description"], "    "))
        if t["args"]:
            out(f"    Args: {t['args']}")

    # 7. Method docstrings
    out(heading("7. KEY METHOD DOCSTRINGS"))
    for mname, doc in method_docs.items():
        out(f"\n  {mname}():")
        out(fmt(doc))

    # Summary
    out(heading("SUMMARY"))
    out(f"  Orchestrator prompts: {len(orchestrator_prompts)}")
    out(f"  Imported constants:   {len(imported_prompts)}")
    out(f"  Sub-agents:           {len(sub_agents_data)}")
    out(f"  Tools (total):        {len(all_tools_data)}")
    out(f"  Documented methods:   {len(method_docs)}")
    total_prompt_chars = (
        sum(len(c) for _, c in orchestrator_prompts)
        + sum(len(v) for v in imported_prompts.values())
        + sum(len(sa["system_message"]) for sa in sub_agents_data)
    )
    out(f"  Total prompt chars:  {total_prompt_chars:,}")
    out()

    finish()


if __name__ == "__main__":
    main()
