"""A disease-vs-normal omics run needs a cell type and a database disease name: without
them the tool runs nothing and returns the cell types or disease names to choose from, the
task fails without a retry, and the replanner receives the list."""

import asyncio
import functools
import json
import os
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)
agent_module = pytest.importorskip("agent.langgraph_agent")
Agent = agent_module.LangGraphOmniCellAgent


def _metadata(n=12):
    rows = []
    for cell_type, disease, count in [("astrocyte", "AD", n), ("astrocyte", "normal", n),
                                      ("microglial cell", "AD", n), ("microglial cell", "normal", 3)]:
        for i in range(count):
            rows.append({"CMT_name": cell_type, "tissue_general": "brain", "disease_BMG_name": disease,
                         "donor_id": f"{cell_type}-{disease}-{i}", "matrix_file_path": "matrix.npy",
                         "matrix_row_idx": len(rows)})
    return pd.DataFrame(rows)


@pytest.fixture
def tool_env(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(agent_module, "_GLOBAL_SESSION_DIR", str(tmp_path))
    monkeypatch.setattr(agent_module, "read_lookup_metadata", lambda path: _metadata())
    monkeypatch.setattr(agent_module, "_omic_workflow",
                        lambda **kw: calls.append(kw) or {"success": True, "message": "recorded"})
    return calls


def test_disease_run_without_cell_type_lists_cell_types_and_runs_nothing(tool_env):
    out = agent_module.omic_analysis_tool.invoke({"disease": "ad", "organ": "brain"})
    assert tool_env == []
    assert out["success"] is False and out["reason"] == "cell_type_required"
    # microglia has 3 normal metacells, below the lookup's minimum of 10
    assert out["cell_type_options"] == {"disease": "AD", "organ": "brain", "cell_types": ["astrocyte"],
                                        "disease_candidates": []}
    assert out["message"].startswith("No analysis was run: a disease-vs-normal comparison needs one cell type")
    assert '- "astrocyte" (disease 12, normal 12 metacells available before sampling)' in out["message"]


@pytest.mark.parametrize("cell_type", [None, "astrocyte"])
def test_unknown_disease_returns_the_closest_disease_names_and_runs_nothing(tool_env, cell_type):
    out = agent_module.omic_analysis_tool.invoke({"disease": "a d", "organ": "brain", "cell_type": cell_type})
    assert tool_env == [] and out["reason"] == "disease_not_found"
    assert out["message"].startswith("No analysis was run: the disease name was not found.")
    assert out["cell_type_options"]["cell_types"] == []
    assert out["cell_type_options"]["disease_candidates"] == ["AD"]


@pytest.mark.parametrize("args", [
    {"disease": "AD", "organ": "brain", "label": "gender"},
    {"disease": "AD", "organ": "brain", "label": "cell_type"},
    # The lookup's minimum does not apply to the analysis itself
    {"disease": "AD", "organ": "brain", "cell_type": "microglial cell"},
])
def test_other_runs_still_reach_the_workflow(tool_env, args):
    out = agent_module.omic_analysis_tool.invoke(args)
    assert out["success"] is True and len(tool_env) == 1
    assert tool_env[0]["label"] == args.get("label", "disease")
    assert tool_env[0].get("cell_type") == args.get("cell_type")


REFUSAL = {"success": False, "reason": "cell_type_required", "message": "No analysis was run ...",
           "cell_type_options": {"disease": "AD", "organ": "brain", "cell_types": ["astrocyte"],
                                 "disease_candidates": []}}


def test_cell_type_options_come_from_the_last_refusal():
    first = {**REFUSAL, "reason": "disease_not_found",
             "cell_type_options": {**REFUSAL["cell_type_options"], "cell_types": []}}
    outputs = [{"tool": "omic_analysis_tool", "args": {}, "output": first},
               {"tool": "omic_analysis_tool", "args": {}, "output": {"success": True}},
               {"tool": "omic_analysis_tool", "args": {}, "output": REFUSAL}]
    assert agent_module._cell_type_options(outputs) == {"reason": "cell_type_required",
                                                        **REFUSAL["cell_type_options"]}
    assert agent_module._cell_type_options(outputs[:1])["reason"] == "disease_not_found"
    assert agent_module._cell_type_options(outputs[1:2]) is None
    assert agent_module._cell_type_options([]) is None


def test_refusal_is_not_recorded_as_an_omics_cohort():
    shared = Agent._extract_structured_data(
        None, "OmicMiningAgent", {"result": ""}, {"omic_runs": []},
        [{"tool": "omic_analysis_tool", "args": {}, "output": REFUSAL}])
    assert shared.get("omic_runs") == []


def _state(task):
    return {"query": "What are the key dysfunctional genes in AD?", "session_id": "s", "plan": [task],
            "current_task_index": 0, "plan_revision_count": 0, "max_plan_revisions": 2,
            "agent_outputs": {}, "shared_data": {"omic_runs": []}, "process_log": []}


@pytest.mark.parametrize("reason, error", [
    ("cell_type_required", "No cell type was given, so no analysis was run"),
    ("disease_not_found", "The disease name was not found, so no analysis was run"),
])
def test_refused_task_fails_without_a_retry_and_carries_the_options(monkeypatch, reason, error):
    refusal = {**REFUSAL, "reason": reason}

    async def execute(task, context):
        return {"success": False, "result": "No cell type was named.", "error": "None of the 1 tool calls ...",
                "tool_outputs": [{"tool": "omic_analysis_tool", "args": {}, "output": refusal}]}

    fake = SimpleNamespace(sub_agents={"OmicMiningAgent": SimpleNamespace(execute=execute)},
                           usage_tracker=SimpleNamespace(record_task=lambda *args: None))
    fake._extract_structured_data = functools.partial(Agent._extract_structured_data, fake)
    task = {"id": "task_1", "description": "Differential expression for AD", "assigned_agent": "OmicMiningAgent",
            "status": "pending", "result": None, "error": None, "attempts": 0}
    state = _state(task)
    state.update(asyncio.run(Agent._execution_node(fake, state)))
    assert task["status"] == "failed" and task["attempts"] == 2
    assert task["cell_type_options"] == {"reason": reason, **REFUSAL["cell_type_options"]}
    assert task["error"].startswith(error)
    assert Agent._should_continue_or_replan(fake, state) == "replan"


def test_replanner_receives_the_options_and_the_rule():
    prompts = []

    class LeadLLM:
        async def ainvoke(self, messages):
            prompts.append(messages[-1].content)
            return SimpleNamespace(content=json.dumps({"revision_reason": "pick cell types", "tasks": [
                {"id": "task_new_1", "assigned_agent": "OmicMiningAgent",
                 "description": 'Differential expression for disease "AD", organ "brain", cell type "astrocyte"'}]}))

    fake = SimpleNamespace(sub_agents={"OmicMiningAgent": SimpleNamespace(description="omics")},
                           lead_llm=LeadLLM())
    task = {"id": "task_1", "description": "Differential expression for AD", "assigned_agent": "OmicMiningAgent",
            "status": "failed", "result": None, "error": "No cell type was given", "attempts": 2,
            "cell_type_options": REFUSAL["cell_type_options"]}
    update = asyncio.run(Agent._replanning_node(fake, _state(task)))
    assert '"cell_type_options": {' in prompts[0] and '"astrocyte"' in prompts[0]
    assert "up to 3 cell\n     types from `cell_type_options.cell_types`" in prompts[0]
    assert '`reason` "disease_not_found"' in prompts[0] and "`cell_type_options.disease_candidates`" in prompts[0]
    assert [t["id"] for t in update["plan"]] == ["task_1", "task_new_1"]
    assert update["current_task_index"] == 1 and task["replaced"]
