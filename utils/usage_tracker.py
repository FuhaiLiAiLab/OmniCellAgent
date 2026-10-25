"""
Per-session usage tracking for the LangGraph agent.

Records, for one agent run:
  - every LangChain LLM call (tokens, wall time), attributed to the agent or graph node that made it
  - wall time of every executed task
  - timing checkpoints and summarizer tokens of every PubMed full-text search
    (the stats_out dict of query_medical_research_async)

The tracker is written to <session_dir>/usage_log.json by LangGraphOmniCellAgent.

Not covered: LLM and embedding calls made through the OpenAI SDK inside tools (web-search page filter,
expert knowledge bases) and calls inside the BioMarkerKG G-Retriever service, which runs in its own process.
"""

import contextvars
import json
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult


# Name of the agent / graph node an LLM call is attributed to
_current_label = contextvars.ContextVar("usage_label", default="unattributed")

# Tracker of the running agent session; read by tools (e.g. pubmed_full_search_tool)
_CURRENT_TRACKER: Optional["UsageTracker"] = None

TOKEN_FIELDS = ["input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"]


@contextmanager
def usage_label(label: str):
    """Attribute LLM calls made inside this block (including awaited ones) to `label`."""
    token = _current_label.set(label)
    try:
        yield
    finally:
        _current_label.reset(token)


def set_current_tracker(tracker: Optional["UsageTracker"]) -> None:
    global _CURRENT_TRACKER
    _CURRENT_TRACKER = tracker


def get_current_tracker() -> Optional["UsageTracker"]:
    return _CURRENT_TRACKER


class UsageTracker:
    def __init__(self, session_id: str, model: str = None, lead_model: str = None):
        self.session_id = session_id
        self.model = model                # sub-agents
        self.lead_model = lead_model      # planner, replanner, reporter, report revision
        self.query = None
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.finished_at = None
        self._start = time.perf_counter()
        self.total_seconds = None
        self.llm_calls: List[Dict[str, Any]] = []
        self.tasks: List[Dict[str, Any]] = []
        self.pubmed_calls: List[Dict[str, Any]] = []

    def record_llm_call(self, call: Dict[str, Any]) -> None:
        self.llm_calls.append(call)

    def record_task(self, task_id: Any, agent: str, seconds: float, success: bool) -> None:
        self.tasks.append({"task_id": task_id, "agent": agent,
                           "seconds": round(seconds, 2), "success": success})

    def record_pubmed_call(self, query: str, stats: Dict[str, Any]) -> None:
        self.pubmed_calls.append({"query": query, **stats})

    def finish(self) -> None:
        self.finished_at = datetime.now().isoformat(timespec="seconds")
        self.total_seconds = round(time.perf_counter() - self._start, 2)

    def summary(self) -> Dict[str, Any]:
        by_agent: Dict[str, Dict[str, Any]] = {}
        for call in self.llm_calls:
            agg = by_agent.setdefault(call["agent"], {"calls": 0, "errors": 0, "llm_seconds": 0.0,
                                                      **{f: 0 for f in TOKEN_FIELDS}})
            agg["calls"] += 1
            agg["errors"] += 1 if call.get("error") else 0
            agg["llm_seconds"] = round(agg["llm_seconds"] + (call.get("seconds") or 0), 2)
            for f in TOKEN_FIELDS:
                agg[f] += call.get(f) or 0
        agent_totals = {f: sum(a[f] for a in by_agent.values()) for f in TOKEN_FIELDS}
        agent_totals["calls"] = len(self.llm_calls)

        task_seconds: Dict[str, float] = {}
        for task in self.tasks:
            task_seconds[task["agent"]] = round(task_seconds.get(task["agent"], 0) + task["seconds"], 2)

        pubmed_tokens = {"prompt_token_count": 0, "candidates_token_count": 0,
                         "thoughts_token_count": 0, "total_token_count": 0}
        for call in self.pubmed_calls:
            for f in pubmed_tokens:
                pubmed_tokens[f] += call.get("llm", {}).get("tokens_total", {}).get(f, 0)
        pubmed = {
            "calls": len(self.pubmed_calls),
            "seconds": round(sum(c["timing_seconds"]["total"] for c in self.pubmed_calls), 2),
            "search_and_download_seconds": round(
                sum(c["timing_seconds"]["search_and_download"] for c in self.pubmed_calls), 2),
            "summarization_seconds": round(
                sum(c["timing_seconds"]["summarization"] for c in self.pubmed_calls), 2),
            "papers_summarized": sum(c.get("llm", {}).get("papers_summarized", 0) for c in self.pubmed_calls),
            "summarizer_tokens": pubmed_tokens,
        }

        return {
            "total_seconds": self.total_seconds,
            "task_seconds_by_agent": task_seconds,
            "agent_llm_tokens": agent_totals,
            "agent_llm_by_agent": by_agent,
            "pubmed_full_text": pubmed,
            # agent LLM calls + PubMed per-paper summaries
            "total_tokens_all_llm_calls": agent_totals["total_tokens"] + pubmed_tokens["total_token_count"],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "model": self.model,
            "lead_model": self.lead_model,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "summary": self.summary(),
            "not_counted": ["Web-search page filter LLM calls (google_search_w3m)",
                            "Expert knowledge-base LLM and embedding calls (scientist_tool, in the agent process)",
                            "BioMarkerKG G-Retriever service (port 8001), including its query embedding"],
            "token_conventions": {
                "agent_llm": "LangChain usage_metadata: output_tokens includes reasoning_tokens; "
                             "total = input + output (thinking tokens that Gemini's OpenAI-compatible "
                             "endpoint reports only in its total are added to output and reasoning)",
                "pubmed_summarizer": "Gemini usage_metadata: candidates_token_count excludes "
                                     "thoughts_token_count; total = prompt + candidates + thoughts",
            },
            "tasks": self.tasks,
            "pubmed_calls": self.pubmed_calls,
            "llm_calls": self.llm_calls,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)


class LLMUsageCallback(AsyncCallbackHandler):
    """Records token usage and wall time of every LLM call made through the model it is attached to."""

    def __init__(self, tracker: UsageTracker):
        self.tracker = tracker
        self._runs: Dict[Any, tuple] = {}

    @staticmethod
    def _requested_model(kwargs) -> Optional[str]:
        params = kwargs.get("invocation_params") or {}
        return params.get("model") or params.get("model_name")

    async def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        self._runs[run_id] = (time.perf_counter(), _current_label.get(), self._requested_model(kwargs))

    async def on_llm_start(self, serialized, prompts, *, run_id, **kwargs) -> None:
        self._runs[run_id] = (time.perf_counter(), _current_label.get(), self._requested_model(kwargs))

    async def on_llm_end(self, response: LLMResult, *, run_id, **kwargs) -> None:
        start, label, requested = self._runs.pop(run_id, (None, _current_label.get(), None))
        usage, model = {}, None
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                if message is not None and getattr(message, "usage_metadata", None):
                    usage = message.usage_metadata
                    model = (message.response_metadata or {}).get("model_name")
                    break
        input_tokens, output_tokens = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        reasoning_tokens = (usage.get("output_token_details") or {}).get("reasoning", 0)
        # Gemini's OpenAI-compatible endpoint leaves thinking tokens out of output and its details
        hidden = total_tokens - input_tokens - output_tokens
        if hidden > 0 and not reasoning_tokens:
            reasoning_tokens = hidden
            output_tokens += hidden
        self.tracker.record_llm_call({
            "agent": label,
            "model": model or requested or self.tracker.model,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seconds": round(time.perf_counter() - start, 2) if start else None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
        })

    async def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        start, label, requested = self._runs.pop(run_id, (None, _current_label.get(), None))
        self.tracker.record_llm_call({
            "agent": label,
            "model": requested or self.tracker.model,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seconds": round(time.perf_counter() - start, 2) if start else None,
            "error": str(error)[:300],
            **{f: 0 for f in TOKEN_FIELDS},
        })
