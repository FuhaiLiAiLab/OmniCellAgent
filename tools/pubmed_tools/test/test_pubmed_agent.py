"""
Small end-to-end test of the full-text PubMed pipeline on a single query.

Runs PubMed search -> paper download -> text extraction (-> per-paper LLM summary)
through query_medical_research_async and reports, for every returned paper, how much
content came back and whether the LLM summary is valid JSON with all expected fields.

It also reports the pipeline's timing checkpoints (search start -> output end, search start ->
papers downloaded, papers downloaded -> LLM outputs) and the LLM token usage of every paper.

Each run is saved to search_results/<query>/ next to this file (papers.json with the full
paper records, run_info.json with the run settings, timing/token stats and per-paper status).
Re-running the same query overwrites that folder's files.

Usage (from the project root; needs network, and GEMINI/GOOGLE API key when LLM is on):
    python tools/pubmed_tools/test/test_pubmed_agent.py
    python tools/pubmed_tools/test/test_pubmed_agent.py --no-llm
    python tools/pubmed_tools/test/test_pubmed_agent.py --query "Alzheimer disease treatment"
"""

import os
import re
import sys
import json
import time
import asyncio
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from tools.pubmed_tools.query_pubmed_tool import query_medical_research_async
from tools.pubmed_tools.get_papers_info_tools import NUM_PAPERS_TO_DOWNLOAD


DEFAULT_QUERY = "what is the ad treatment"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_results")

# Fields the summarizer prompt in get_paper_summary.py asks for
SUMMARY_KEYS = [
    "objective", "methodology", "key_findings", "authors_conclusions",
    "relevance_to_search_query", "limitations_noted", "analysis_status",
]


def classify_content(content):
    """Return (kind, missing_keys) for a paper's llm_content."""
    if not content:
        return "empty", []
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return "raw_text", []
    if not isinstance(parsed, dict):
        return "raw_text", []
    missing = [k for k in SUMMARY_KEYS if not parsed.get(k)]
    return "llm_summary", missing


def query_to_folder_name(query: str) -> str:
    """Turn a query into a safe folder name, e.g. 'Alzheimer disease treatment' -> 'Alzheimer_disease_treatment'."""
    name = re.sub(r"[^A-Za-z0-9]+", "_", query).strip("_")
    return name[:100] or "empty_query"


def save_results(query, papers, rows, run_info, results_dir):
    """Write papers.json and run_info.json to <results_dir>/<query folder>/ and return that folder."""
    out_dir = os.path.join(results_dir, query_to_folder_name(query))
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "papers.json"), "w", encoding="utf-8") as f:
        json.dump(papers or [], f, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(out_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump({**run_info, "papers": rows}, f, indent=2, ensure_ascii=False, default=str)
    return out_dir


async def run(query: str, num_papers: int, use_llm: bool, timeout: float, session_id: str, stats: dict):
    print(f"Query:          {query!r}")
    print(f"Papers wanted:  {num_papers}")
    print(f"LLM processing: {'on' if use_llm else 'off'}")
    print(f"Session id:     {session_id}\n")

    start = time.time()
    papers = await asyncio.wait_for(
        query_medical_research_async(
            query=query,
            top_k=num_papers,
            use_llm_processing=use_llm,
            max_concurrent=10,
            session_id=session_id,
            stats_out=stats,
        ),
        timeout=timeout,
    )
    elapsed = time.time() - start

    if isinstance(papers, str):
        print(f"FAIL: pipeline returned an error: {papers}")
        return None, elapsed
    return papers, elapsed


def report(papers, elapsed, num_papers, use_llm):
    """Print a per-paper table; return (ok, rows) where rows are saved to run_info.json."""
    counts = {"llm_summary": 0, "raw_text": 0, "empty": 0}
    incomplete = 0
    rows = []

    print(f"{'#':>2}  {'kind':<11} {'fmt':<3} {'chars':>6} {'in_tok':>7} {'out_tok':>7} "
          f"{'think':>6} {'total':>7} {'llm_s':>6}  title")
    for i, paper in enumerate(papers, 1):
        content = paper.get("llm_content")
        kind, missing = classify_content(content)
        counts[kind] += 1
        if missing:
            incomplete += 1
        usage = paper.get("llm_usage") or {}
        rows.append({
            "rank": i, "title": paper.get("title"), "doi": paper.get("doi"),
            "pmid": paper.get("pmid"), "kind": kind, "file_format": paper.get("file_format"),
            "chars": len(content or ""), "missing_fields": missing, "llm_usage": usage or None,
        })
        title = (paper.get("title") or "N/A")[:60]
        print(f"{i:>2}  {kind:<11} {paper.get('file_format') or '-':<3} {len(content or ''):>6} "
              f"{usage.get('prompt_token_count', '-'):>7} {usage.get('candidates_token_count', '-'):>7} "
              f"{usage.get('thoughts_token_count', '-'):>6} {usage.get('total_token_count', '-'):>7} "
              f"{usage.get('llm_seconds', '-'):>6}  {title}")
        print(f"    DOI: {paper.get('doi') or '-'} | PMID: {paper.get('pmid') or '-'} | {paper.get('year') or ''}")
        if missing:
            print(f"    missing summary fields: {', '.join(missing)}")

    print(f"\nReturned {len(papers)}/{num_papers} papers in {elapsed:.1f}s")
    print(f"  LLM summaries: {counts['llm_summary']} ({incomplete} incomplete)")
    print(f"  raw text:      {counts['raw_text']}")
    print(f"  no content:    {counts['empty']}")

    ok = len(papers) > 0 and len(papers) <= num_papers
    if use_llm:
        # Raw text in LLM mode means the summarizer failed and the PDF path fell back
        ok = ok and counts["llm_summary"] > 0
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return ok, rows


def print_stats(stats):
    """Print the pipeline's timing checkpoints, download counts and token totals."""
    if not stats:
        return
    t, d, llm = stats["timing_seconds"], stats["downloads"], stats["llm"]
    print("\nTiming (s)")
    print(f"  search start -> output end:              {t['total']:>8.1f}")
    print(f"  search start -> papers downloaded:       {t['search_and_download']:>8.1f}"
          f"   (PubMed metadata search: {t['of_which_pubmed_metadata_search']:.1f})")
    print(f"  papers downloaded -> LLM outputs:        {t['summarization']:>8.1f}")
    print(f"Downloads: {d['downloaded']} new / {d['download_attempts']} attempts, {d['cache_hits']} cache hits")
    if llm["enabled"]:
        tot = llm["tokens_total"]
        print(f"LLM ({llm['model']}): {llm['papers_summarized']} papers summarized, "
              f"{llm['papers_summarized_but_discarded']} of them discarded to reach top_k")
        print(f"  tokens total: {tot['total_token_count']:,} (input {tot['prompt_token_count']:,}, "
              f"output {tot['candidates_token_count']:,}, thinking {tot['thoughts_token_count']:,})")
        print(f"  tokens on discarded papers: {llm['tokens_discarded_papers']['total_token_count']:,}")


def main():
    parser = argparse.ArgumentParser(description="Test the full-text PubMed pipeline on one query")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--num-papers", type=int, default=NUM_PAPERS_TO_DOWNLOAD)
    parser.add_argument("--no-llm", action="store_true", help="skip per-paper LLM summaries (raw text only)")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds before the run is aborted")
    parser.add_argument("--results-dir", default=RESULTS_DIR,
                        help="parent folder for the per-query result folders")
    args = parser.parse_args()

    use_llm = not args.no_llm
    session_id = time.strftime("test_pubmed_%Y%m%d-%H%M%S")
    run_info = {
        "query": args.query,
        "session_id": session_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_papers_requested": args.num_papers,
        "use_llm_processing": use_llm,
    }

    papers, elapsed, rows, ok = None, None, [], False
    stats = {}
    try:
        papers, elapsed = asyncio.run(run(args.query, args.num_papers, use_llm, args.timeout, session_id, stats))
        if papers is None:
            run_info["status"] = "pipeline_error"
        else:
            ok, rows = report(papers, elapsed, args.num_papers, use_llm)
            print_stats(stats)
            run_info["status"] = "pass" if ok else "fail"
    except asyncio.TimeoutError:
        print(f"FAIL: timed out after {args.timeout:.0f}s")
        run_info["status"] = "timeout"

    run_info["elapsed_seconds"] = round(elapsed, 1) if elapsed is not None else None
    run_info["num_papers_returned"] = len(papers or [])
    run_info["stats"] = stats or None
    out_dir = save_results(args.query, papers, rows, run_info, args.results_dir)
    print(f"Saved results to {out_dir}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
