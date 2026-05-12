#!/usr/bin/env python3
"""
Paper Improvement & Review Pipeline
====================================

Uses the LangGraph agent's `revise_report_from_feedback` to improve reports
based on aggregated AI reviews, then re-runs reviewers on the revised PDFs
to measure improvement.

Usage:
  python benchmark/improve_and_review.py                    # all cases
  python benchmark/improve_and_review.py --case AD          # specific case
  python benchmark/improve_and_review.py --skip-improve     # only re-review
  python benchmark/improve_and_review.py --visualize        # only visualize
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ── Setup paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from agent.langgraph_agent import LangGraphOmniCellAgent

# ── Configuration ────────────────────────────────────────────────────────
SESSIONS_DIR = PROJECT_ROOT / "webapp" / "sessions"
RESULTS_BASE = PROJECT_ROOT / "benchmark" / "ai_review_results"

CASES: dict[str, dict[str, str]] = {
    "AD": dict(
        label="Alzheimer's Disease (Microglia)",
        session="AD-test",
        pdf="report_20260311_140104.pdf",
        md="report_20260311_140104.md",
        query="What are the key dysfunctional genes and pathways in Alzheimers Disease?",
    ),
    "LungCancer": dict(
        label="Lung Adenocarcinoma",
        session="LungCancer-test",
        pdf="report_20260311_142036.pdf",
        md="report_20260311_142036.md",
        query="What are the key dysfunctional genes and pathways in Lung Adenocarcinoma?",
    ),
    "PDAC": dict(
        label="Pancreatic Ductal Adenocarcinoma",
        session="PDAC-test",
        pdf="report_20260311_131601.pdf",
        md="report_20260311_131601.md",
        query="What are the key dysfunctional genes and pathways in Pancreatic Ductal Adenocarcinoma?",
    ),
}

# Reviewers to run on revised reports (API-based reviewers only for speed)
# Full list: ["apr", "cycle", "openrev", "sea", "litllm", "reviewadvisor"]
REVIEWERS = ["apr", "litllm", "openrev"]


def _resolve_latest_report(case: dict) -> Path:
    """Resolve the latest report_*.md in the session dir.
    Updates case['md']/case['pdf'] in place when auto-discovery succeeds."""
    d = SESSIONS_DIR / case["session"]
    md = d / case["md"] if case.get("md") else None
    if md is None or not md.exists():
        candidates = sorted(
            (p for p in d.glob("report_*.md") if "_static" not in p.name and "-revised" not in p.name),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"No report_*.md found in {d}")
        md = candidates[0]
        case["md"] = md.name
        pdf = md.with_suffix(".pdf")
        if pdf.exists():
            case["pdf"] = pdf.name
    return md


def _read_original_report(case: dict) -> str:
    """Read the original markdown report."""
    md_path = _resolve_latest_report(case)
    return md_path.read_text()


def _aggregate_feedback(case_key: str) -> str:
    """
    Aggregate all review feedback for a case into a single structured
    revision instruction document.
    """
    review_dir = RESULTS_BASE / case_key
    if not review_dir.exists():
        raise FileNotFoundError(f"No reviews found for {case_key}")

    feedback_parts = []

    # Priority: APR meta-review (most comprehensive)
    apr_meta = review_dir / "apr" / "meta_review.md"
    if apr_meta.exists():
        feedback_parts.append(f"## AI Peer Review Meta-Review\n\n{apr_meta.read_text()}")

    # Add CycleReviewer feedback
    cycle_review = review_dir / "cycle" / "review.md"
    if cycle_review.exists():
        feedback_parts.append(f"## CycleReviewer Feedback\n\n{cycle_review.read_text()}")

    # Add OpenReviewer feedback
    openrev_review = review_dir / "openrev" / "review.md"
    if openrev_review.exists():
        feedback_parts.append(f"## OpenReviewer Feedback\n\n{openrev_review.read_text()}")

    # Add SEA feedback
    sea_review = review_dir / "sea" / "review.md"
    if sea_review.exists():
        feedback_parts.append(f"## SEA Framework Feedback\n\n{sea_review.read_text()}")

    # Add LitLLM (citation analysis)
    litllm_review = review_dir / "litllm" / "related_works.md"
    if litllm_review.exists():
        feedback_parts.append(f"## LitLLM Citation Analysis\n\n{litllm_review.read_text()}")

    if not feedback_parts:
        raise ValueError(f"No review feedback found for {case_key}")

    aggregated = (
        "# Aggregated Reviewer Feedback\n\n"
        "Please address the following issues identified by multiple AI reviewers. "
        "Focus on:\n"
        "1. Methodological transparency (add details on normalization, batch correction)\n"
        "2. Removing or flagging fabricated/future-dated citations\n"
        "3. Adding decision criteria for validation experiments\n"
        "4. Acknowledging limitations of bulk RNA-seq (cell-type composition)\n"
        "5. Improving presentation and clarity where noted\n\n"
        + "\n\n---\n\n".join(feedback_parts)
    )
    return aggregated


async def improve_report(case_key: str, overwrite: bool = False) -> Path | None:
    """
    Use the LangGraph agent to revise a report based on aggregated feedback.
    Returns the path to the revised PDF/MD.
    """
    case = CASES[case_key]
    session_dir = SESSIONS_DIR / case["session"]

    # Check if revised report already exists
    existing_revised = list(session_dir.glob("*-revised.pdf")) + list(
        session_dir.glob("*-revised.md")
    )
    if existing_revised and not overwrite:
        print(f"    ⏭️  {case_key}: revised report exists (use --overwrite)")
        return existing_revised[0]

    print(f"    📝 Reading original report...")
    original_report = _read_original_report(case)

    print(f"    📋 Aggregating reviewer feedback...")
    feedback = _aggregate_feedback(case_key)

    print(f"    🤖 Calling revision agent...")
    # Create agent instance with the case's session ID
    agent = LangGraphOmniCellAgent(
        model_name="gemini-2.5-pro",
        session_id=case["session"],
    )

    result = await agent.revise_report_from_feedback(
        original_report=original_report,
        feedback=feedback,
        query=case["query"],
    )

    revised_md = result.get("report_path")
    if revised_md and Path(revised_md).exists():
        # Rename to include -revised suffix for clarity
        revised_path = Path(revised_md)
        new_name = revised_path.stem.replace("_revised_", "-revised_") + revised_path.suffix
        final_path = revised_path.parent / new_name
        if not final_path.exists():
            revised_path.rename(final_path)
            revised_md = str(final_path)

        # Also rename the PDF if it exists
        pdf_path = revised_path.with_suffix(".pdf")
        if pdf_path.exists():
            new_pdf = final_path.with_suffix(".pdf")
            if not new_pdf.exists():
                pdf_path.rename(new_pdf)

        print(f"    ✅ Revised report saved: {revised_md}")
        return Path(revised_md)
    else:
        print(f"    ❌ Revision failed")
        return None


# ═══════════════════════════════════════════════════════════════════════
# Self-contained reviewer implementations (API-based, no torch/transformers)
# ═══════════════════════════════════════════════════════════════════════

REPORT_CONTEXT = (
    "The document you are reviewing is an AI-generated computational "
    "biology research report produced by OmniCellAgent, an automated "
    "multi-agent pipeline. The pipeline mines omics databases (GEO/"
    "ArrayExpress), knowledge graphs, and the biomedical literature "
    "to produce an end-to-end analysis."
)

SCORE_INSTRUCTION = (
    "\n\nFinally, provide an overall quality score on a 1-10 scale "
    "(1 = fundamentally flawed, 5 = acceptable with major revisions, "
    "10 = exceptional). Output the score on its own line in EXACTLY "
    "this format:\n  OVERALL_SCORE: <integer>\n"
)

import re


def _extract_score(text: str) -> int | None:
    """Parse OVERALL_SCORE: N from reviewer output text."""
    m = re.search(r'OVERALL_SCORE\s*[:=]\s*(\d+)', text)
    if m:
        return max(1, min(10, int(m.group(1))))
    m = re.search(r'\*\*Rating:?\*\*\s*\n?\s*(\d+)', text)
    if m:
        return max(1, min(10, int(m.group(1))))
    m = re.search(r'Rating\s*\(?\s*(\d+)', text)
    if m:
        return max(1, min(10, int(m.group(1))))
    return None


def _run_apr_reviewer(text: str, out: Path) -> dict:
    """AI Peer Review using direct OpenAI API calls (multi-model ensemble)."""
    from openai import OpenAI

    system_prompt = (
        "You are a computational biologist and expert in multi-omics "
        "data analysis, single-cell transcriptomics, pathway enrichment, "
        "and biomarker discovery. You review AI-generated research "
        "reports (not journal papers)."
    )
    review_prompt = (
        "Please provide a thorough and critical review of this AI-generated "
        "computational biology research report. In your review:\n"
        "1. Summarise the report's objectives and main findings.\n"
        "2. Evaluate the statistical and methodological rigour.\n"
        "3. Assess whether the hypotheses are mechanistically plausible.\n"
        "4. Evaluate the proposed validation experiments.\n"
        "5. Check that all cited references are real (flag any fabricated ones).\n"
        "6. Provide specific, actionable suggestions for improvement.\n\n"
        + REPORT_CONTEXT + SCORE_INSTRUCTION +
        f"\n\nHere is the report to review:\n\n{text[:30000]}"
    )

    reviews = {}
    client = OpenAI()

    # Use GPT-4o for first review
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": review_prompt},
            ],
            temperature=0.3,
        )
        reviews["gpt4o"] = resp.choices[0].message.content
        (out / "review_gpt4o.md").write_text(reviews["gpt4o"])
    except Exception as e:
        reviews["gpt4o"] = f"Error: {e}"

    # Use GPT-4o-mini for second review (different perspective, lower cost)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt + "\nFocus especially on methodological concerns."},
                {"role": "user", "content": review_prompt},
            ],
            temperature=0.5,
        )
        reviews["gpt4o_mini"] = resp.choices[0].message.content
        (out / "review_gpt4o_mini.md").write_text(reviews["gpt4o_mini"])
    except Exception as e:
        reviews["gpt4o_mini"] = f"Error: {e}"

    # Generate meta-review
    meta_prompt = (
        "The following are independent reviews of an AI-generated "
        "computational biology research report. Please synthesise them into "
        "a meta-review that highlights consensus, key concerns, and "
        "actionable recommendations.\n\n" +
        "\n\n---\n\n".join([f"## {k}\n\n{v}" for k, v in reviews.items()]) +
        SCORE_INSTRUCTION
    )
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You synthesise peer reviews into meta-reviews."},
                {"role": "user", "content": meta_prompt},
            ],
            temperature=0.2,
        )
        meta = resp.choices[0].message.content
        (out / "meta_review.md").write_text(meta)
    except Exception as e:
        meta = f"Meta-review error: {e}"

    score = _extract_score(meta)
    if score is None:
        indiv = [_extract_score(v) for v in reviews.values()]
        indiv = [s for s in indiv if s is not None]
        if indiv:
            score = round(sum(indiv) / len(indiv))

    # Save config for reference
    cfg = {"models": list(reviews.keys()), "system": system_prompt}
    (out / "_config.json").write_text(json.dumps(cfg, indent=2))

    return {"reviews": {m: len(v) for m, v in reviews.items()}, "meta_review_chars": len(meta), "score": score}


def _run_litllm_reviewer(text: str, out: Path) -> dict:
    """LitLLM citation analysis via OpenAI API."""
    from openai import OpenAI

    system = (
        "You are an expert research assistant specialising in "
        "computational biology, single-cell transcriptomics, and "
        "biomarker discovery. " + REPORT_CONTEXT
    )
    user = (
        "Below is an AI-generated computational biology research report. "
        "Your task is to:\n"
        "1. Identify the key claims and hypotheses in this report.\n"
        "2. Evaluate whether the cited references are real and relevant.\n"
        "3. Suggest the most important missing references that should be cited.\n"
        "4. Write a related-works section that contextualises this report.\n"
        "5. Assess the overall quality of the literature coverage."
        + SCORE_INSTRUCTION +
        f"\n\nReport:\n\n{text[:12000]}"
    )
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=3000,
        temperature=0.2,
    )
    response = resp.choices[0].message.content
    (out / "related_works.md").write_text(response)
    score = _extract_score(response)
    return {"output_chars": len(response), "score": score}


def _run_openrev_reviewer(text: str, out: Path) -> dict:
    """OpenReviewer via OpenRouter API."""
    from openai import OpenAI

    system = """You are an expert reviewer for computational biology research. """ + REPORT_CONTEXT + """

Reviewer guidelines:
1. Read the report carefully.
2. Consider: Objective, Strong points, Weak points.
3. Write your review including: summary, strong/weak points, rating.

Your response must contain these sections:
## Summary
## Soundness (1-4)
## Presentation (1-4)
## Contribution (1-4)
## Strengths
## Weaknesses
## Questions
## Rating (1-10)
"""
    user = f"Review the following AI-generated research report:\n\n{text[:40_000]}"

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if os.environ.get("OPENROUTER_API_KEY"):
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        model = "meta-llama/llama-3.1-70b-instruct"
    else:
        client = OpenAI(api_key=api_key)
        model = "gpt-4o"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
    )
    review = resp.choices[0].message.content
    (out / "review.md").write_text(review)
    score = _extract_score(review)
    return {"review_chars": len(review), "score": score}


REVIEWER_DISPATCH = {
    "apr": _run_apr_reviewer,
    "litllm": _run_litllm_reviewer,
    "openrev": _run_openrev_reviewer,
}


def run_reviews_on_revised(case_key: str, overwrite: bool = False) -> dict:
    """
    Run API-based AI reviewers on the revised report.
    """
    import time

    case = CASES[case_key]
    session_dir = SESSIONS_DIR / case["session"]

    # Find the revised report
    revised_mds = sorted(session_dir.glob("*-revised*.md"), reverse=True)
    revised_pdfs = sorted(session_dir.glob("*-revised*.pdf"), reverse=True)

    if not revised_mds and not revised_pdfs:
        print(f"    ⚠️  No revised report found for {case_key}")
        return {}

    # Prefer MD over PDF for text extraction
    if revised_mds:
        report_path = revised_mds[0]
        text = report_path.read_text()
    else:
        report_path = revised_pdfs[0]
        import subprocess as sp
        r = sp.run(["pdftotext", str(report_path), "-"], capture_output=True, text=True)
        text = r.stdout if r.returncode == 0 else ""

    if not text:
        print(f"    ⚠️  Could not extract text from revised report")
        return {}

    results: dict[str, Any] = {}
    out_base = RESULTS_BASE / f"{case_key}_revised"
    out_base.mkdir(parents=True, exist_ok=True)

    for rkey in REVIEWERS:
        if rkey not in REVIEWER_DISPATCH:
            print(f"        ⏭️  {rkey}: not available (API-only mode)")
            continue

        out = out_base / rkey
        out.mkdir(parents=True, exist_ok=True)

        meta_file = out / "result.json"
        if meta_file.exists() and not overwrite:
            print(f"        ⏭️  {rkey}: cached")
            try:
                results[rkey] = json.loads(meta_file.read_text())
            except Exception:
                pass
            continue

        print(f"        🔬 {rkey} …", end=" ", flush=True)
        t0 = time.time()

        try:
            res = REVIEWER_DISPATCH[rkey](text, out)
            res["success"] = True
            res["elapsed_seconds"] = round(time.time() - t0, 1)
            print(f"✅ {res['elapsed_seconds']}s (score: {res.get('score', '?')})")
        except Exception as e:
            res = {
                "success": False,
                "error": str(e),
                "elapsed_seconds": round(time.time() - t0, 1),
            }
            print(f"❌ {e}")

        meta_file.write_text(json.dumps(res, indent=2, default=str))
        results[rkey] = res

    return results


def collect_scores() -> pd.DataFrame:
    """Collect scores from both original and revised reviews into a DataFrame."""
    rows = []

    for case_key in CASES.keys():
        # Original scores
        for rkey in REVIEWERS:
            result_file = RESULTS_BASE / case_key / rkey / "result.json"
            if result_file.exists():
                try:
                    res = json.loads(result_file.read_text())
                    score = res.get("score")
                    if score is not None:
                        rows.append({
                            "case": case_key,
                            "reviewer": rkey,
                            "version": "original",
                            "score": int(score),
                        })
                except Exception:
                    pass

        # Revised scores
        for rkey in REVIEWERS:
            result_file = RESULTS_BASE / f"{case_key}_revised" / rkey / "result.json"
            if result_file.exists():
                try:
                    res = json.loads(result_file.read_text())
                    score = res.get("score")
                    if score is not None:
                        rows.append({
                            "case": case_key,
                            "reviewer": rkey,
                            "version": "revised",
                            "score": int(score),
                        })
                except Exception:
                    pass

    return pd.DataFrame(rows)


def visualize_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Create comparison visualization of original vs revised scores."""
    if df.empty:
        print("⚠️  No scores to visualize")
        return

    # Pivot for grouped bar chart
    pivot_orig = df[df["version"] == "original"].pivot(
        index="reviewer", columns="case", values="score"
    )
    pivot_rev = df[df["version"] == "revised"].pivot(
        index="reviewer", columns="case", values="score"
    )

    cases = list(CASES.keys())
    reviewers = sorted(df["reviewer"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, case in zip(axes, cases):
        orig_scores = []
        rev_scores = []
        labels = []

        for rev in reviewers:
            orig = pivot_orig.loc[rev, case] if rev in pivot_orig.index and case in pivot_orig.columns else None
            rev_score = pivot_rev.loc[rev, case] if rev in pivot_rev.index and case in pivot_rev.columns else None

            if orig is not None or rev_score is not None:
                labels.append(rev)
                orig_scores.append(orig if orig is not None else 0)
                rev_scores.append(rev_score if rev_score is not None else 0)

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width / 2, orig_scores, width, label="Original", color="#4C72B0", alpha=0.8)
        bars2 = ax.bar(x + width / 2, rev_scores, width, label="Revised", color="#55A868", alpha=0.8)

        ax.set_xlabel("Reviewer")
        ax.set_ylabel("Score (1-10)")
        ax.set_title(f"{CASES[case]['label']}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylim(0, 10)
        ax.axhline(y=5, color="gray", linestyle="--", alpha=0.5, label="Accept threshold")
        ax.legend(loc="upper right")

        # Add value labels
        for bar in bars1:
            if bar.get_height() > 0:
                ax.annotate(
                    f"{int(bar.get_height())}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9,
                )
        for bar in bars2:
            if bar.get_height() > 0:
                ax.annotate(
                    f"{int(bar.get_height())}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9,
                )

    plt.suptitle("AI Review Scores: Original vs Revised Reports", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"📊 Visualization saved: {output_path}")

    # Also create a summary table
    summary_path = output_path.with_suffix(".csv")
    df.to_csv(summary_path, index=False)
    print(f"📊 Summary CSV: {summary_path}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("📊 SCORE COMPARISON: Original → Revised")
    print("=" * 70)

    for case in cases:
        print(f"\n  {CASES[case]['label']}")
        print(f"  {'-' * 50}")
        for rev in reviewers:
            orig = None
            revised = None
            try:
                orig = pivot_orig.loc[rev, case] if rev in pivot_orig.index else None
            except KeyError:
                pass
            try:
                revised = pivot_rev.loc[rev, case] if rev in pivot_rev.index else None
            except KeyError:
                pass

            if orig is not None or revised is not None:
                orig_str = f"{int(orig)}" if orig is not None else "—"
                rev_str = f"{int(revised)}" if revised is not None else "—"
                delta = ""
                if orig is not None and revised is not None:
                    d = int(revised) - int(orig)
                    delta = f" ({'+' if d > 0 else ''}{d})"
                print(f"    {rev:<15s}  {orig_str:>3s} → {rev_str:>3s}{delta}")

    print("\n" + "=" * 70)


async def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--case", nargs="+", choices=list(CASES),
        default=list(CASES), help="Case(s) to process",
    )
    parser.add_argument("--skip-improve", action="store_true", help="Skip improvement step")
    parser.add_argument("--skip-review", action="store_true", help="Skip re-review step")
    parser.add_argument("--visualize", action="store_true", help="Only visualize existing results")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    parser.add_argument(
        "--session-suffix", default="-test",
        help="Suffix to append to case keys to form session id (default '-test'). "
             "Use '-test-2' to point at re-runs.",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Override RESULTS_BASE (default benchmark/ai_review_results). "
             "Set to ai_review_results_v2 to keep v1 outputs intact.",
    )
    args = parser.parse_args()

    # Apply session-suffix and results-dir overrides BEFORE any case reads happen.
    global RESULTS_BASE
    if args.session_suffix != "-test":
        for ck, cfg in CASES.items():
            cfg["session"] = f"{ck}{args.session_suffix}"
            cfg["pdf"] = None
            cfg["md"] = None
    if args.results_dir:
        RESULTS_BASE = Path(args.results_dir)
        if not RESULTS_BASE.is_absolute():
            RESULTS_BASE = PROJECT_ROOT / "benchmark" / args.results_dir
        RESULTS_BASE.mkdir(parents=True, exist_ok=True)
        print(f"📁 Using results dir: {RESULTS_BASE}")

    print("\n" + "=" * 70)
    print("🔬 Paper Improvement & Review Pipeline")
    print("=" * 70)

    if args.visualize:
        df = collect_scores()
        visualize_comparison(df, RESULTS_BASE / "score_comparison.png")
        return

    # Step 1: Improve reports
    if not args.skip_improve:
        print("\n📝 Phase 1: Improving Reports from AI Feedback")
        print("-" * 50)
        for case_key in args.case:
            print(f"\n  📄 {CASES[case_key]['label']}")
            try:
                await improve_report(case_key, args.overwrite)
            except Exception as e:
                print(f"    ❌ Error: {e}")

    # Step 2: Re-run reviews on revised reports
    if not args.skip_review:
        print("\n\n🔬 Phase 2: Re-running AI Reviews on Revised Reports")
        print("-" * 50)
        for case_key in args.case:
            print(f"\n  📄 {CASES[case_key]['label']}")
            try:
                run_reviews_on_revised(case_key, args.overwrite)
            except Exception as e:
                print(f"    ❌ Error: {e}")

    # Step 3: Visualize comparison
    print("\n\n📊 Phase 3: Score Comparison")
    print("-" * 50)
    df = collect_scores()
    visualize_comparison(df, RESULTS_BASE / "score_comparison.png")


if __name__ == "__main__":
    asyncio.run(main())
