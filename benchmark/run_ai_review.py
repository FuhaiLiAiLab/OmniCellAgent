#!/usr/bin/env python3
"""
Unified AI Review Benchmark — OmniCellAgent Reports
====================================================

Runs multiple automated-review frameworks on the three OmniCellAgent
case-study reports and writes structured results.

Supported reviewers:

  Key             Framework         Reference
  ──────────────  ────────────────  ──────────────────────────────────
  apr             AI Peer Review    poldrack/ai-peer-review
  cycle           CycleReviewer     arxiv:2411.00816 (ICLR 2025)
  openrev         OpenReviewer      arxiv:2412.11948 (NAACL 2025)
  sea             SEA Framework     arxiv:2407.12857 (EMNLP 2024)
  reviewadvisor   ReviewAdvisor     arxiv:2202.00176
  litllm          LitLLM            arxiv:2402.01788 (2024)

Usage
-----
  python benchmark/run_ai_review.py                        # all reviewers
  python benchmark/run_ai_review.py --reviewer apr cycle    # specific
  python benchmark/run_ai_review.py --case AD --overwrite   # one case
  python benchmark/run_ai_review.py --list                  # show status
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from vllm import LLM, SamplingParams

from ai_peer_review.review import (
    generate_meta_review,
    process_paper,
    save_concerns_as_csv,
)
from ai_researcher import CycleReviewer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPOS_DIR = PROJECT_ROOT / "benchmark" / "repos"

load_dotenv(PROJECT_ROOT / ".env")

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

SESSIONS_DIR = PROJECT_ROOT / "webapp" / "sessions"

CASES: dict[str, dict[str, str]] = {
    "AD": dict(
        label="Alzheimer's Disease (Microglia)",
        session="AD-test",
        pdf="report_20260311_140104.pdf",
        md="report_20260311_140104.md",
    ),
    "LungCancer": dict(
        label="Lung Adenocarcinoma",
        session="LungCancer-test",
        pdf="report_20260311_142036.pdf",
        md="report_20260311_142036.md",
    ),
    "PDAC": dict(
        label="Pancreatic Ductal Adenocarcinoma",
        session="PDAC-test",
        pdf="report_20260311_131601.pdf",
        md="report_20260311_131601.md",
    ),
}

RESULTS_BASE = PROJECT_ROOT / "benchmark" / "ai_review_results"

# ── Report-aware framing (shared across all reviewers) ──────────────
# This describes what the reviewers are evaluating so every prompt
# treats the input as an AI-generated research report, not a
# conference submission paper.
REPORT_CONTEXT = (
    "The document you are reviewing is an AI-generated computational "
    "biology research report produced by OmniCellAgent, an automated "
    "multi-agent pipeline. The pipeline mines omics databases (GEO/"
    "ArrayExpress), knowledge graphs, and the biomedical literature "
    "to produce an end-to-end analysis. A typical report includes:\n"
    "  1. Omics data analysis (DEG tables, volcano plots)\n"
    "  2. Knowledge-graph neighbourhood analysis\n"
    "  3. Literature-validated targets\n"
    "  4. Pathway enrichment (KEGG, Reactome, GO)\n"
    "  5. Gene-anchored mechanistic hypotheses scored 0-100\n"
    "  6. Minimal validation experiment designs\n\n"
    "Evaluate it as a *research report*, NOT as a peer-reviewed "
    "journal paper. Focus on whether the computational analysis is "
    "sound, the hypotheses are mechanistically plausible and testable, "
    "the cited references are real, and the proposed experiments are "
    "well-designed."
)

SCORE_INSTRUCTION = (
    "\n\nFinally, provide an overall quality score on a 1-10 scale "
    "(1 = fundamentally flawed, 5 = acceptable with major revisions, "
    "10 = exceptional). Output the score on its own line in EXACTLY "
    "this format:\n  OVERALL_SCORE: <integer>\n"
)


def _extract_score(text: str) -> int | None:
    """Parse OVERALL_SCORE: N from reviewer output text."""
    m = re.search(r'OVERALL_SCORE\s*[:=]\s*(\d+)', text)
    if m:
        return max(1, min(10, int(m.group(1))))
    # Fallback: look for Rating: N patterns (OpenReviewer, SEA)
    m = re.search(r'\*\*Rating:?\*\*\s*\n?\s*(\d+)', text)
    if m:
        return max(1, min(10, int(m.group(1))))
    m = re.search(r'Rating\s*\(?\s*(\d+)', text)
    if m:
        return max(1, min(10, int(m.group(1))))
    return None


REVIEWERS = [
    ("apr",           "AI Peer Review",  "poldrack/ai-peer-review"),
    ("cycle",         "CycleReviewer",   "arxiv:2411.00816 (ICLR 2025)"),
    ("openrev",       "OpenReviewer",    "arxiv:2412.11948 (NAACL 2025)"),
    ("sea",           "SEA Framework",   "arxiv:2407.12857 (EMNLP 2024)"),
    ("reviewadvisor", "ReviewAdvisor",   "arxiv:2202.00176"),
    ("litllm",        "LitLLM",          "arxiv:2402.01788 (2024)"),
]

_REVIEWER_NAMES: dict[str, str] = {k: n for k, n, _ in REVIEWERS}


# ═══════════════════════════════════════════════════════════════════════
# Report helpers
# ═══════════════════════════════════════════════════════════════════════


def _report_text(case: dict) -> str:
    d = SESSIONS_DIR / case["session"]
    md = d / case["md"]
    if md.exists():
        return md.read_text()
    pdf = d / case["pdf"]
    if not pdf.exists():
        raise FileNotFoundError(f"No report in {d}")
    import subprocess
    r = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True)
    if r.returncode == 0:
        return r.stdout
    raise RuntimeError(f"Cannot extract text from {pdf}")


def _report_path(case: dict) -> Path:
    d = SESSIONS_DIR / case["session"]
    for f in (case["pdf"], case["md"]):
        p = d / f
        if p.exists():
            return p
    raise FileNotFoundError(f"No report in {d}")


# ═══════════════════════════════════════════════════════════════════════
# Shared: HuggingFace model cache
# ═══════════════════════════════════════════════════════════════════════

_HF_CACHE: dict[str, tuple] = {}


def _hf_generate(
    model_id: str,
    system_prompt: str,
    user_text: str,
    max_new_tokens: int = 4096,
    temperature: float = 0.3,
) -> str:
    if model_id not in _HF_CACHE:
        tok = AutoTokenizer.from_pretrained(model_id)
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id, device_map={"": 0}, quantization_config=bnb_cfg,
        )
        _HF_CACHE[model_id] = (tok, mdl)
    tok, mdl = _HF_CACHE[model_id]
    ids = tok.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_text[:80_000]}],
        return_tensors="pt", add_generation_prompt=True,
    ).to(mdl.device)
    out = mdl.generate(
        ids, max_new_tokens=max_new_tokens,
        do_sample=True, temperature=temperature, top_p=0.95,
    )
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════════════
# 1. AI Peer Review  (poldrack/ai-peer-review)
# ═══════════════════════════════════════════════════════════════════════

_APR_CFG: dict = {
    "api_keys": {},
    "prompts": {
        "system": (
            "You are a computational biologist and expert in multi-omics "
            "data analysis, single-cell transcriptomics, pathway enrichment, "
            "and biomarker discovery. You review AI-generated research "
            "reports (not journal papers)."
        ),
        "review": (
            "You are a computational biologist reviewing an AI-generated "
            "research report produced by OmniCellAgent, an automated "
            "multi-agent pipeline that mines omics databases, knowledge "
            "graphs, and the biomedical literature.\n\n"
            "Please provide a thorough and critical review. In your review:\n"
            "1. Summarise the report's objectives and main findings.\n"
            "2. Evaluate the statistical and methodological rigour of the "
            "   omics analysis pipeline.\n"
            "3. Assess whether the gene-pathway-phenotype hypotheses are "
            "   mechanistically plausible and well-supported by the data.\n"
            "4. Evaluate the proposed validation experiments: are the "
            "   controls appropriate, readouts quantitative, and decision "
            "   criteria clear?\n"
            "5. Check that all cited references are real (flag fabricated ones).\n"
            "6. Provide specific, actionable suggestions for improvement.\n\n"
            "Here is the report to review:\n\n{paper_text}\n\n"
            "Finally, provide an overall quality score on a 1-10 scale "
            "(1 = fundamentally flawed, 5 = acceptable with major revisions, "
            "10 = exceptional). Output the score on its own line in EXACTLY "
            "this format:\n  OVERALL_SCORE: <integer>"
        ),
        "metareview": (
            "The following are independent reviews of an AI-generated "
            "computational biology research report (produced by an automated "
            "multi-agent pipeline). Please synthesise them into a meta-review."
            "\n\nReviews:\n\n{reviews_text}\n\n"
            "Finally, provide an overall quality score on a 1-10 scale "
            "(1 = fundamentally flawed, 5 = acceptable with major revisions, "
            "10 = exceptional). Output the score on its own line in EXACTLY "
            "this format:\n  OVERALL_SCORE: <integer>"
        ),
    },
}


def run_apr(text: str, path: Path, out: Path, **kw) -> dict:
    models = kw.get("models", ["gemini-2.5-pro", "gpt4-o1", "gpt4-o3-mini"])
    cfg = out / "_config.json"
    cfg.write_text(json.dumps(_APR_CFG, indent=2))
    reviews = process_paper(str(path), models, config_file=str(cfg))
    for m, t in reviews.items():
        (out / f"review_{m}.md").write_text(t)
    meta, nato = generate_meta_review(reviews, config_file=str(cfg))
    (out / "meta_review.md").write_text(meta)
    try:
        save_concerns_as_csv(meta, out)
    except Exception:
        pass
    # Extract score from meta-review first, fall back to averaging individual scores
    score = _extract_score(meta)
    if score is None:
        indiv = [_extract_score(t) for t in reviews.values()]
        indiv = [s for s in indiv if s is not None]
        if indiv:
            score = round(sum(indiv) / len(indiv))
    return {
        "reviews": {m: len(t) for m, t in reviews.items()},
        "meta_review_chars": len(meta),
        "reviewer_map": nato,
        "score": score,
    }


# ═══════════════════════════════════════════════════════════════════════
# 2. CycleReviewer  (ai_researcher — vLLM, 8B/70B/123B)
# ═══════════════════════════════════════════════════════════════════════

_cycle_instance: CycleReviewer | None = None


def run_cycle(text: str, path: Path, out: Path, **kw) -> dict:
    global _cycle_instance
    sz = kw.get("model_size", "8B")
    if _cycle_instance is None:
        _cycle_instance = CycleReviewer(
            custom_model_name="WestlakeNLP/CycleReviewer-ML-Llama-3.1-8B",
            gpu_memory_utilization=0.85,
        )
    results = _cycle_instance.evaluate(text)
    # evaluate() returns a list; take first element
    result = results[0] if isinstance(results, list) else results
    if result is None:
        raise RuntimeError("CycleReviewer returned None (parsing failed)")
    md = f"# CycleReviewer ({sz})\n\n"
    md += f"**Average Rating:** {result.get('avg_rating', 'N/A')}\n"
    md += f"**Decision:** {result.get('paper_decision', 'N/A')}\n\n"
    if result.get("meta_review"):
        md += f"## Meta-Review\n\n{result['meta_review']}\n\n"
    for i, rev in enumerate(result.get("reviews", []), 1):
        md += f"## Reviewer {i}\n\n{rev}\n\n"
    (out / "review.md").write_text(md)
    (out / "raw.json").write_text(json.dumps(result, indent=2, default=str))
    # CycleReviewer avg_rating is already 1-10 scale
    raw = result.get("avg_rating")
    score = max(1, min(10, round(float(raw)))) if raw is not None else None
    return {
        "avg_rating": raw,
        "decision": result.get("paper_decision"),
        "num_reviews": len(result.get("reviews", [])),
        "score": score,
    }


# ═══════════════════════════════════════════════════════════════════════
# 3. OpenReviewer  (maxidl/Llama-OpenReviewer-8B via vLLM or OpenRouter)
# ═══════════════════════════════════════════════════════════════════════

# Adapted from llm_training/generate.py — reframed for report review
_OPENREV_SYSTEM = """You are an expert reviewer for computational biology research. You follow best practices and review AI-generated research reports according to the guidelines below.

""" + REPORT_CONTEXT + """

Reviewer guidelines:
1. Read the report carefully, paying attention to the omics pipeline, knowledge-graph analysis, mechanistic hypotheses, and proposed validation experiments.
2. While reading, consider the following:
    - Objective: What biological question does the report address?
    - Strong points: Is the analysis pipeline sound, are hypotheses testable, are citations real?
    - Weak points: Are there unsupported claims, missing controls, or logical gaps?
    - Be mindful of potential biases (dataset selection, pathway database bias).
3. Answer four key questions:
    - What biological question is addressed?
    - Is the computational pipeline well-motivated?
    - Does the evidence support the hypotheses?
    - What is the potential impact of the findings?
4. Write your review including: summary, strong/weak points, recommendation, supporting arguments, questions, additional feedback.

Your write reviews in markdown format. Your reviews contain the following sections:

# Review

{review_fields}

Your response must only contain the review in markdown format with sections as defined above.
"""

_OPENREV_FIELDS = """## Summary
## Soundness (1-4)
## Presentation (1-4)
## Contribution (1-4)
## Strengths
## Weaknesses
## Questions
## Rating (1-10)
## Confidence (1-5)"""

_OPENREV_MODEL = os.environ.get("OPENREV_MODEL", "maxidl/Llama-OpenReviewer-8B")


def run_openrev(text: str, path: Path, out: Path, **kw) -> dict:
    system = _OPENREV_SYSTEM.format(review_fields=_OPENREV_FIELDS)
    user = f"Review the following AI-generated research report:\n\n{text[:40_000]}"

    if os.environ.get("OPENROUTER_API_KEY"):
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        resp = client.chat.completions.create(
            model=_OPENREV_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            extra_body={"temperature": 0.0},
        )
        review = resp.choices[0].message.content
        mode = "openrouter"
    else:
        # Use HF transformers with 4-bit quantization (fits 24GB GPU)
        review = _hf_generate(
            _OPENREV_MODEL, system, user,
            max_new_tokens=4096, temperature=0.1,
        )
        mode = "hf_4bit"

    (out / "review.md").write_text(review)
    score = _extract_score(review)
    return {"review_chars": len(review), "mode": mode, "score": score}


# ═══════════════════════════════════════════════════════════════════════
# 4. SEA Framework  (ECNU-SEA/SEA-E, exact instruction from repo)
# ═══════════════════════════════════════════════════════════════════════

_SEA_TEMPLATE = json.loads(
    (REPOS_DIR / "SEA" / "inference" / "template.json").read_text()
)

# Report-adapted SEA instruction (based on original instruction_e)
_SEA_REPORT_INSTRUCTION = (
    "You are a highly experienced, conscientious, and fair reviewer of "
    "AI-generated computational biology research reports (NOT journal "
    "papers). " + REPORT_CONTEXT + "\n\n"
    "Please review this report. Organise your review into these sections:\n"
    "1. **Summary**: A summary of the report in 100-150 words.\n"
    "2. **Strengths/Weaknesses/Questions**: Listed as bullet points with "
    "specific examples from the report.\n"
    "3. **Soundness/Contribution/Presentation**: Rate 1-4 each "
    "(1=poor, 2=fair, 3=good, 4=excellent).\n"
    "4. **Rating**: Overall rating 1-10 "
    "(1=fundamentally flawed … 5=acceptable with revisions … 10=exceptional).\n"
    "5. **Report Assessment**: Accept/Revise/Reject with reasons based on "
    "analytical rigour, hypothesis plausibility, and experimental design quality.\n\n"
    "Use this format:\n"
    "**Summary:**\nContent\n\n"
    "**Strengths:**\n- ...\n\n**Weaknesses:**\n- ...\n\n"
    "**Questions:**\n- ...\n\n"
    "**Soundness:**\nScore\n\n**Presentation:**\nScore\n\n"
    "**Contribution:**\nScore\n\n"
    "**Rating:**\nScore\n\n"
    "**Report Assessment:**\n- Decision: Accept/Revise/Reject\n- Reasons: ...\n\n"
    "OVERALL_SCORE: <integer 1-10>\n\n"
    "Please ensure your feedback is objective and constructive. "
    "The report is as follows:"
)


def run_sea(text: str, path: Path, out: Path, **kw) -> dict:
    paper = text.split("## References")[0].strip() if "## References" in text else text
    review = _hf_generate(
        "ECNU-SEA/SEA-E", _SEA_REPORT_INSTRUCTION, paper,
        max_new_tokens=8192, temperature=0.7,
    )
    (out / "review.md").write_text(review)
    score = _extract_score(review)
    return {"review_chars": len(review), "score": score}


# ═══════════════════════════════════════════════════════════════════════
# 5. ReviewAdvisor  (NeuLab — sequence tagger for aspect-level analysis)
# ═══════════════════════════════════════════════════════════════════════

_RA_DIR = REPOS_DIR / "ReviewAdvisor"
_ra_annotator = None


def _get_annotator():
    global _ra_annotator
    if _ra_annotator is not None:
        return _ra_annotator
    for p in [str(_RA_DIR / "tagger"), str(_RA_DIR)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    from tagger.annotator import Annotator  # type: ignore
    _ra_annotator = Annotator(
        label_file=str(_RA_DIR / "tagger" / "labels.txt"),
        model_file_path=str(_RA_DIR / "seqlab_final"),
        device="gpu" if torch.cuda.is_available() else "cpu",
    )
    return _ra_annotator


def run_reviewadvisor(text: str, path: Path, out: Path, **kw) -> dict:
    annotator = _get_annotator()
    tagged: list[tuple[str, str]] = annotator.annotate(text[:30_000])

    # Group consecutive same-label tokens into spans
    aspects: dict[str, list[str]] = {}
    current_label: str | None = None
    current_words: list[str] = []
    for word, label in tagged:
        if label == current_label:
            current_words.append(word)
        else:
            if current_label and current_label != "O" and current_words:
                span = " ".join(current_words)
                aspects.setdefault(current_label, []).append(span)
            current_label = label
            current_words = [word]
    # flush last span
    if current_label and current_label != "O" and current_words:
        span = " ".join(current_words)
        aspects.setdefault(current_label, []).append(span)

    md = "# ReviewAdvisor — Aspect-Tagged Analysis\n\n"
    for label in sorted(aspects):
        md += f"## {label}\n\n"
        for s in aspects[label]:
            md += f"- {s}\n"
        md += "\n"
    if not aspects:
        md += "_No aspect-tagged sentences found._\n"
    (out / "review.md").write_text(md)
    (out / "tagged.json").write_text(
        json.dumps([{"sentence": s, "label": l} for s, l in tagged], indent=2)
    )
    # Derive a 1-10 score from aspect-tag sentiment balance
    pos = sum(len(v) for k, v in aspects.items() if "positive" in k.lower())
    neg = sum(len(v) for k, v in aspects.items() if "negative" in k.lower())
    total = pos + neg
    if total > 0:
        # Ratio of positive tags → scale to 1-10
        score = max(1, min(10, round(1 + 9 * pos / total)))
    else:
        score = None
    return {
        "num_tagged": len(tagged),
        "aspects": {k: len(v) for k, v in aspects.items()},
        "score": score,
    }


# ═══════════════════════════════════════════════════════════════════════
# 6. LitLLM  (related-works / citation analysis via modern OpenAI API)
# ═══════════════════════════════════════════════════════════════════════

_LITLLM_DIR = REPOS_DIR / "LitLLM"
_litllm_prompts: dict | None = None


def _get_litllm_prompts() -> dict:
    global _litllm_prompts
    if _litllm_prompts is None:
        _litllm_prompts = json.loads(
            (_LITLLM_DIR / "resources" / "prompts.json").read_text()
        )
    return _litllm_prompts


def run_litllm(text: str, path: Path, out: Path, **kw) -> dict:
    system = (
        "You are an expert research assistant specialising in "
        "computational biology, single-cell transcriptomics, and "
        "biomarker discovery. " + REPORT_CONTEXT
    )
    user = (
        "Below is an AI-generated computational biology research report "
        "produced by an automated pipeline. "
        "Your task is to:\n"
        "1. Identify the key claims and hypotheses in this report.\n"
        "2. Evaluate whether the cited references are real and relevant.\n"
        "3. Suggest the most important missing references that should be cited.\n"
        "4. Write a related-works section that contextualises this report "
        "within the existing literature.\n"
        "5. Assess the overall quality of the literature coverage."
        + SCORE_INSTRUCTION +
        f"\n\nReport:\n\n{text[:12000]}"
    )
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=kw.get("litllm_model", "gpt-4o"),
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


# ═══════════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════════

DISPATCH: dict[str, callable] = {
    "apr":           run_apr,
    "cycle":         run_cycle,
    "openrev":       run_openrev,
    "sea":           run_sea,
    "reviewadvisor": run_reviewadvisor,
    "litllm":        run_litllm,
}


# ═══════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════


def review_case(
    case_key: str,
    reviewers: list[str],
    base: Path,
    overwrite: bool = False,
    **kw: Any,
) -> dict:
    case = CASES[case_key]
    text = _report_text(case)
    path = _report_path(case)
    results: dict[str, Any] = {"case": case_key, "label": case["label"]}

    for rkey in reviewers:
        name = _REVIEWER_NAMES.get(rkey, rkey)
        out = base / case_key / rkey
        out.mkdir(parents=True, exist_ok=True)

        meta_file = out / "result.json"
        if meta_file.exists() and not overwrite:
            print(f"    ⏭️  {name}: cached")
            results[rkey] = json.loads(meta_file.read_text())
            continue

        print(f"    🔬 {name} …", end=" ", flush=True)
        t0 = time.time()
        try:
            res = DISPATCH[rkey](text, path, out, **kw)
            res["success"] = True
            res["elapsed_seconds"] = round(time.time() - t0, 1)
            print(f"✅ {res['elapsed_seconds']}s")
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


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--list", action="store_true", help="Show reviewers and exit")
    ap.add_argument(
        "--reviewer", nargs="+", metavar="KEY",
        help=f"Reviewer(s) to run (keys: {', '.join(DISPATCH)})",
    )
    ap.add_argument(
        "--case", nargs="+", choices=list(CASES),
        default=list(CASES), help="Case study(ies)",
    )
    ap.add_argument(
        "--models", default="gemini-2.5-pro,gpt4-o1,gpt4-o3-mini",
        help="Models for ai-peer-review (comma-sep)",
    )
    ap.add_argument(
        "--model-size", default="8B",
        help="Model size for CycleReviewer (8B/70B/123B)",
    )
    ap.add_argument("--output-dir", default=str(RESULTS_BASE))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("\nRegistered reviewers:\n")
        for key, name, ref in REVIEWERS:
            print(f"  ✅ {key:16s} {name:22s} [{ref}]")
        print()
        return

    chosen = args.reviewer or list(DISPATCH)
    bad = [k for k in chosen if k not in DISPATCH]
    if bad:
        print(f"❌ Unknown reviewer(s): {', '.join(bad)}")
        sys.exit(1)

    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    kw: dict[str, Any] = {
        "models": [m.strip() for m in args.models.split(",")],
        "model_size": args.model_size,
    }

    for k in ("OPENAI_API_KEY", "GOOGLE_API_KEY"):
        if not os.environ.get(k):
            print(f"⚠️  {k} not set")

    print("🔬 OmniCellAgent — Unified AI Review Benchmark")
    print(f"   Cases:     {', '.join(args.case)}")
    print(f"   Reviewers: {', '.join(chosen)}")
    print(f"   Output:    {base}\n")

    all_results: dict = {}
    t0 = time.time()

    for ck in args.case:
        print(f"\n{'─' * 60}")
        print(f"  📄 {CASES[ck]['label']}  ({ck})")
        print(f"{'─' * 60}")
        all_results[ck] = review_case(ck, chosen, base, args.overwrite, **kw)

    elapsed = time.time() - t0
    summary = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "reviewers": chosen,
        "cases": all_results,
    }
    sp = base / "summary.json"
    sp.write_text(json.dumps(summary, indent=2, default=str))

    # ── Generate CSV score table ─────────────────────────────────────
    csv_path = base / "scores.csv"
    _generate_scores_csv(base, chosen, list(CASES.keys()), csv_path)

    print(f"\n{'═' * 60}")
    print(f"  ✅ Done in {elapsed:.0f}s")
    print(f"  📊 Summary: {sp}")
    print(f"  📊 Scores:  {csv_path}\n")

    # Print score table to stdout
    _print_score_table(csv_path)


def _generate_scores_csv(
    base: Path,
    reviewers: list[str],
    cases: list[str],
    csv_path: Path,
) -> None:
    """Read result.json from ALL reviewer/case combos and emit a CSV table."""
    # Scan all reviewer directories, not just the ones from this invocation
    all_reviewers = list(dict.fromkeys(
        list(DISPATCH.keys()) + list(reviewers)
    ))
    rows: list[dict] = []
    for ck in cases:
        for rkey in all_reviewers:
            rfile = base / ck / rkey / "result.json"
            if not rfile.exists():
                continue
            try:
                res = json.loads(rfile.read_text())
            except Exception:
                continue
            rows.append({
                "case": ck,
                "reviewer": rkey,
                "score": res.get("score", ""),
                "success": res.get("success", False),
                "elapsed_s": res.get("elapsed_seconds", ""),
            })
    if not rows:
        return
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "reviewer", "score", "success", "elapsed_s"])
        w.writeheader()
        w.writerows(rows)


def _print_score_table(csv_path: Path) -> None:
    """Pretty-print the scores CSV as a table."""
    if not csv_path.exists():
        return
    with open(csv_path) as f:
        reader = list(csv.DictReader(f))
    if not reader:
        return

    # Pivot: rows = reviewers, cols = cases
    cases_seen = list(dict.fromkeys(r["case"] for r in reader))
    revs_seen = list(dict.fromkeys(r["reviewer"] for r in reader))
    lookup = {(r["case"], r["reviewer"]): r["score"] for r in reader}

    hdr = f"  {'Reviewer':<16s}" + "".join(f"{c:>14s}" for c in cases_seen) + f"{'Avg':>8s}"
    print(f"\n  {'─' * len(hdr)}")
    print(hdr)
    print(f"  {'─' * len(hdr)}")
    for rv in revs_seen:
        scores = []
        cells = []
        for ck in cases_seen:
            s = lookup.get((ck, rv), "")
            if s and s != "":
                try:
                    scores.append(int(s))
                    cells.append(f"{s:>14s}")
                except ValueError:
                    cells.append(f"{'—':>14s}")
            else:
                cells.append(f"{'—':>14s}")
        avg = f"{sum(scores)/len(scores):.1f}" if scores else "—"
        print(f"  {rv:<16s}" + "".join(cells) + f"{avg:>8s}")
    print(f"  {'─' * len(hdr)}")


if __name__ == "__main__":
    main()
