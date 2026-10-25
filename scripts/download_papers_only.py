#!/usr/bin/env python3
"""
Download papers for all configured experts WITHOUT building the RAG index.
Use this to verify paper availability before the expensive entity-extraction step.

Usage:
    python scripts/download_papers_only.py                         # all configured experts
    python scripts/download_papers_only.py --expert CancerExpert   # one expert (repeatable)
"""
import sys, os, asyncio, argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.scientist_rag_tools.scientist_tool import (
    EXPERT_SCIENTISTS, AuthorKnowledgeBase
)


async def download_all(aliases=None):
    results = {}
    for scientist in EXPERT_SCIENTISTS:
        real_name = scientist["real_name"]
        alias = scientist["alias"]
        cache_key = scientist["cache_key"]
        if aliases is not None and alias not in aliases:
            continue

        if real_name == "CHANGE_ME":
            print(f"\n⚠️  Skipping {alias}: real_name not configured")
            results[alias] = "SKIPPED"
            continue

        print(f"\n{'='*60}")
        print(f"📥 Downloading papers for {alias} (cache_key={cache_key})")
        print(f"{'='*60}")

        kb = AuthorKnowledgeBase(
            cache_key=cache_key,
            pubmed_name=real_name,
            top_k=scientist["top_k"],
        )
        await kb.fetch_and_cache_papers()

        count = len(await kb._fulltext_files())
        status = (f"✅ {count} full-text papers" if count >= kb.MIN_PAPERS
                  else f"⚠️  {count} full-text papers (< {kb.MIN_PAPERS}; the KB will be built from these)")
        results[alias] = status
        print(f"\n{alias}: {status}")

    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for alias, status in results.items():
        print(f"  {alias}: {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download expert papers without building the KBs")
    parser.add_argument("--expert", action="append", choices=[e["alias"] for e in EXPERT_SCIENTISTS],
                        help="download only this expert's papers (alias from configs/experts.yaml); repeatable")
    args = parser.parse_args()
    asyncio.run(download_all(args.expert))
