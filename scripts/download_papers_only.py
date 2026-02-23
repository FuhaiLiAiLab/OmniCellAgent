#!/usr/bin/env python3
"""
Download papers for all configured experts WITHOUT building the RAG index.
Use this to verify paper availability before the expensive entity-extraction step.

Usage:
    python scripts/download_papers_only.py
"""
import sys, os, asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.scientist_rag_tools.scientist_tool import (
    EXPERT_SCIENTISTS, AuthorKnowledgeBase
)


async def download_all():
    results = {}
    for scientist in EXPERT_SCIENTISTS:
        real_name = scientist["real_name"]
        alias = scientist["alias"]
        cache_key = scientist["cache_key"]

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

        count = kb._count_doi_papers()
        status = f"✅ {count} papers" if count >= kb.MIN_PAPERS else f"⚠️  {count} papers (< {kb.MIN_PAPERS})"
        results[alias] = status
        print(f"\n{alias}: {status}")

    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for alias, status in results.items():
        print(f"  {alias}: {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(download_all())
