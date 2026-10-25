#!/usr/bin/env python3
"""
Standalone script to build expert knowledge bases.
Run with nohup to prevent interruption:
    nohup python scripts/build_expert_kbs.py > /tmp/kb_build.log 2>&1 &

Build one expert per process to run experts in parallel (each has its own cache/author_kb/<cache_key>/
folder; never run two processes for the same expert):
    for e in GenomicsExpert NeuroscienceExpert CancerExpert; do
        nohup python scripts/build_expert_kbs.py --expert $e > logs/kb_build_$e.log 2>&1 &
    done
Download papers first with scripts/download_papers_only.py: parallel downloads from one node
trigger rate limits (429) at NCBI, bioRxiv and BMJ.
"""
import sys
import os
import argparse
import asyncio
import signal

# Ignore SIGHUP so nohup works properly
signal.signal(signal.SIGHUP, signal.SIG_IGN)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.scientist_rag_tools.scientist_tool import EXPERT_SCIENTISTS, initialize_all_authors

async def main(aliases=None):
    print("=" * 60)
    print("Starting Expert Knowledge Base Build" + (f" for {', '.join(aliases)}" if aliases else ""))
    print("=" * 60, flush=True)

    results = await initialize_all_authors(aliases)

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    for alias, result in results.items():
        print(f"  {alias}: {result}")
    print("=" * 60, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build expert knowledge bases (all configured experts by default)")
    parser.add_argument("--expert", action="append", choices=[e["alias"] for e in EXPERT_SCIENTISTS],
                        help="build only this expert (alias from configs/experts.yaml); repeatable")
    args = parser.parse_args()
    asyncio.run(main(args.expert))
