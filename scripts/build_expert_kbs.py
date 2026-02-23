#!/usr/bin/env python3
"""
Standalone script to build expert knowledge bases.
Run with nohup to prevent interruption:
    nohup python scripts/build_expert_kbs.py > /tmp/kb_build.log 2>&1 &
"""
import sys
import os
import asyncio
import signal

# Ignore SIGHUP so nohup works properly
signal.signal(signal.SIGHUP, signal.SIG_IGN)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.scientist_rag_tools.scientist_tool import initialize_all_authors

async def main():
    print("=" * 60)
    print("Starting Expert Knowledge Base Build")
    print("=" * 60, flush=True)
    
    results = await initialize_all_authors()
    
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    for alias, result in results.items():
        print(f"  {alias}: {result}")
    print("=" * 60, flush=True)

if __name__ == "__main__":
    asyncio.run(main())
