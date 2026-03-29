"""
Test the full PubMed paper search tool standalone.
Tests both raw text extraction and the pubmed_full_search_tool.

Usage:
    # Quick test with 5 papers (uses cached papers if available)
    python scripts/test_pubmed_full.py

    # Test with more papers
    python scripts/test_pubmed_full.py --num-papers 20

    # Test with custom query
    python scripts/test_pubmed_full.py --query "EGFR lung cancer" --num-papers 10

    # Test the full PubMedResearcher agent
    python scripts/test_pubmed_full.py --test-agent --query "What genes are involved in Alzheimer's disease?"
"""
import asyncio
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_raw_text_extraction():
    """Test that raw text extraction works when use_llm_processing=False"""
    from tools.pubmed_tools.get_papers_info_tools import get_papers_info
    from utils.path_config import get_path
    
    doi_cache_dir = get_path('cache.pubmed_doi', absolute=True)
    
    if not os.path.exists(doi_cache_dir):
        print(f"[Skip] DOI cache dir not found: {doi_cache_dir}")
        print("  Run a pubmed search first to populate the cache")
        return False``
    
    # Count available files
    files = [f for f in os.listdir(doi_cache_dir) if f.endswith(('.pdf', '.xml'))]
    if not files:
        print(f"[Skip] No PDF/XML files in DOI cache: {doi_cache_dir}")
        return False
    
    print(f"\n{'='*60}")
    print(f"TEST 1: Raw Text Extraction (use_llm_processing=False)")
    print(f"{'='*60}")
    print(f"DOI cache: {doi_cache_dir}")
    print(f"Files available: {len(files)}")
    
    # Process a few papers without LLM
    papers = await get_papers_info(
        doi_cache_dir, 
        use_llm_processing=False,  # This is what we fixed
        max_concurrent=5
    )
    
    # Check results
    total = len(papers)
    with_content = sum(1 for p in papers if p.get('llm_content'))
    with_abstract = sum(1 for p in papers if p.get('abstract'))
    
    print(f"\n--- Results ---")
    print(f"Total papers processed: {total}")
    print(f"Papers with extracted content (llm_content): {with_content}")
    print(f"Papers with abstract: {with_abstract}")
    
    if with_content > 0:
        # Show a sample
        sample = next(p for p in papers if p.get('llm_content'))
        content = sample['llm_content']
        print(f"\n--- Sample Paper ---")
        print(f"Title: {sample.get('title', 'N/A')}")
        print(f"DOI: {sample.get('doi', 'N/A')}")
        print(f"Content length: {len(content)} chars")
        print(f"Content preview: {content[:300]}...")
        print(f"\n✅ Raw text extraction works!")
        return True
    else:
        print(f"\n❌ No papers have extracted content. Check the extraction logic.")
        return False


async def test_pubmed_full_search_tool(query: str, num_papers: int = 5):
    """Test the pubmed_full_search_tool function"""
    from agent.langgraph_agent import pubmed_full_search_tool
    
    print(f"\n{'='*60}")
    print(f"TEST 2: pubmed_full_search_tool")
    print(f"{'='*60}")
    print(f"Query: {query}")
    print(f"Num papers: {num_papers}")
    
    t0 = time.time()
    result = await pubmed_full_search_tool.ainvoke({
        "query": query,
        "num_papers": num_papers
    })
    elapsed = time.time() - t0
    
    print(f"\n--- Result ({elapsed:.1f}s) ---")
    print(f"Output length: {len(result)} chars")
    
    # Count papers in output
    paper_count = result.count("## [")
    has_content = result.count("**Extracted Content:**")
    has_abstract = result.count("**Abstract:**")
    
    print(f"Papers in output: {paper_count}")
    print(f"Papers with extracted content: {has_content}")
    print(f"Papers with abstract: {has_abstract}")
    
    # Print first 2000 chars as preview
    print(f"\n--- Output Preview ---")
    print(result[:2000])
    if len(result) > 2000:
        print(f"\n... [{len(result) - 2000} more chars]")
    
    if paper_count > 0 and (has_content > 0 or has_abstract > 0):
        print(f"\n✅ pubmed_full_search_tool works! ({paper_count} papers, {has_content} with full text)")
        return True
    else:
        print(f"\n❌ Tool returned no useful content")
        return False


async def test_pubmed_researcher_agent(query: str):
    """Test the full PubMedResearcher agent end-to-end"""
    from agent.langgraph_agent import create_llm, SubAgent, pubmed_full_search_tool, pubmed_lite_tool
    
    print(f"\n{'='*60}")
    print(f"TEST 3: PubMedResearcher Agent (end-to-end)")
    print(f"{'='*60}")
    print(f"Query: {query}")
    
    llm = create_llm()
    
    agent = SubAgent(
        name="PubMedResearcher",
        description="Downloads and reads full biomedical papers from PubMed",
        system_message="""You are a biomedical literature specialist with access to FULL paper content from PubMed.
Use pubmed_full_search_tool to download and read papers. Provide a comprehensive summary with citations.
Keep your response focused and include a References section.""",
        tools=[pubmed_full_search_tool, pubmed_lite_tool],
        llm=llm
    )
    
    context = {"query": query, "top_genes": ["TREM2", "APOE"]}
    
    t0 = time.time()
    result = await agent.execute(query, context)
    elapsed = time.time() - t0
    
    print(f"\n--- Agent Result ({elapsed:.1f}s) ---")
    print(f"Success: {result.get('success')}")
    print(f"Iterations: {result.get('iterations')}")
    
    if result.get('success'):
        output = result.get('result', '')
        print(f"Output length: {len(output)} chars")
        print(f"\n--- Agent Output ---")
        print(output[:3000])
        if len(output) > 3000:
            print(f"\n... [{len(output) - 3000} more chars]")
        print(f"\n✅ PubMedResearcher agent works!")
        return True
    else:
        print(f"Error: {result.get('error')}")
        print(f"\n❌ PubMedResearcher agent failed")
        return False


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test full PubMed search tool")
    parser.add_argument("--query", default="TREM2 Alzheimer's disease microglia", help="Search query")
    parser.add_argument("--num-papers", type=int, default=5, help="Number of papers to retrieve")
    parser.add_argument("--test-agent", action="store_true", help="Also test the full PubMedResearcher agent")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip raw text extraction test")
    args = parser.parse_args()
    
    results = {}
    
    # Test 1: Raw text extraction
    if not args.skip_extraction:
        results['raw_extraction'] = await test_raw_text_extraction()
    
    # Test 2: pubmed_full_search_tool
    results['full_tool'] = await test_pubmed_full_search_tool(args.query, args.num_papers)
    
    # Test 3: PubMedResearcher agent (optional)
    if args.test_agent:
        results['agent'] = await test_pubmed_researcher_agent(args.query)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}")
    return all_passed


if __name__ == "__main__":
    asyncio.run(main())
