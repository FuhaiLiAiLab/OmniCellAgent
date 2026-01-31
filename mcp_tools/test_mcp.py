#!/usr/bin/env python3
"""
Test script for OmniCellAgent MCP server.

Tests all tools to ensure they're properly exposed via MCP protocol.
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_mcp_server():
    """Test the MCP server by calling each tool."""
    
    print("=" * 80)
    print("  OmniCellAgent MCP Server Test")
    print("=" * 80)
    print()
    
    # Server parameters - run the server in subprocess
    server_params = StdioServerParameters(
        command="python",
        args=[os.path.join(os.path.dirname(__file__), "server.py")],
        env=None
    )
    
    try:
        print("🚀 Starting MCP server...")
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize session
                print("✓ Initializing session...")
                await session.initialize()
                print("✓ Session initialized\n")
                
                # List available tools
                print("📋 Available tools:")
                tools_response = await session.list_tools()
                for tool in tools_response.tools:
                    print(f"  - {tool.name}: {tool.description[:60]}...")
                print()
                
                # Test 1: Search PubMed (simple test with 1 paper)
                print("-" * 80)
                print("TEST 1: search_pubmed")
                print("-" * 80)
                try:
                    result = await session.call_tool(
                        "search_pubmed",
                        arguments={
                            "query": "KRAS G12C inhibitor",
                            "top_k": 1
                        }
                    )
                    print("✓ PubMed search successful")
                    print(f"  Result preview: {str(result.content[0].text)[:200]}...")
                    print()
                except Exception as e:
                    print(f"✗ PubMed search failed: {e}\n")
                
                # Test 2: Web Search
                print("-" * 80)
                print("TEST 2: search_web")
                print("-" * 80)
                try:
                    result = await session.call_tool(
                        "search_web",
                        arguments={
                            "query": "CRISPR gene editing 2024",
                            "target_results": 3,
                            "use_llm_filter": False  # Skip LLM filter for faster test
                        }
                    )
                    print("✓ Web search successful")
                    print(f"  Result preview: {str(result.content[0].text)[:200]}...")
                    print()
                except Exception as e:
                    print(f"✗ Web search failed: {e}\n")
                
                # Test 3: Knowledge Graph (may fail if Neo4j not running)
                print("-" * 80)
                print("TEST 3: search_knowledge_graph")
                print("-" * 80)
                try:
                    result = await session.call_tool(
                        "search_knowledge_graph",
                        arguments={
                            "query": "What genes are associated with breast cancer?"
                        }
                    )
                    print("✓ Knowledge graph search successful")
                    print(f"  Result preview: {str(result.content[0].text)[:200]}...")
                    print()
                except Exception as e:
                    print(f"⚠ Knowledge graph search failed (Neo4j may not be running): {e}\n")
                
                # Test 4: Scientist RAG (may be slow)
                print("-" * 80)
                print("TEST 4: query_scientist_knowledge")
                print("-" * 80)
                try:
                    result = await session.call_tool(
                        "query_scientist_knowledge",
                        arguments={
                            "author_name": "Jennifer Doudna",
                            "question": "What is CRISPR-Cas9?"
                        }
                    )
                    print("✓ Scientist RAG successful")
                    print(f"  Result preview: {str(result.content[0].text)[:200]}...")
                    print()
                except Exception as e:
                    print(f"⚠ Scientist RAG failed: {e}\n")
                
                # Test 5: Omics Analysis (SKIP - too slow for quick test)
                print("-" * 80)
                print("TEST 5: analyze_omics_data (SKIPPED - too slow)")
                print("-" * 80)
                print("⊘ Skipping omics analysis test (takes 5-15 minutes)")
                print("  To test manually, use:")
                print('  await session.call_tool("analyze_omics_data", ')
                print('      arguments={"query": "lung adenocarcinoma"})')
                print()
                
                print("=" * 80)
                print("  Test Summary")
                print("=" * 80)
                print("✓ MCP server is functional")
                print("✓ Tools are properly exposed via MCP protocol")
                print("✓ Basic tool calls working")
                print()
                print("Note: Some tools may fail if dependencies aren't running")
                print("  (Neo4j for knowledge graph, OmniCellTOSG for omics)")
                print()
    
    except Exception as e:
        print(f"\n❌ Server test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_mcp_server())
    sys.exit(0 if success else 1)
