"""
Example: Integrating MCP Servers with LangGraph Agent

This demonstrates how to connect OmniCellAgent's modular MCP servers
with a LangGraph agent for orchestrated biomedical research.

Architecture:
- Each MCP server runs independently (ports 9001-9005)
- LangGraph agent connects via MCP protocol
- Tools are exposed as LangChain tools
- Session ID passed to maintain file organization
"""

import asyncio
import uuid
from typing import List, Dict, Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent


class MCPToolBuilder:
    """Builder for creating LangChain tools from MCP servers."""
    
    def __init__(self, project_root: str, conda_env: str = "a2a-dev"):
        self.project_root = project_root
        self.conda_env = conda_env
        self.connections = {}
    
    async def create_tool(
        self, 
        server_script: str, 
        tool_name: str,
        port: int = None
    ):
        """
        Create a LangChain tool from an MCP server.
        
        Args:
            server_script: Name of the server script (e.g., "pubmed_server.py")
            tool_name: Name of the tool to expose (e.g., "search_pubmed")
            port: Optional port for SSE connection (None for stdio)
        """
        # Set up server parameters
        if port:
            # SSE connection
            server_params = StdioServerParameters(
                command="conda",
                args=[
                    "run", "-n", self.conda_env,
                    "python", f"{self.project_root}/mcp_servers/{server_script}",
                    "--sse"
                ]
            )
        else:
            # stdio connection (default)
            server_params = StdioServerParameters(
                command="conda",
                args=[
                    "run", "-n", self.conda_env,
                    "python", f"{self.project_root}/mcp_servers/{server_script}"
                ]
            )
        
        # Connect to MCP server
        read, write = await stdio_client(server_params).__aenter__()
        session = await ClientSession(read, write).__aenter__()
        await session.initialize()
        
        # Store connection for cleanup
        self.connections[tool_name] = (read, write, session)
        
        # Get tool info from server
        tools_response = await session.list_tools()
        tool_info = next(
            (t for t in tools_response.tools if t.name == tool_name),
            None
        )
        
        if not tool_info:
            raise ValueError(f"Tool {tool_name} not found in {server_script}")
        
        # Create LangChain tool wrapper
        @tool
        async def mcp_tool(**kwargs) -> str:
            """Dynamically created MCP tool."""
            result = await session.call_tool(tool_name, arguments=kwargs)
            return result.content[0].text
        
        # Update tool metadata from MCP server
        mcp_tool.name = tool_name
        mcp_tool.description = tool_info.description or f"MCP tool: {tool_name}"
        
        return mcp_tool
    
    async def cleanup(self):
        """Close all MCP connections."""
        for connection in self.connections.values():
            try:
                await connection[2].__aexit__(None, None, None)
                await connection[0].__aexit__(None, None, None)
                await connection[1].__aexit__(None, None, None)
            except:
                pass


async def create_omnicell_agent(project_root: str):
    """
    Create a LangGraph agent with all OmniCellAgent MCP tools.
    
    Args:
        project_root: Path to OmniCellAgent directory
    
    Returns:
        Tuple of (agent, tool_builder, session_id)
    """
    # Initialize MCP tool builder
    tool_builder = MCPToolBuilder(project_root)
    
    # Create session ID for file organization
    session_id = f"langgraph_{uuid.uuid4().hex[:8]}"
    
    print("🔧 Connecting to MCP servers...")
    
    # Create tools from each MCP server
    tools = []
    
    try:
        # PubMed (port 9001)
        pubmed_tool = await tool_builder.create_tool(
            "pubmed_server.py",
            "search_pubmed"
        )
        tools.append(pubmed_tool)
        print("  ✓ PubMed tool connected")
        
        # Web Search (port 9002)
        web_tool = await tool_builder.create_tool(
            "websearch_server.py",
            "search_web"
        )
        tools.append(web_tool)
        print("  ✓ Web Search tool connected")
        
        # Knowledge Graph (port 9003)
        kg_tool = await tool_builder.create_tool(
            "knowledge_graph_server.py",
            "search_knowledge_graph"
        )
        tools.append(kg_tool)
        print("  ✓ Knowledge Graph tool connected")
        
        # Scientist RAG (port 9004)
        scientist_tool = await tool_builder.create_tool(
            "scientist_rag_server.py",
            "query_scientist_knowledge"
        )
        tools.append(scientist_tool)
        print("  ✓ Scientist RAG tool connected")
        
        # Omics Analysis (port 9005)
        omics_tool = await tool_builder.create_tool(
            "omics_server.py",
            "analyze_omics_data"
        )
        tools.append(omics_tool)
        print("  ✓ Omics Analysis tool connected")
        
    except Exception as e:
        print(f"⚠ Error connecting to MCP servers: {e}")
        print("  Make sure all MCP servers are running:")
        print("  bash mcp_servers/start_all_mcp.sh")
        await tool_builder.cleanup()
        raise
    
    print(f"✓ All {len(tools)} MCP tools connected")
    print(f"📁 Session ID: {session_id}")
    print()
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    
    # Create LangGraph agent with system message
    system_message = f"""You are OmniCellAgent, an AI research assistant specialized in biomedical research.

You have access to 5 powerful research tools via MCP protocol:
1. search_pubmed - Search PubMed for papers, download PDFs, extract full-text
2. search_web - Google Custom Search with intelligent content extraction
3. search_knowledge_graph - Query biomedical knowledge graph (genes, diseases, pathways)
4. query_scientist_knowledge - RAG over specific scientist's publications
5. analyze_omics_data - Comprehensive single-cell RNA-seq analysis

CRITICAL: For tools that save files (search_pubmed, analyze_omics_data), ALWAYS pass:
  session_id="{session_id}"

This ensures all outputs are organized under: webapp/sessions/{session_id}/

Example workflow for "Analyze lung cancer":
1. search_pubmed(query="lung adenocarcinoma recent reviews", top_k=3, session_id="{session_id}")
2. search_knowledge_graph(query="genes associated with lung adenocarcinoma")
3. analyze_omics_data(query="lung adenocarcinoma", session_id="{session_id}")
4. Synthesize findings from all sources

Be thorough, cite sources, and always pass session_id to file-generating tools.
"""
    
    agent = create_react_agent(llm, tools, messages_modifier=system_message)
    
    return agent, tool_builder, session_id


async def main():
    """Example usage of OmniCellAgent with MCP servers."""
    
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Create agent
    agent, tool_builder, session_id = await create_omnicell_agent(project_root)
    
    # Example query
    query = "What are the key genes and pathways in pancreatic cancer? Use omics data analysis."
    
    print("=" * 80)
    print("  OmniCellAgent Query")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Session: {session_id}")
    print("=" * 80)
    print()
    
    try:
        # Run agent
        async for event in agent.astream_events(
            {"messages": [("user", query)]},
            version="v1"
        ):
            kind = event["event"]
            
            if kind == "on_chat_model_stream":
                # Stream LLM output
                content = event["data"]["chunk"].content
                if content:
                    print(content, end="", flush=True)
            
            elif kind == "on_tool_start":
                # Tool execution started
                tool_name = event["name"]
                print(f"\n\n🔧 Calling tool: {tool_name}")
                print(f"   Arguments: {event['data'].get('input')}")
            
            elif kind == "on_tool_end":
                # Tool execution completed
                print(f"   ✓ Tool completed")
        
        print("\n\n" + "=" * 80)
        print(f"  Analysis Complete!")
        print("=" * 80)
        print(f"Results saved to: webapp/sessions/{session_id}/")
        print()
    
    finally:
        # Cleanup MCP connections
        await tool_builder.cleanup()
        print("✓ MCP connections closed")


if __name__ == "__main__":
    asyncio.run(main())
