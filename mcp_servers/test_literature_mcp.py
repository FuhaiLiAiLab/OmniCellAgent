"""Smoke-test the Literature MCP server payload contract.

Examples:
    # Test a running SSE server.
    python mcp_servers/test_literature_mcp.py --transport sse

    # Test by importing the server module directly.
    python mcp_servers/test_literature_mcp.py --transport direct

    # Use a custom query/session.
    python mcp_servers/test_literature_mcp.py --query "TP53 cancer" --session-id dev_lit_test

    # Query 32 abstracts and also test full-text extraction payloads.
    python mcp_servers/test_literature_mcp.py --query "TP53 cancer" --include-full-text

    # Run multiple cases.
    python mcp_servers/test_literature_mcp.py --query "TP53 cancer" --query "KRAS pancreatic cancer"
"""
import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any, Callable, Dict, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


EXPECTED_TOOLS = {
    "search_pubmed_abstracts",
    "search_pubmed_abstracts_langchain",
    "search_pubmed_full_papers",
    "search_web_full_text",
    "search_literature_bundle",
}


def assert_payload_contract(payload: Dict[str, Any], expected_tool: str) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise AssertionError("payload must be a JSON object")
    if not isinstance(payload.get("message"), str) or not payload["message"]:
        raise AssertionError("payload.message must be a non-empty string")
    if not isinstance(payload.get("metadata"), dict):
        raise AssertionError("payload.metadata must be an object")

    metadata = payload["metadata"]
    required_fields = ["service", "tool", "query", "session_id", "saved_file"]
    missing = [field for field in required_fields if not metadata.get(field)]
    if missing:
        raise AssertionError(f"metadata missing required fields: {missing}")

    if metadata["service"] != "literature_search":
        raise AssertionError(f"unexpected service: {metadata['service']}")
    if metadata["tool"] != expected_tool:
        raise AssertionError(f"unexpected tool: {metadata['tool']}")
    if not os.path.exists(metadata["saved_file"]):
        raise AssertionError(f"saved_file does not exist: {metadata['saved_file']}")

    return metadata


def safe_case_id(query: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", query).strip("_").lower()
    return (safe or "query")[:80]


def validate_paper_payload(metadata: Dict[str, Any], require_papers: bool) -> List[Dict[str, Any]]:
    papers = metadata.get("papers", [])
    if not isinstance(papers, list):
        raise AssertionError("metadata.papers must be a list")
    if metadata.get("paper_count") != len(papers):
        raise AssertionError("metadata.paper_count must equal len(metadata.papers)")
    if require_papers and not papers:
        raise AssertionError("expected at least one paper for this query")
    return papers


def print_payload_summary(label: str, metadata: Dict[str, Any], papers: List[Dict[str, Any]]) -> None:
    print(f"{label}_OK")
    print(f"  tool: {metadata['tool']}")
    print(f"  source: {metadata.get('source', '')}")
    print(f"  query: {metadata['query']}")
    print(f"  paper_count: {metadata.get('paper_count')}")
    print(f"  saved_file: {metadata['saved_file']}")
    if metadata.get("error"):
        print(f"  error: {metadata['error']}")
    if papers:
        first = papers[0]
        print(f"  first_title: {first.get('title', '')}")
        print(f"  first_pmid: {first.get('pmid', '')}")
        if first.get("full_text"):
            print(f"  first_text_length: {first.get('text_length', len(first['full_text']))}")


def parse_tool_json(text: str, tool_name: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        prefix = text[:500].replace("\n", " ")
        raise ValueError(f"{tool_name} returned non-JSON text: {prefix}") from exc


async def call_direct_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    import mcp_servers.literature_server as server

    tools = await server.mcp.list_tools()
    tool_names = {tool.name for tool in tools}
    missing = EXPECTED_TOOLS - tool_names
    if missing:
        raise AssertionError(f"missing MCP tools: {sorted(missing)}")

    tool_fn: Callable[..., Any] = getattr(server, tool_name)
    text = await tool_fn(**arguments)
    return parse_tool_json(text, tool_name)


async def call_sse_tool(url: str, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_response = await session.list_tools()
            tool_names = {tool.name for tool in tools_response.tools}
            missing = EXPECTED_TOOLS - tool_names
            if missing:
                raise AssertionError(f"missing MCP tools: {sorted(missing)}")

            result = await session.call_tool(
                tool_name,
                arguments=arguments,
            )

    return parse_tool_json(result.content[0].text, tool_name)


async def call_tool(
    transport: str,
    url: str,
    tool_name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    if transport == "sse":
        return await call_sse_tool(url, tool_name, arguments)
    return await call_direct_tool(tool_name, arguments)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test Literature MCP query and metadata contract.")
    parser.add_argument("--transport", choices=["sse", "direct"], default="sse")
    parser.add_argument("--url", default="http://127.0.0.1:9012/sse")
    parser.add_argument("--query", action="append", help="Query/case to run. Can be repeated.")
    parser.add_argument("--top-k", type=int, default=None, help="Deprecated alias for --abstract-top-k.")
    parser.add_argument("--abstract-top-k", type=int, default=32)
    parser.add_argument("--include-full-text", action="store_true")
    parser.add_argument("--full-top-k", type=int, default=8)
    parser.add_argument("--min-full-text", type=int, default=8)
    parser.add_argument("--case-delay", type=float, default=1.0, help="Seconds to wait between query cases.")
    parser.add_argument("--session-id", default="literature_mcp_test")
    args = parser.parse_args()

    print("LITERATURE_MCP_TEST_OK")
    print(f"transport: {args.transport}")

    queries = args.query or ["TP53 cancer"]
    abstract_top_k = args.top_k if args.top_k is not None else args.abstract_top_k

    for index, query in enumerate(queries):
        case_session_id = args.session_id
        if len(queries) > 1:
            case_session_id = f"{args.session_id}_{safe_case_id(query)}"

        print(f"\nCASE: {query}")

        abstract_payload = await call_tool(
            args.transport,
            args.url,
            "search_pubmed_abstracts_langchain",
            {
                "query": query,
                "top_k": abstract_top_k,
                "session_id": f"{case_session_id}_abstracts",
            },
        )
        abstract_metadata = assert_payload_contract(
            abstract_payload,
            "search_pubmed_abstracts_langchain",
        )
        abstract_papers = validate_paper_payload(abstract_metadata, require_papers=True)
        print_payload_summary("ABSTRACT", abstract_metadata, abstract_papers)

        if args.include_full_text:
            full_payload = await call_tool(
                args.transport,
                args.url,
                "search_pubmed_full_papers",
                {
                    "query": query,
                    "top_k": args.full_top_k,
                    "session_id": f"{case_session_id}_full_text",
                    "min_full_text": args.min_full_text,
                },
            )
            full_metadata = assert_payload_contract(
                full_payload,
                "search_pubmed_full_papers",
            )
            full_papers = validate_paper_payload(
                full_metadata,
                require_papers=not bool(full_metadata.get("error")),
            )
            if args.min_full_text > 0 and not full_metadata.get("meets_min_full_text", False):
                print(
                    "FULL_TEXT_REQUIREMENT_NOT_MET "
                    f"{len(full_papers)}/{args.min_full_text}"
                )
            print_payload_summary("FULL_TEXT", full_metadata, full_papers)

        if index < len(queries) - 1 and args.case_delay > 0:
            await asyncio.sleep(args.case_delay)


if __name__ == "__main__":
    asyncio.run(main())
