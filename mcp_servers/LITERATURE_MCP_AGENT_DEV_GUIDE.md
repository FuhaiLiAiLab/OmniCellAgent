# Building Agents Against the Literature MCP Payload Contract

This guide is for developers building LLM agents that call
`mcp_servers/literature_server.py` and need to consume both:

- `message`: compact text intended for immediate LLM reasoning.
- `metadata`: structured raw evidence intended for storage, citation, audit,
  re-ranking, and later full-text processing.

The server runs as `LiteratureSearch` on port `9012` in SSE mode and also
supports stdio MCP clients.

## Core Design

The literature MCP server is intentionally not just a search-summary service.
It returns a dual payload so one agent can hand another agent both a readable
tool result and the underlying source material.

Every tool returns a JSON string:

```json
{
  "message": "Short LLM-readable summary and previews",
  "metadata": {
    "service": "literature_search",
    "tool": "search_pubmed_full_papers",
    "query": "TREM2 microglia Alzheimer's disease",
    "session_id": "ad_microglia_001",
    "saved_file": "mcp_servers/literature_outputs/ad_microglia_001/...",
    "papers": []
  }
}
```

Your agent should treat this as two different channels:

- Put `message` into the LLM conversation or scratchpad.
- Put `metadata` into durable agent state, vector/RAG storage, or an evidence
  table. Do not rely only on the message; it is intentionally abbreviated.

## Tool Selection

Use `search_pubmed_abstracts_langchain` for broad, fast PubMed abstract recall.
Despite the name, it falls back to direct NCBI E-utilities if
`langchain_community` is unavailable.

Use `search_pubmed_abstracts` when you specifically want the paperscraper
metadata path. This may provide richer DOI-oriented metadata, but requires
`paperscraper`.

Use `search_pubmed_full_papers` when the agent needs full raw paper text from
cached or downloadable PDF/XML files. This requires the existing PubMed full
paper dependencies, including `paperscraper` and `pymupdf` for PDFs.

Use `search_web_full_text` for non-PubMed literature sources, publisher pages,
preprints, protocols, documentation pages, or other web evidence. It stores raw
text from the text-browser extraction pipeline in metadata.

Use `search_literature_bundle` when you want one initial research packet:
abstract search, PubMed full-paper search, and web text search together.

## Recommended Agent Flow

1. Generate a stable `session_id` for the user task.
2. Call `search_pubmed_abstracts_langchain` with broad search terms.
3. Save the whole returned payload.
4. Send only `payload["message"]` to the reasoning model.
5. Refine queries from the abstract metadata.
6. Call `search_pubmed_full_papers` for the most relevant focused query.
7. Store `metadata.papers[].full_text` in a long-term evidence store.
8. Call `search_web_full_text` if PubMed misses preprints, guidelines, or
   publisher context.
9. Synthesize from stored metadata, not from preview snippets alone.

## Python MCP Client Example

```python
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def call_literature_tool(project_root: str, query: str, session_id: str):
    server_params = StdioServerParameters(
        command="python",
        args=[f"{project_root}/mcp_servers/literature_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "search_pubmed_abstracts_langchain",
                arguments={
                    "query": query,
                    "top_k": 10,
                    "session_id": session_id,
                },
            )

    payload = json.loads(result.content[0].text)
    return payload
```

For SSE clients, connect to the server started with:

```bash
python mcp_servers/literature_server.py --sse
```

or:

```bash
bash mcp_servers/start_all_mcp.sh
```

## Minimal Agent State Model

A practical agent should split literature output into at least three stores:

```python
agent_state = {
    "literature_messages": [],
    "literature_payload_files": [],
    "evidence_items": [],
}
```

After each MCP call:

```python
payload = json.loads(tool_result_text)

agent_state["literature_messages"].append(
    {
        "tool": payload["metadata"]["tool"],
        "query": payload["metadata"]["query"],
        "message": payload["message"],
    }
)

agent_state["literature_payload_files"].append(
    payload["metadata"].get("saved_file")
)
```

Then normalize evidence items from tool-specific metadata:

```python
metadata = payload["metadata"]

if "papers" in metadata:
    for paper in metadata["papers"]:
        agent_state["evidence_items"].append(
            {
                "kind": "paper",
                "tool": metadata["tool"],
                "query": metadata["query"],
                "title": paper.get("title"),
                "pmid": paper.get("pmid"),
                "doi": paper.get("doi"),
                "journal": paper.get("journal"),
                "year": paper.get("year") or paper.get("published"),
                "abstract": paper.get("abstract"),
                "full_text": paper.get("full_text"),
                "source_file": paper.get("source_file"),
                "raw": paper,
            }
        )

if "sources" in metadata:
    for source in metadata["sources"]:
        agent_state["evidence_items"].append(
            {
                "kind": "web_source",
                "tool": metadata["tool"],
                "query": metadata["query"],
                "title": source.get("title"),
                "url": source.get("url"),
                "snippet": source.get("snippet"),
                "raw_text": source.get("raw_text"),
                "raw": source,
            }
        )
```

## Message Handling

`message` is designed to be copied into the LLM context as a normal tool
observation. It contains a count, citation-ish labels, URLs or source paths,
and short previews.

Example agent instruction:

```text
You have received a literature MCP message below. Use it for immediate
reasoning, but treat it as a preview. If you need exact evidence, inspect the
stored metadata item IDs listed in agent state.
```

Avoid asking the LLM to infer details from omitted text. If the summary says a
paper has `text_length=50000`, the details are in metadata, not in `message`.

## Metadata Handling

Always parse the returned JSON and persist the full object or use
`metadata.saved_file`. The saved file is useful for inter-agent handoff because
it avoids sending huge full texts through every model call.

Recommended metadata policies:

- Keep the complete raw payload for reproducibility.
- Normalize paper/source records into a smaller evidence table.
- Store full texts separately from short conversation memory.
- Assign local evidence IDs such as `paper:pmid:12345` or `web:https://...`.
- Preserve `query`, `tool`, and `session_id` on each evidence item.

## Tool-Specific Metadata Shapes

### `search_pubmed_abstracts_langchain`

Primary fields:

```json
{
  "metadata": {
    "tool": "search_pubmed_abstracts_langchain",
    "source": "pubmed_langchain or pubmed_eutils",
    "fallback_reason": "",
    "paper_count": 10,
    "papers": [
      {
        "index": 1,
        "title": "...",
        "pmid": "...",
        "published": "...",
        "year": "...",
        "journal": "...",
        "authors": [],
        "doi": "...",
        "abstract": "...",
        "metadata": {},
        "raw_document": {}
      }
    ]
  }
}
```

Notes:

- If `langchain_community` is installed, records come from LangChain's
  `PubMedAPIWrapper`.
- If not, records come from direct NCBI E-utilities.
- Some PubMed records have no abstract. Your agent should handle empty strings.

### `search_pubmed_abstracts`

Primary fields:

```json
{
  "metadata": {
    "tool": "search_pubmed_abstracts",
    "source": "pubmed",
    "pubmed_jsonl": "...",
    "paper_count": 10,
    "papers": [
      {
        "title": "...",
        "authors": [],
        "journal": "...",
        "year": "...",
        "doi": "...",
        "pmid": "...",
        "abstract": "...",
        "raw_pubmed_metadata": {}
      }
    ]
  }
}
```

If `paperscraper` is unavailable, this tool returns a valid JSON payload with
`metadata.error` and zero papers. The server still starts.

### `search_pubmed_full_papers`

Primary fields:

```json
{
  "metadata": {
    "tool": "search_pubmed_full_papers",
    "paper_cache_dir": "...",
    "papers": [
      {
        "title": "...",
        "doi": "...",
        "pmid": "...",
        "abstract": "...",
        "source_file": "...",
        "file_type": "pdf or xml",
        "full_text": "...",
        "text_length": 123456,
        "extraction_error": ""
      }
    ]
  }
}
```

Notes:

- New paper downloads go into the shared configured PubMed DOI cache.
- Before downloading, the server checks the shared DOI cache and `logs/` for
  an existing `{doi-with-slashes-replaced}.pdf` or `.xml`.
- Do not put `full_text` directly into a normal chat prompt unless it is short
  enough. Use chunking, retrieval, or targeted section extraction.

### `search_web_full_text`

Primary fields:

```json
{
  "metadata": {
    "tool": "search_web_full_text",
    "source": "text_browser",
    "source_count": 5,
    "sources": [
      {
        "title": "...",
        "url": "...",
        "snippet": "...",
        "raw_text": "...",
        "text_length": 12000,
        "llm_summary": "",
        "llm_confidence": null,
        "llm_key_findings": [],
        "raw_search_result": {}
      }
    ]
  }
}
```

Notes:

- Requires `GOOGLE_API_KEY` and `GOOGLE_SEARCH_ENGINE_ID`.
- `use_llm_filter=True` also requires `OPENAI_API_KEY`.
- If the backend cannot load, the tool returns JSON with `metadata.error`.

### `search_literature_bundle`

Primary fields:

```json
{
  "metadata": {
    "tool": "search_literature_bundle",
    "abstract_source": "langchain",
    "abstract_search": {},
    "full_paper_search": {},
    "web_full_text_search": {}
  }
}
```

Flatten the nested results before storing them in your evidence table. Include
the parent `query` and `session_id` on all flattened evidence items.

## Robust Parsing Helper

Use defensive parsing. MCP tool calls can fail at the transport layer, but
backend failures should usually arrive as JSON with `metadata.error`.

```python
import json


def parse_literature_payload(tool_text: str) -> tuple[str, dict]:
    try:
        payload = json.loads(tool_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Literature MCP returned non-JSON text: {tool_text[:500]}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Literature MCP payload is not a JSON object")

    message = payload.get("message", "")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Literature MCP payload metadata is not an object")

    return message, metadata
```

Then inspect `metadata.error`:

```python
message, metadata = parse_literature_payload(result.content[0].text)

if metadata.get("error"):
    agent_state["literature_messages"].append(
        {
            "tool": metadata.get("tool"),
            "query": metadata.get("query"),
            "message": message,
            "error": metadata["error"],
        }
    )
    # Choose another literature path instead of crashing.
```

## Inter-Agent Handoff

When handing literature results to another agent, do not paste all raw text by
default. Send a compact handoff object:

```json
{
  "task": "Use these papers to validate candidate AD microglia targets.",
  "literature_message": "...",
  "payload_file": "mcp_servers/literature_outputs/ad_001/...",
  "evidence_ids": [
    "paper:pmid:12345",
    "paper:doi:10.1234_example",
    "web:https://example.org/preprint"
  ]
}
```

The receiving agent can:

1. Read `literature_message` for immediate orientation.
2. Load `payload_file` for raw abstracts/full text.
3. Pull only needed `evidence_ids` into its working context.

## Citation Strategy

Prefer citation keys in this order:

1. DOI, if present.
2. PMID, if present.
3. URL for web sources.
4. Title plus year when no stable identifier exists.

Example:

```python
def citation_key(item: dict) -> str:
    if item.get("doi"):
        return f"doi:{item['doi']}"
    if item.get("pmid"):
        return f"pmid:{item['pmid']}"
    if item.get("url"):
        return f"url:{item['url']}"
    return f"title:{item.get('title', 'unknown')}:{item.get('year', '')}"
```

## Recommended Prompt Pattern

When passing `message` to an LLM, pair it with a reminder that raw evidence is
available separately:

```text
Tool result from LiteratureSearch:

{payload["message"]}

Important: this is a preview. Raw abstracts, full texts, and web page text are
stored in structured metadata under evidence IDs. Ask for or retrieve those
records before making fine-grained claims.
```

For full-paper analysis, first retrieve only relevant chunks or sections from
`metadata.papers[].full_text`. Avoid dropping many full papers into one prompt.

## Failure And Fallback Behavior

The server is designed to start even when optional backends are missing.

Expected recoverable cases:

- `langchain_community` missing: `search_pubmed_abstracts_langchain` uses
  direct NCBI E-utilities.
- `paperscraper` missing: paperscraper abstract and full-paper tools return
  JSON with `metadata.error`.
- `configs/paths.yaml` missing: server import still works; cache paths fall
  back to `cache/PudMed_DB/...`.
- Google search credentials missing: `search_web_full_text` returns a backend
  error payload.

Agent behavior should be:

- Log `metadata.error`.
- Keep the returned `message`.
- Fall back to another tool where possible.
- Continue with partial evidence when appropriate.

## Testing An Agent Integration

Use a low-cost abstract call first:

```python
result = await session.call_tool(
    "search_pubmed_abstracts_langchain",
    arguments={
        "query": "TP53 cancer",
        "top_k": 1,
        "session_id": "dev_smoke_test",
    },
)

payload = json.loads(result.content[0].text)
assert payload["metadata"]["tool"] == "search_pubmed_abstracts_langchain"
assert "message" in payload
assert "papers" in payload["metadata"]
```

Then test a failure-safe path if your environment intentionally lacks
`paperscraper`:

```python
result = await session.call_tool(
    "search_pubmed_full_papers",
    arguments={
        "query": "TP53 cancer",
        "top_k": 1,
        "session_id": "dev_missing_fulltext_backend",
    },
)

payload = json.loads(result.content[0].text)
if payload["metadata"].get("error"):
    print("Full-paper backend unavailable, but JSON contract is intact")
```

## Implementation Checklist

- Use a stable `session_id` across all literature calls for one user task.
- Parse JSON before sending anything to another system.
- Send `message` to the LLM as the immediate tool observation.
- Persist `metadata` or at least `metadata.saved_file`.
- Normalize `papers` and `sources` into your agent evidence model.
- Never assume full text exists; check `full_text` and `text_length`.
- Never assume abstracts exist; PubMed records can be metadata-only.
- Handle `metadata.error` without crashing the agent.
- Use DOI/PMID/URL as stable evidence keys.
- Use chunking or retrieval before asking an LLM to reason over long full texts.
