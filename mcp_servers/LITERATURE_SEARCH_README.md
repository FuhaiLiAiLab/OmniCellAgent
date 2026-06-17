# Literature Search MCP Server

`literature_server.py` is the agent-facing literature service for OmniCellAgent.
It runs on port `9012` in SSE mode and also supports stdio MCP clients.

For a detailed guide on building agents that consume the `message` plus
`metadata` payload contract, see
[LITERATURE_MCP_AGENT_DEV_GUIDE.md](LITERATURE_MCP_AGENT_DEV_GUIDE.md).

## Start

```bash
python mcp_servers/literature_server.py
python mcp_servers/literature_server.py --sse
```

The shared startup script also launches it:

```bash
bash mcp_servers/start_all_mcp.sh
```

## Smoke Test

After the server is running, verify the query and metadata contract:

```bash
python mcp_servers/test_literature_mcp.py --transport sse
```

Run multiple cases, retrieve 32 abstracts per case, and also validate the
full-text extraction payload contract:

```bash
python mcp_servers/test_literature_mcp.py \
  --transport sse \
  --query "TP53 cancer" \
  --query "TREM2 microglia Alzheimer disease" \
  --abstract-top-k 32 \
  --include-full-text \
  --full-top-k 8 \
  --min-full-text 3
```

To test without launching SSE, import the server module directly:

```bash
python mcp_servers/test_literature_mcp.py --transport direct
```

## Storage

The server reuses the same configured PubMed cache as the existing PubMed tools:

- `cache.pubmed_jsonl`: PubMed JSONL metadata, from `configs/paths.yaml`.
- `cache.pubmed_doi`: cached PubMed PDF/XML files, from `configs/paths.yaml`.
- `mcp_servers/literature_outputs/{session_id}/`: JSON tool payloads returned to agents.

The cache paths are already covered by the repo cache ignore rules.
`mcp_servers/literature_outputs/` is also ignored by git.

For DOI full-text retrieval, new downloads go to the shared configured
`cache.pubmed_doi` directory. Before downloading, the server also checks that
cache and `logs/` for matching `{doi-with-slashes-replaced-by-underscores}.pdf`
or `.xml` files. If a matching paper is already present, the server reuses it
and does not download it again.

## Response Contract

Every tool returns a JSON string with this shape:

```json
{
  "message": "LLM-readable summary of the search result",
  "metadata": {
    "service": "literature_search",
    "tool": "tool_name",
    "query": "original query",
    "session_id": "stable session id",
    "saved_file": "path to the persisted JSON payload"
  }
}
```

Agents should pass `message` directly into their reasoning context. They should
also keep `metadata` because it contains raw abstracts, full paper text, or raw
web page text for later citation, re-ranking, summarization, and audit.

## Tools

### `search_pubmed_abstracts_langchain(query, top_k=32, session_id=None, doc_content_chars_max=5000)`

Searches PubMed abstracts through LangChain's NCBI `PubMedAPIWrapper` when
`langchain_community` is installed. If that package is unavailable, the same
tool falls back to direct NCBI E-utilities and returns the same metadata shape.

Important metadata:

- `metadata.papers[].abstract`
- `metadata.papers[].pmid`
- `metadata.papers[].metadata`
- `metadata.papers[].raw_document`

Use this for fast abstract-only recall when you want broad abstract availability
and do not need DOI/full-text download eligibility. This matches the lite
LangChain PubMed intent used by `agent/langgraph_agent.py`, with a direct NCBI
fallback for lean environments.

### `search_pubmed_abstracts(query, top_k=32, session_id=None)`

Searches PubMed metadata and abstracts without downloading full papers.

Important metadata:

- `metadata.papers[].abstract`
- `metadata.papers[].raw_pubmed_metadata`
- `metadata.pubmed_jsonl`

Use this when richer paperscraper metadata is useful or when you want the
abstract search path to match the full-paper downloader's metadata source.

### `search_pubmed_full_papers(query, top_k=8, session_id=None, max_message_chars_per_paper=1200, min_full_text=8)`

Searches PubMed, downloads readable full papers when available, extracts raw
text from PDF/XML, and returns the full text in metadata.

Important metadata:

- `metadata.papers[].full_text`
- `metadata.papers[].text_length`
- `metadata.papers[].source_file`
- `metadata.papers[].abstract`
- `metadata.paper_cache_dir`

Use `message` for immediate synthesis. Use `metadata.papers[].full_text` for
deep reading, section extraction, secondary summarization, or saving evidence.
Set `min_full_text` when an agent requires at least N extracted full-text papers
inside the `top_k` retrieval target. If fewer are available, the tool still
returns JSON but sets `metadata.meets_min_full_text=false` and `metadata.error`.

### `search_web_full_text(query, target_results=32, session_id=None, use_llm_filter=False, max_message_chars_per_source=1000)`

Runs a Google Custom Search and extracts page content with the text-browser
pipeline. This is useful for preprints, publisher pages, protocols, lab pages,
and literature sources outside PubMed.

Important metadata:

- `metadata.sources[].raw_text`
- `metadata.sources[].url`
- `metadata.sources[].snippet`
- `metadata.sources[].raw_search_result`

This requires `GOOGLE_API_KEY` and `GOOGLE_SEARCH_ENGINE_ID` in `.env`.
LLM filtering additionally requires `OPENAI_API_KEY`.

### `search_literature_bundle(query, abstract_top_k=32, full_paper_top_k=8, web_results=32, session_id=None, abstract_source="langchain", min_full_text=8)`

Runs abstract search, PubMed full-paper retrieval, and text-browser retrieval
as one bundled literature package.

Important metadata:

- `metadata.abstract_search.papers`
- `metadata.full_paper_search.papers`
- `metadata.web_full_text_search.sources`

Use this when an agent needs a complete initial literature packet and can
tolerate the longer runtime. By default it uses the LangChain abstract search
for broader abstract availability. Set `abstract_source="paperscraper"` to use
the DOI-oriented paperscraper abstract path instead.
Set `min_full_text` to enforce a minimum number of extracted full-text papers
within `full_paper_top_k`.

## Example MCP Client Call

```python
result = await session.call_tool(
    "search_pubmed_full_papers",
    arguments={
        "query": "microglia APOE TREM2 Alzheimer disease single-cell RNA-seq",
        "top_k": 3,
        "session_id": "ad_microglia_lit"
    },
)

payload = json.loads(result.content[0].text)
llm_context = payload["message"]
raw_papers = payload["metadata"]["papers"]
first_full_text = raw_papers[0]["full_text"]
```

## Agent Usage Pattern

1. Call `search_pubmed_abstracts_langchain` for broad abstract coverage.
2. Choose refined query terms from titles/abstracts.
3. Call `search_pubmed_full_papers` for full-text evidence.
4. Call `search_web_full_text` for preprints, publisher pages, or non-PubMed context.
5. Store the complete returned JSON or use `metadata.saved_file`.

The `message` field is intentionally concise. The raw evidence is in metadata.
