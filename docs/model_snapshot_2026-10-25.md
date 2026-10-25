# Model snapshot, 2026-10-25 (before the switch to Gemini 3.8 Flash / Gemini 4)

The models that every step and every agent of the LangGraph agent called, as configured on branch
`revision` (HEAD `5b4386d` plus uncommitted changes) and as observed in the Alzheimer's disease case-study
run of 2026-10-25 (session `AD_20261025`, query "What are the key dysfunctional genes and pathways in
Alzheimer's Disease?", 35 min). Line numbers refer to the code before the switch.

Six distinct models were called in that run: four that generate text (two Gemini models, gpt-6-luna,
and the Llama-3.1-8B inside G-Retriever) and two OpenAI embedding models. Counting G-Retriever's graph
network and its Llama separately gives seven. The omics analysis itself uses no AI model.

## Agent LLM (LangChain `ChatGoogleGenerativeAI`)

One shared instance, created by `create_llm()` (`agent/langgraph_agent.py:865`), temperature 0, no
thinking setting. Requested model `gemini-3.5-flash` (CLI `--model`, default at `agent/langgraph_agent.py:3594`;
web UI default at `webapp/UI/dash_UI.py:903` and `webapp/UI/langgraph_adapter.py:51,456`). Every one of the
28 responses in the run reported `gemini-3.6-flash` as the serving model (`usage_log.json`, `llm_calls[].model`);
the log's top-level `model` field shows only the requested name.

| Step / agent | Call site | Calls in AD run |
|---|---|---|
| Planner (structured `ResearchPlan`; JSON fallback) | `agent/langgraph_agent.py:1898`, `:1947` | 1 |
| Replanner | `agent/langgraph_agent.py:2317` | 0 |
| OmicMiningAgent (tool loop over `omic_analysis_tool`) | `SubAgent.execute`, `agent/langgraph_agent.py:925–974` | 6 |
| BioMarkerKGAgent (tool loop over `gretriever_tool`) | same | 2 |
| PubMedResearcher (tool loop over the PubMed tools) | same | 4 |
| GoogleSearcher (tool loop over `web_search_tool`) | same | 7 |
| ScientistsAgent: expert routing (structured `ExpertRouting`), the expert sub-agents (GenomicsExpert, NeuroscienceExpert, LongevityBiostatsExpert, CancerExpert) and the synthesis | `agent/langgraph_agent.py:1375`, `:1306–1313`, `:1500` | 7 |
| Reporter | `agent/langgraph_agent.py:2650` | 1 |
| Report revision (human-in-the-loop feedback) | `agent/langgraph_agent.py:3406` | 0 |

Sub-agents are built with the shared instance in `agent/langgraph_agent.py:1686–1766`.

## Models inside the tools

| Step | Model | Settings | Where it runs | Code | Calls in AD run |
|---|---|---|---|---|---|
| Omics: cohort loading, HGNC filter, limma DE, Enrichr enrichment, plots | none | — | local R/Python; Enrichr web service | `tools/omic_tools/` | — |
| PubMed: search and full-text download | none | — | NCBI E-utilities, publishers | `tools/pubmed_tools/` | — |
| PubMed: per-paper full-text summary | `gemini-3.8-flash` | no thinking setting; `GEMINI_API_KEY` | Google API (google-genai SDK) | `tools/pubmed_tools/get_paper_summary.py:174` (hard-coded) | 43 papers summarized (34 kept, 9 dropped) |
| Web search: result retrieval | none of ours | Google Custom Search API | Google | `tools/google_search_tools/google_search_w3m.py:54` | 5 searches |
| Web search: judge and summarize each page | `gpt-6-luna` | reasoning effort `high`; `OPENAI_API_KEY` | OpenAI API | `tools/google_search_tools/google_search_w3m.py:29–30`, call at `:488`; env `WEB_SEARCH_LLM_MODEL`, `WEB_SEARCH_REASONING_EFFORT` | 52 pages (12 more skipped as too short) |
| Expert knowledge base query (HyperRAG `hyper` mode): keyword extraction and answer | `gpt-6-luna` | reasoning effort `high`; `OPENAI_API_KEY` | OpenAI API, in the agent process | `tools/scientist_rag_tools/scientist_tool.py:61–65`; env `EXPERT_KB_LLM_MODEL`, `EXPERT_KB_REASONING_EFFORT` | 4 (2 queries × 2: GenomicsExpert, NeuroscienceExpert) |
| Expert knowledge base query: entity / relationship retrieval | `text-embedding-3-small` (1536 dims) | `OPENAI_API_KEY` | OpenAI API | `tools/scientist_rag_tools/scientist_tool.py:67–70` (hard-coded) | ≥ 2 (not logged) |
| Knowledge graph: query embedding for the Neo4j vector index | `text-embedding-ada-002` (1536 dims) | `OPENAI_API_KEY` | OpenAI API, from the G-Retriever service | `tools/gretriever_tools/gretriever_service.py:212` (hard-coded) | 1 |
| Knowledge graph: subgraph answer | G-Retriever: GAT (4 layers, 4 heads, 1536 in/out) + fine-tuned Llama-3.1-8B-Instruct, checkpoint `models/g-retriever/0_0_0_gnn-llm-llama3.1-8b_best_val_loss_ckpt.pt`; config and tokenizer from `unsloth/Meta-Llama-3.1-8B-Instruct` (`GRETRIEVER_BASE_LLM`) | GPU, port 8001 | local | `tools/gretriever_tools/gretriever_service.py:98–130`, `scripts/start_kg_service.sh` | 1 (answer: "alzheimer's disease") |
| Report PDF | none | pandoc + xelatex | local | `agent/langgraph_agent.py` `_compile_pdf` | — |

Expert panel in this run: the router picked GenomicsExpert, NeuroscienceExpert and LongevityBiostatsExpert.
LongevityBiostatsExpert has no knowledge base, so its answer came from the agent LLM alone. CancerExpert's
knowledge base was loaded but not selected.

## Models used to build inputs (not called during the run)

| Input | Model | When |
|---|---|---|
| Expert knowledge bases (GenomicsExpert, NeuroscienceExpert, CancerExpert; 50 papers each), entity and hyperedge extraction | `gpt-6-luna`, reasoning effort `high` | 2026-10-24, `scripts/build_expert_kbs.py` |
| Expert knowledge bases, vector stores (`vdb_entities.json`, `vdb_relationships.json`, `vdb_chunks.json`) | `text-embedding-3-small`, 1536 dims | 2026-10-24 |
| G-Retriever checkpoint | GAT + Llama-3.1-8B-Instruct, fine-tuned | before this project's current runs |
| PrimeKG node embeddings in Neo4j (`textEmbedding`, 1536 dims) | not recorded; must share `text-embedding-ada-002`'s space for the query search to work (inferred) | — |
| HyperRAG token counting | `gpt-4o-mini` tokenizer (tiktoken), not a model call | `hyperrag/hyperrag.py:69` |

## Model names in the code that the LangGraph agent does not call

`agent/simple_magentic_agent.py` (`gemini-2.5-pro`, `o4-mini`), `tools/google_search_tools/google_search_scraper_tool.py`
(`gemini-2.5-flash`), `tools/pubmed_tools/query_pubmed_tool.py:723` (`gemini-2.5-pro`, AutoGen agent mode),
`utils/ambiguity.py` and `utils/relevance_validation.py` (`gpt-4o`), `utils/ai_scientist_UI.py` (`gemini-2.5-pro-preview-03-25`).

## What the usage log covered

`usage_log.json` counted the 28 agent-LLM calls (878,263 tokens) and the PubMed summarizer (556,605 tokens):
1,434,868 tokens in total. It did not count the gpt-6-luna calls (52 web-filter + 4 expert), the embeddings,
or the G-Retriever call. Its `not_counted` note named a ScientistsAgent service on port 8000, but the expert
knowledge bases ran inside the agent process in this run.

## Configuration after the switch (2026-10-25)

| Step / agent | Model now | Change |
|---|---|---|
| Planner, replanner, reporter, report revision | `gemini-4-argon` (the only Gemini 4 model the API lists), reasoning effort `high`, through Gemini's OpenAI-compatible endpoint (`create_lead_llm`, `LEAD_MODEL`, CLI `--lead-model`) | was the shared `gemini-3.5-flash` instance |
| All sub-agents, expert routing, expert sub-agents, ScientistsAgent synthesis | `gemini-3.8-flash` (`DEFAULT_MODEL`, CLI `--model`, web UI default) | was `gemini-3.5-flash` (served as 3.6) |
| PubMed full-text summaries | `gemini-3.8-flash` | unchanged |
| Web-search page filter | `gemini-3.8-flash`, reasoning effort `high`, OpenAI-compatible endpoint, `GEMINI_API_KEY` | was `gpt-6-luna` |
| Expert knowledge-base queries (and future builds) | `gemini-3.8-flash`, reasoning effort `high`, OpenAI-compatible endpoint, `GEMINI_API_KEY` | was `gpt-6-luna` |
| Expert knowledge-base embeddings | `text-embedding-3-small` | unchanged (the stored vectors are from this model) |
| G-Retriever, including its `text-embedding-ada-002` query embedding | unchanged | — |

Why the OpenAI-compatible endpoint for Gemini 4: `langchain_google_genai` 3.1 cannot send `thinking_level`
(its `google-ai-generativelanguage` protos, up to 0.12.1, have no such field); the endpoint maps
`reasoning_effort` onto the thinking level. Checked on 2026-10-25: `high` produced reasoning tokens, while
`low` and no setting produced none. `gemini-3.8-flash` accepts reasoning effort none | low | medium | high.
`gemini-4-argon` has a 32,768-token output limit.

The expert knowledge bases were built with `gpt-6-luna`; only the query-time model changed, so the
extracted entities and hyperedges still come from `gpt-6-luna`.

## Restoring this configuration

- Agent: `--model gemini-3.5-flash`, and pass `lead_llm=create_llm("gemini-3.5-flash")` to `LangGraphOmniCellAgent`
  (the planner, replanner, reporter and revision then use the same kind of instance as before; `--lead-model`
  alone keeps the OpenAI-compatible endpoint at high reasoning effort).
- Web filter: `WEB_SEARCH_LLM_MODEL=gpt-6-luna`, `WEB_SEARCH_REASONING_EFFORT=high`, `WEB_SEARCH_LLM_BASE_URL=https://api.openai.com/v1/`,
  and the OpenAI key in `setup_openai()` (it now reads `GEMINI_API_KEY`).
- Expert knowledge bases: `EXPERT_KB_LLM_MODEL=gpt-6-luna`, `EXPERT_KB_REASONING_EFFORT=high`,
  `EXPERT_KB_LLM_BASE_URL=https://api.openai.com/v1/`, and `LLM_API_KEY = os.getenv("OPENAI_API_KEY")` in
  `tools/scientist_rag_tools/scientist_tool.py` (it now reads `GEMINI_API_KEY`). Embeddings are unchanged.
