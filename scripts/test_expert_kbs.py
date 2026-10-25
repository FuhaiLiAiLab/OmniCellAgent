#!/usr/bin/env python3
"""
Test the built expert knowledge bases. Nothing is built or downloaded; experts without a ragstore are skipped.

For each built expert and each case-study question it is routed to:
  1. Retrieve: HyperRAG context only (1 LLM call: keyword extraction). Lists which of the expert's
     papers the retrieved source chunks come from.
  2. Answer: the query the agent runs (AuthorKnowledgeBase.query_system, mode 'hyper'; 2 LLM calls).
     Flags HyperRAG's fail response and query errors.
Plus one off-topic control question per expert (answer only). HyperRAG's answer prompt allows
"relevant general knowledge", so a confident control answer means answers mix model knowledge
with the papers.

Query responses are not cached (the ragstore is left unchanged), and the keyword-extraction call varies
from run to run, so reruns can retrieve different papers. Full answers go to logs/expert_kb_test_<timestamp>.md.

Usage:
    python scripts/test_expert_kbs.py                           # all built experts
    python scripts/test_expert_kbs.py --expert CancerExpert     # one expert (repeatable)
    python scripts/test_expert_kbs.py --expert NeuroscienceExpert --question "TREM2 microglia synapse loss"
    python scripts/test_expert_kbs.py --context-only            # retrieval only (1 LLM call per question)
"""
import sys, os, re, json, time, asyncio, argparse, resource
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.scientist_rag_tools.scientist_tool import (
    EXPERT_SCIENTISTS, AuthorKnowledgeBase, LLM_MODEL, LLM_REASONING_EFFORT
)
from hyperrag import QueryParam
from hyperrag.prompt import PROMPTS
from hyperrag.utils import compute_mdhash_id

AD = "What are the key dysfunctional genes and pathways in Alzheimer's Disease?"
PDAC = "What are the key dysfunctional genes and pathways in pancreatic ductal adenocarcinoma (PDAC)?"
LUAD = "What are the key dysfunctional genes and pathways in Lung adenocarcinoma (LUAD)?"
# Case-study questions each expert is routed to (ScientistsAgent routing, 3 trials per case study)
CASE_QUESTIONS = {
    "GenomicsExpert": [AD, PDAC, LUAD],
    "NeuroscienceExpert": [AD],
    "CancerExpert": [PDAC, LUAD],
}
CONTROL = "How do C4 plants concentrate CO2 around RuBisCO?"
MAX_PAPERS_SHOWN = 8


def _norm(s: str) -> str:
    # HyperRAG's context CSV re-splits rows on commas and doubles quotes, so drop both before comparing
    return re.sub(r'[\s",]+', ' ', s).strip()


async def paper_titles(kb: AuthorKnowledgeBase) -> dict:
    """doc id -> paper title, recomputed from doi_cache the way build_knowledge_base inserted the papers."""
    titles = {}
    for fname in os.listdir(kb.jsonl_cache_dir):
        if fname.endswith(".jsonl"):
            with open(os.path.join(kb.jsonl_cache_dir, fname)) as f:
                for line in f:
                    p = json.loads(line)
                    if p.get("doi"):
                        titles[p["doi"].replace("/", "_")] = p.get("title") or ""
    docs = {}
    for fname in sorted(os.listdir(kb.doi_cache_dir)):
        path = os.path.join(kb.doi_cache_dir, fname)
        if fname.endswith(".pdf"):
            text = await kb._extract_text_from_pdf(path)
        elif fname.endswith(".xml"):
            text = await kb._extract_text_from_xml(path)
        else:
            continue
        if len(text) >= kb.MIN_FULLTEXT_CHARS:
            stem = os.path.splitext(fname)[0]
            docs[compute_mdhash_id(text, prefix="doc-")] = titles.get(stem) or stem
    return docs


def chunk_docs(kb: AuthorKnowledgeBase) -> dict:
    """normalized chunk prefix -> doc id."""
    with open(os.path.join(kb.working_dir, "kv_store_text_chunks.json")) as f:
        chunks = json.load(f)
    return {_norm(c["content"])[:150]: c["full_doc_id"] for c in chunks.values()}


def source_papers(context: str, prefixes: dict, titles: dict) -> list:
    """(title, retrieved chunks) for the papers in a context's Sources section, most chunks first."""
    sources = context.split("-----Sources-----", 1)[-1]
    counts = {}
    for row in re.split(r'\n\s*\d+,', sources)[1:]:
        text = _norm(row)
        # Chunks overlap by 100 tokens, so a row can also contain the next chunk's prefix; both
        # belong to the same paper
        doc_id = next((d for p, d in prefixes.items() if p and p in text), "unmatched chunk")
        counts[doc_id] = counts.get(doc_id, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return [(titles.get(doc_id, doc_id), n) for doc_id, n in ranked]


async def run_question(kb, question, prefixes, titles, context_only, answer_only=False):
    out = {"question": question}
    if not answer_only:
        t0 = time.time()
        context = await kb.rag.aquery(question, param=QueryParam(mode="hyper", only_need_context=True))
        out["context_seconds"] = time.time() - t0
        out["context_chars"] = len(context or "")
        out["context_failed"] = context == PROMPTS["fail_response"]
        out["papers"] = [] if out["context_failed"] else source_papers(context, prefixes, titles)
    if not context_only:
        t0 = time.time()
        answer = await kb.query_system(question)
        out["answer_seconds"] = time.time() - t0
        out["answer"] = answer
        out["answer_failed"] = (answer == PROMPTS["fail_response"]
                                or str(answer).startswith("Error querying"))
    return out


async def test_expert(scientist, questions, context_only, report):
    alias, cache_key = scientist["alias"], scientist["cache_key"]
    kb = AuthorKnowledgeBase(cache_key=cache_key, pubmed_name=scientist["real_name"], top_k=scientist["top_k"])
    if not os.path.exists(os.path.join(kb.working_dir, "kv_store_full_docs.json")):
        print(f"\n⚠️  Skipping {alias}: no built KB in {kb.working_dir}")
        return []

    print(f"\n{'='*60}\n🧪 {alias}\n{'='*60}", flush=True)
    t0 = time.time()
    await kb.initialize_rag()
    load_seconds = time.time() - t0
    with open(os.path.join(kb.working_dir, "kv_store_full_docs.json")) as f:
        n_docs = len(json.load(f))
    hg = kb.rag.chunk_entity_relation_hypergraph._hg
    titles = await paper_titles(kb)
    prefixes = chunk_docs(kb)
    print(f"  Loaded in {load_seconds:.0f}s: {n_docs} papers, {len(prefixes)} chunks, "
          f"{hg.num_v} vertices, {hg.num_e} hyperedges")
    unmatched = n_docs - len(set(titles) & {d for d in prefixes.values()})
    if unmatched:
        print(f"  ⚠️  {unmatched} stored papers have no matching file in doi_cache (titles shown as doc ids)")

    tasks = [run_question(kb, q, prefixes, titles, context_only) for q in questions]
    if not context_only:
        tasks.append(run_question(kb, CONTROL, prefixes, titles, context_only=False, answer_only=True))
    results = await asyncio.gather(*tasks)

    report.append(f"\n## {alias}\n\nLoaded in {load_seconds:.0f}s: {n_docs} papers, {len(prefixes)} chunks, "
                  f"{hg.num_v} vertices, {hg.num_e} hyperedges.\n")
    for r in results:
        control = r["question"] == CONTROL
        print(f"\n  Q{' (control)' if control else ''}: {r['question']}")
        report.append(f"\n### {'Control: ' if control else ''}{r['question']}\n")
        if "papers" in r:
            status = "❌ fail response" if r["context_failed"] else f"{len(r['papers'])} papers"
            print(f"    Retrieval: {r['context_seconds']:.0f}s, {r['context_chars']:,} chars, {status}")
            report.append(f"Retrieval: {r['context_seconds']:.0f}s, {r['context_chars']:,} chars, {status}\n")
            for title, n in r["papers"][:MAX_PAPERS_SHOWN]:
                print(f"      {n:2d} chunks  {title[:100]}")
            report.extend(f"- {n} chunks: {title}" for title, n in r["papers"])
        if "answer" in r:
            status = "❌ FAILED" if r["answer_failed"] else f"{len(r['answer']):,} chars"
            print(f"    Answer: {r['answer_seconds']:.0f}s, {status}")
            preview = re.sub(r"\s+", " ", str(r["answer"]))[:300]
            print(f"      {preview}...")
            report.append(f"\nAnswer ({r['answer_seconds']:.0f}s, {status}):\n\n{r['answer']}\n")
    return [dict(r, expert=alias) for r in results]


async def main(aliases=None, question=None, context_only=False):
    print(f"KB query model: {LLM_MODEL}, reasoning effort: {LLM_REASONING_EFFORT}")
    report = [f"# Expert KB test {datetime.now():%Y-%m-%d %H:%M}\n\n"
              f"Model {LLM_MODEL}, reasoning effort {LLM_REASONING_EFFORT}.\n"]
    all_results = []
    for scientist in EXPERT_SCIENTISTS:
        alias = scientist["alias"]
        if aliases is not None and alias not in aliases:
            continue
        questions = [question] if question else CASE_QUESTIONS.get(alias, [])
        if not questions:
            print(f"\n⚠️  Skipping {alias}: no case-study question for it (use --question)")
            continue
        all_results += await test_expert(scientist, questions, context_only, report)

    os.makedirs("logs", exist_ok=True)
    report_path = os.path.join("logs", f"expert_kb_test_{datetime.now():%Y%m%d_%H%M%S}.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report))

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    failed = [r for r in all_results if r.get("context_failed") or r.get("answer_failed")]
    print(f"  {len(all_results)} queries, {len(failed)} failed")
    for r in failed:
        print(f"  ❌ {r['expert']}: {r['question']}")
    print(f"  Peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
    print(f"  Full answers: {report_path}")
    print(f"{'='*60}")
    return not failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the built expert knowledge bases")
    parser.add_argument("--expert", action="append", choices=[e["alias"] for e in EXPERT_SCIENTISTS],
                        help="test only this expert (alias from configs/experts.yaml); repeatable")
    parser.add_argument("--question", help="ask this instead of the case-study questions")
    parser.add_argument("--context-only", action="store_true",
                        help="retrieval only: skip the answers and the control question")
    args = parser.parse_args()
    ok = asyncio.run(main(args.expert, args.question, args.context_only))
    sys.exit(0 if ok else 1)
