#!/usr/bin/env python3
"""
Re-embed the built expert knowledge bases with the embedding model in scientist_tool (EMB_MODEL, EMB_DIM),
without re-running the LLM extraction. Needed whenever EMB_MODEL changes: a query embedded by one model
cannot be compared with vectors from another.

HyperRAG keeps three vector files per ragstore but not the text it embedded:
  vdb_entities.json       entity_name + description             (text from the hypergraph vertex)
  vdb_relationships.json  keywords + str(id_set) + description  (text from the hypergraph hyperedge)
  vdb_chunks.json         chunk content                         (text from kv_store_text_chunks.json)
(hyperrag/operate.py extract_entities, hyperrag/hyperrag.py insert). The script rebuilds that text, checks
it on a sample (re-embedding with the old model must reproduce the stored vectors), embeds everything
with the new model, and swaps the files in. The old files are kept as vdb_<name>.<old model>.json.

Usage:
    python scripts/reembed_expert_kbs.py --old-model text-embedding-3-small                # all built experts
    python scripts/reembed_expert_kbs.py --old-model text-embedding-3-small --expert CancerExpert
    python scripts/reembed_expert_kbs.py --old-model text-embedding-3-small --check-only   # sample check only
"""
import os, sys, json, time, asyncio, argparse, random
import numpy as np
import tiktoken

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.scientist_rag_tools.scientist_tool import (
    EXPERT_SCIENTISTS, AuthorKnowledgeBase, EMB_MODEL, EMB_DIM, EMB_API_KEY, EMB_BASE_URL
)
from hyperrag.llm import openai_embedding
from hyperrag.storage import HypergraphStorage
from nano_vectordb.dbs import load_storage, array_to_buffer_string

NAMESPACES = ["entities", "relationships", "chunks"]
MAX_INPUT_TOKENS = 8191        # per text, OpenAI embedding limit
BATCH_TOKENS = 200_000         # per request (limit 300k)
BATCH_TEXTS = 512              # per request (limit 2048)
CONCURRENT_REQUESTS = 6
CHECK_SAMPLE = 200             # texts per vector file re-embedded with the old model
CHECK_MIN_COSINE = 0.995       # OpenAI embeddings of identical text agree to ~1e-4

_enc = tiktoken.get_encoding("cl100k_base")


def rebuild_texts(working_dir: str, namespace: str, data: list) -> list:
    """The text HyperRAG embedded for each stored vector, in storage order (None if the source is missing)."""
    if namespace == "chunks":
        with open(os.path.join(working_dir, "kv_store_text_chunks.json")) as f:
            chunks = json.load(f)
        return [chunks.get(d["__id__"], {}).get("content") for d in data]
    hg = HypergraphStorage.load_hypergraph(os.path.join(working_dir, "hypergraph_chunk_entity_relation.hgdb"))
    texts = []
    for d in data:
        if namespace == "entities":
            v = hg.v(d["entity_name"]) if hg.has_v(d["entity_name"]) else None
            texts.append(d["entity_name"] + v["description"] if v else None)
        else:
            # id_set was a tuple when embedded; JSON stored it as a list
            id_set = tuple(d["id_set"])
            e = hg.e(id_set) if hg.has_e(id_set) else None
            texts.append(e["keywords"] + str(id_set) + e["description"] if e else None)
    return texts


def batches(texts: list) -> list:
    """(start, end) index ranges that respect the per-request text and token limits."""
    out, start, tokens = [], 0, 0
    for i, t in enumerate(texts):
        n = min(len(_enc.encode(t, disallowed_special=())), MAX_INPUT_TOKENS)
        if i > start and (i - start >= BATCH_TEXTS or tokens + n > BATCH_TOKENS):
            out.append((start, i))
            start, tokens = i, 0
        tokens += n
    if start < len(texts):
        out.append((start, len(texts)))
    return out


def truncate(text: str) -> str:
    tokens = _enc.encode(text, disallowed_special=())
    return text if len(tokens) <= MAX_INPUT_TOKENS else _enc.decode(tokens[:MAX_INPUT_TOKENS])


async def embed(texts: list, model: str, label: str) -> np.ndarray:
    texts = [truncate(t) for t in texts]
    ranges = batches(texts)
    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)
    done, t0 = 0, time.time()

    async def one(a, b):
        nonlocal done
        async with sem:
            for attempt in range(6):
                try:
                    vectors = await openai_embedding(texts[a:b], model=model, api_key=EMB_API_KEY,
                                                     base_url=EMB_BASE_URL)
                    break
                except Exception as e:
                    if attempt == 5:
                        raise
                    print(f"    retry {attempt + 1} for texts {a}-{b}: {type(e).__name__}: {str(e)[:120]}")
                    await asyncio.sleep(30 * (attempt + 1))
            done += 1
            if done % 50 == 0 or done == len(ranges):
                print(f"    {label}: {done}/{len(ranges)} requests, {time.time() - t0:.0f}s", flush=True)
            return a, vectors

    out = np.zeros((len(texts), 0), dtype=np.float32)
    results = await asyncio.gather(*[one(a, b) for a, b in ranges])
    for a, vectors in sorted(results, key=lambda r: r[0]):
        if out.shape[1] == 0:
            out = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
        out[a:a + len(vectors)] = vectors
    return out


def unit(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True)


async def check_sample(storage: dict, texts: list, old_model: str, label: str) -> bool:
    """Re-embed a sample with the old model; the result must match the stored vectors."""
    idx = random.Random(0).sample(range(len(texts)), min(CHECK_SAMPLE, len(texts)))
    new = await embed([texts[i] for i in idx], old_model, f"{label} check")
    cos = np.sum(unit(new) * unit(storage["matrix"][idx]), axis=1)
    bad = int(np.sum(cos < CHECK_MIN_COSINE))
    print(f"    {label}: rebuilt text vs stored vectors over {len(idx)} samples: cosine min {cos.min():.5f}, "
          f"median {np.median(cos):.5f}, below {CHECK_MIN_COSINE}: {bad}")
    return bad == 0


async def convert(alias: str, working_dir: str, old_model: str, check_only: bool) -> bool:
    print(f"\n=== {alias}: {working_dir}")
    ok = True
    for ns in NAMESPACES:
        path = os.path.join(working_dir, f"vdb_{ns}.json")
        backup = os.path.join(working_dir, f"vdb_{ns}.{old_model}.json")
        storage = load_storage(path)
        if storage is None:
            print(f"  {ns}: no vector file, skipped")
            continue
        if storage["embedding_dim"] == EMB_DIM and os.path.exists(backup):
            print(f"  {ns}: already {EMB_DIM}-dim with a {old_model} backup, skipped")
            continue
        data = storage["data"]
        texts = rebuild_texts(working_dir, ns, data)
        missing = sum(t is None for t in texts)
        tokens = sum(len(_enc.encode(t, disallowed_special=())) for t in texts if t)
        print(f"  {ns}: {len(data):,} vectors ({storage['embedding_dim']} dims), {missing} without source text, "
              f"{tokens:,} tokens")
        if missing:
            print(f"  ❌ {ns}: cannot rebuild the text of {missing} vectors; nothing converted for {alias}")
            return False
        if not await check_sample(storage, texts, old_model, ns):
            print(f"  ❌ {ns}: rebuilt text does not reproduce the stored vectors; nothing converted for {alias}")
            return False
        if check_only:
            continue
        if os.path.exists(backup):
            print(f"  ❌ {ns}: backup {backup} already exists; not overwriting it")
            return False
        matrix = await embed(texts, EMB_MODEL, ns)
        if matrix.shape != (len(data), EMB_DIM):
            print(f"  ❌ {ns}: got vectors of shape {matrix.shape}, expected ({len(data)}, {EMB_DIM})")
            return False
        new_storage = {k: v for k, v in storage.items() if k != "matrix"}
        new_storage["embedding_dim"] = EMB_DIM
        new_storage["matrix"] = array_to_buffer_string(matrix)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(new_storage, f, ensure_ascii=False)
        os.replace(path, backup)
        os.replace(tmp, path)
        print(f"  ✅ {ns}: {len(data):,} vectors re-embedded with {EMB_MODEL}; old file kept as {os.path.basename(backup)}")
    return ok


async def main(aliases, old_model, check_only):
    print(f"New embedding model: {EMB_MODEL} ({EMB_DIM} dims); old: {old_model}")
    results = {}
    for scientist in EXPERT_SCIENTISTS:
        alias = scientist["alias"]
        if aliases and alias not in aliases:
            continue
        kb = AuthorKnowledgeBase(cache_key=scientist["cache_key"], pubmed_name=scientist["real_name"])
        if not os.path.exists(os.path.join(kb.working_dir, "vdb_entities.json")):
            print(f"\n⚠️  Skipping {alias}: no built ragstore")
            continue
        results[alias] = await convert(alias, kb.working_dir, old_model, check_only)
    print("\nSummary:", {a: ("ok" if r else "FAILED") for a, r in results.items()})
    return all(results.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-embed expert knowledge bases with scientist_tool.EMB_MODEL")
    parser.add_argument("--old-model", required=True, help="embedding model of the current vector files")
    parser.add_argument("--expert", action="append", choices=[e["alias"] for e in EXPERT_SCIENTISTS],
                        help="convert only this expert (repeatable)")
    parser.add_argument("--check-only", action="store_true",
                        help="only check that the rebuilt text reproduces the stored vectors")
    args = parser.parse_args()
    sys.exit(0 if asyncio.run(main(args.expert, args.old_model, args.check_only)) else 1)
