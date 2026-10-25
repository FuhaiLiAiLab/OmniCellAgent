"""Step 2: generate benchmark candidates with draft labels for annotation.

Reads the frozen vocabulary from step 1 and pools candidate test strings from
five sources:

* Cell Ontology / MONDO labels and synonyms of concepts that are in the DB
* LRABR (NLM SPECIALIST) abbreviations whose expansion names a DB concept
* CellLink train+val cell mentions (linked to Cell Ontology ids)
* MedMentions disease mentions (linked to UMLS CUIs, joined through MONDO)
* ontology siblings of DB concepts that are absent from the DB (near misses)

Ambiguous abbreviations found in the corpora become context items: the same
string in real sentences where it means different concepts. Every candidate
carries a machine-drafted action and gold set (see drafting.py) for human
review; nothing here is a gold label yet.

Usage (from the project root, after build_vocabulary):
    python -m tools.omic_tools.tests.name_extraction.generate_candidates
"""

from __future__ import annotations

import argparse
import datetime
import difflib
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.omic_tools.tests.name_extraction import sources  # noqa: E402
from tools.omic_tools.tests.name_extraction.build_vocabulary import (  # noqa: E402
    DEFAULT_OUT as DEFAULT_VOCAB, git_state, read_tsv, write_tsv)
from tools.omic_tools.tests.name_extraction.corpora import (  # noqa: E402
    DISEASE_SEMTYPES, CorpusMention, is_disease_mention, load_celllink, load_lrabr, load_medmentions)
from tools.omic_tools.tests.name_extraction.drafting import (  # noqa: E402
    TRAP_SCORE, FieldIndex, Relation)
from tools.omic_tools.tests.name_extraction.obo import Ontology  # noqa: E402
from tools.omic_tools.tests.name_extraction.text import (  # noqa: E402
    abbreviation_core, norm_basic, norm_loose)

import pandas as pd  # noqa: E402

DEFAULT_OUT = HERE / "candidates"
CATEGORIES = ("synonym", "abbreviation", "surface_variant", "granularity", "absent_concept",
              "context_abbreviation", "near_miss")
ID_PREFIX = {"synonym": "syn", "abbreviation": "abb", "surface_variant": "var",
             "granularity": "gra", "absent_concept": "abs", "context_abbreviation": "ctx",
             "near_miss": "nm"}
KEEP_ALL_ACTIONS = {"resolve", "choose"}
MAX_EXAMPLES = 2


@dataclass
class Candidate:
    field: str
    concept: str
    concept_label: str | None
    ontology: str
    relation: Relation
    casing: Counter = field(default_factory=Counter)
    sources: list[dict] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)
    corpus_count: int = 0
    context: str | None = None
    pair_id: str | None = None
    fixed_category: str | None = None
    forced: tuple[str, str] | None = None     # (action, basis) overriding the draft

    @property
    def mention(self) -> str:
        return max(self.casing.items(), key=lambda kv: (kv[1], -len(kv[0]), kv[0]))[0]


class Pool:
    """Context-free candidates merged by (field, normalized mention, concept)."""

    def __init__(self):
        self.items: dict[tuple[str, str, str], Candidate] = {}
        self.trivial: Counter = Counter()

    def add(self, index: FieldIndex, mention: str, concept: str, label, ontology: str,
            relation: Relation, source: dict, *, count: int = 0, examples=(),
            casing: Counter | None = None) -> None:
        """Pool one source's evidence; ``casing`` counts corpus spellings of the mention."""
        mention = " ".join(str(mention).split())
        if not mention:
            return
        if index.is_trivial(mention):
            self.trivial[(index.field, source["source"])] += count or 1
            return
        key = (index.field, norm_basic(mention), concept)
        candidate = self.items.get(key)
        if candidate is None:
            candidate = self.items[key] = Candidate(index.field, concept, label, ontology, relation)
        candidate.casing.update(casing or {mention: 1})
        candidate.corpus_count += count
        if source not in candidate.sources:
            candidate.sources.append(source)
        for example in examples:
            if len(candidate.examples) >= MAX_EXAMPLES:
                break
            if all(existing["doc"] != example["doc"] for existing in candidate.examples):
                candidate.examples.append(example)


# ------------------------------------------------------------------ generators
def add_ontology_forms(pool: Pool, index: FieldIndex, source_key: str, prefix: str) -> None:
    """Labels and synonyms of every ontology concept linked to a DB term."""
    for node in sorted(index.node_to_names):
        term = index.ontology.terms.get(node)
        if term is None or term.is_obsolete:
            continue
        relation = index.relate([node])
        pool.add(index, term.name or "", node, term.name, prefix, relation,
                 {"source": source_key, "kind": "label", "review_reason": None})
        for synonym in term.synonyms:
            reason = None if synonym.scope == "EXACT" else f"{synonym.scope} synonym in {prefix}"
            pool.add(index, synonym.text, node, term.name, prefix, relation,
                     {"source": source_key, "kind": "synonym", "scope": synonym.scope,
                      "synonym_type": synonym.type, "review_reason": reason})


def add_lrabr(pool: Pool, index: FieldIndex, lrabr, prefix: str) -> None:
    """Abbreviations whose LRABR expansion equals a DB concept's name or EXACT synonym."""
    def forms_for(node):
        term = index.ontology.terms.get(node)
        if term is None:
            return set()
        return {term.name or ""} | {s.text for s in term.synonyms if s.scope == "EXACT"}

    targets = []  # (long forms, concept, label, relation)
    for node in sorted(index.node_to_names):
        targets.append((forms_for(node), node, index.ontology.label(node), index.relate([node])))
    for name in index.names:
        nodes = sorted(node for node, names in index.node_to_names.items() if name in names)
        forms = {name} | {part.split(":", 1)[0] for part in name.split(";")}
        concept = nodes[0] if len(nodes) == 1 else f"DB:{name}"
        label = index.ontology.label(nodes[0]) if len(nodes) == 1 else name
        targets.append((forms, concept, label, index.relate(nodes, db_names=[name])))

    for forms, concept, label, relation in targets:
        for form in sorted(f for f in forms if f.strip()):
            for abbreviation, sense in sorted(lrabr.by_expansion.get(norm_loose(form), ())):
                if abbreviation_core(abbreviation, lrabr.abbreviations) is None:
                    continue
                pool.add(index, abbreviation, concept, label,
                         prefix if not concept.startswith("DB:") else "DB", relation,
                         {"source": "lrabr", "expansion": form, "sense": sense,
                          "n_senses": len(lrabr.senses[abbreviation]), "review_reason": None})


def _examples(mentions: list[CorpusMention], rng: random.Random, limit: int) -> list[CorpusMention]:
    by_doc = {}
    for mention in sorted(mentions, key=lambda m: (m.doc, m.context)):
        by_doc.setdefault(mention.doc, mention)
    chosen = list(by_doc.values())
    rng.shuffle(chosen)
    return chosen[:limit]


def add_corpus(pool: Pool, index: FieldIndex, mentions: list[CorpusMention], *, seed: int,
               relate, label_of, canonical) -> None:
    """Aggregate corpus mentions by (surface, concept, link) and pool them.

    ``canonical(concept) -> (key, ontology)`` lets a UMLS CUI with a MONDO
    equivalent pool under the MONDO id, so 'SLE' from MedMentions and 'SLE'
    from MONDO/LRABR become one candidate.
    """
    groups: dict[tuple, list[CorpusMention]] = defaultdict(list)
    for mention in mentions:
        key, _ = canonical(mention.concept)
        groups[(norm_basic(mention.text), key, mention.link)].append(mention)
    relations = {}
    for (_, key, link), group in sorted(groups.items()):
        concepts = sorted({m.concept for m in group})
        if key not in relations:
            relations[key] = relate(concepts[0])
        rng = random.Random(f"{seed}:{group[0].corpus}:{group[0].text}:{key}")
        reason = "CellLink related (not exact) link" if link == "related" else None
        source = {
            "source": group[0].corpus,
            "link": link,
            "annotated_concepts": concepts,
            "entity_types": sorted({m.entity_type for m in group}),
            "count": len(group),
            "n_docs": len({m.doc for m in group}),
            "splits": dict(sorted(Counter(m.split for m in group).items())),
            "review_reason": reason,
        }
        examples = [{"doc": m.doc, "split": m.split, "text": m.context}
                    for m in _examples(group, rng, MAX_EXAMPLES)]
        spellings = Counter(" ".join(m.text.split()) for m in group)
        pool.add(index, spellings.most_common(1)[0][0], key, label_of(concepts[0]),
                 canonical(concepts[0])[1], relations[key], source, count=len(group),
                 examples=examples, casing=spellings)


def context_candidates(indexes: dict[str, FieldIndex], celllink, medmentions, lrabr, *,
                       per_sense: int, seed: int, label_of) -> tuple[list[Candidate], dict]:
    """The same abbreviation in real sentences where it names different concepts.

    Senses are (field, concept) pairs pooled across both corpora, so 'ILC' as an
    innate lymphoid cell (CellLink) and as invasive lobular carcinoma
    (MedMentions) form one contrastive group. A group is kept when it has at
    least two senses and at least one sense has an equal or narrower DB term.
    MedMentions senses whose semantic types are not disease-like are drafted
    'decline' for the disease field.
    """
    cores: dict[str, str | None] = {}

    def is_abbreviation(text: str) -> bool:
        if text not in cores:
            cores[text] = abbreviation_core(text, lrabr.abbreviations)
        return cores[text] is not None

    groups: dict[str, list[tuple[str, CorpusMention]]] = defaultdict(list)
    for mention in celllink:
        if mention.link == "exact" and is_abbreviation(mention.text):
            groups[mention.text].append(("cell_type", mention))
    for mention in medmentions:
        if is_abbreviation(mention.text):
            groups[mention.text].append(("disease", mention))

    candidates, summary = [], {}
    relation_cache = {}
    for surface in sorted(groups):
        senses: dict[tuple[str, str], list[CorpusMention]] = defaultdict(list)
        for field_name, mention in groups[surface]:
            senses[(field_name, mention.concept)].append(mention)
        if len(senses) < 2:
            continue
        resolved = {}
        for (field_name, concept), mentions in senses.items():
            index = indexes[field_name]
            not_disease = field_name == "disease" and not any(is_disease_mention(m) for m in mentions)
            if (field_name, concept) not in relation_cache:
                if not_disease:
                    relation_cache[(field_name, concept)] = Relation()
                elif field_name == "disease":
                    relation_cache[(field_name, concept)] = index.relate(umls=concept)
                else:
                    relation_cache[(field_name, concept)] = index.relate([concept])
            forced = None
            if not_disease:
                types = sorted({t for m in mentions for t in m.entity_type.split(",")})
                forced = ("decline", f"this sense is not a disease (UMLS semantic types {', '.join(types)})")
            resolved[(field_name, concept)] = (relation_cache[(field_name, concept)], forced)
        if not any(rel.equal or rel.narrower for rel, forced in resolved.values() if not forced):
            continue
        summary[surface] = {f"{f}:{c}": len(ms) for (f, c), ms in sorted(senses.items())}
        for (field_name, concept), mentions in sorted(senses.items()):
            relation, forced = resolved[(field_name, concept)]
            rng = random.Random(f"{seed}:ctx:{surface}:{field_name}:{concept}")
            for mention in _examples(mentions, rng, per_sense):
                candidate = Candidate(field_name, concept, label_of(field_name, concept),
                                      "CL" if field_name == "cell_type" else "UMLS", relation,
                                      context=mention.context, pair_id=surface,
                                      fixed_category="context_abbreviation", forced=forced)
                candidate.casing[mention.text] = 1
                candidate.sources.append({
                    "source": mention.corpus, "doc": mention.doc, "split": mention.split,
                    "link": mention.link, "entity_types": [mention.entity_type],
                    "review_reason": None})
                candidates.append(candidate)
    return candidates, summary


def near_miss_candidates(index: FieldIndex, prefix: str, *, per_concept: int, min_score: float,
                         cap: int) -> list[Candidate]:
    """Ontology siblings of DB concepts that are unrelated to every DB term.

    These are traps: lexically close to a DB name, same parent, absent from
    the data ('acute pancreatitis' next to the DB's chronic pancreatitis).
    """
    scored = []
    for node in sorted(index.node_to_names):
        db_name = sorted(index.node_to_names[node], key=lambda n: (-index.n_rows.get(n, 0), n))[0]
        best = []
        for parent in index.ontology.parents(node):
            for sibling in index.ontology.children(parent):
                if sibling == node or sibling in index.node_to_names:
                    continue
                term = index.ontology.terms[sibling]
                if term.is_obsolete or not term.name or index.is_trivial(term.name):
                    continue
                score = difflib.SequenceMatcher(None, norm_loose(term.name), norm_loose(db_name)).ratio()
                if score < min_score:
                    continue
                relation = index.relate([sibling])
                if index.draft(relation)["expected_action"] != "decline":
                    continue
                best.append((score, sibling, term.name, relation))
        best.sort(key=lambda item: (-item[0], item[1]))
        for score, sibling, label, relation in best[:per_concept]:
            scored.append((score, sibling, label, relation, node, db_name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    candidates, seen = [], set()
    for score, sibling, label, relation, node, db_name in scored:
        if sibling in seen:
            continue
        seen.add(sibling)
        candidate = Candidate(index.field, sibling, label, prefix, relation, fixed_category="near_miss")
        candidate.casing[label] = 1
        candidate.sources.append({"source": f"{prefix} sibling", "near_miss_of": db_name,
                                  "db_concept": node, "similarity": round(score, 3),
                                  "review_reason": None})
        candidates.append(candidate)
        if len(candidates) >= cap:
            break
    return candidates


# ---------------------------------------------------------- sense inventory
def loose_concept_index(ontology: Ontology) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for term in ontology.live_terms():
        for text in [term.name or ""] + [s.text for s in term.synonyms if s.scope == "EXACT"]:
            if text:
                index[norm_loose(text)].add(term.id)
    return index


def sense_inventory(abbreviations, lrabr, indexes, loose_indexes, context_summary,
                    max_unmapped: int = 12) -> list[dict]:
    """Every LRABR sense of each abbreviation, placed against the DB where possible."""
    records = []
    for abbreviation in sorted(abbreviations):
        mapped_rows, unmapped_rows = [], []
        for sense, expansions in sorted(lrabr.senses.get(abbreviation, {}).items()):
            placements = []
            for field_name, loose in loose_indexes.items():
                nodes = sorted(set().union(*(loose.get(norm_loose(e), set()) for e in expansions)))
                if not nodes:
                    continue
                index = indexes[field_name]
                relation = index.relate(nodes)
                status = ("equal" if relation.equal else "narrower" if relation.narrower
                          else "broader" if relation.broader else "absent")
                placements.append({
                    "field": field_name, "concepts": nodes,
                    "labels": [index.ontology.label(n) for n in nodes],
                    "db_status": status,
                    "db_names": sorted(relation.equal or relation.narrower or relation.broader)[:5]})
            row = {"sense": sense, "expansions": expansions[:6], "placements": placements}
            (mapped_rows if placements else unmapped_rows).append(row)
        statuses = {p["db_status"] for row in mapped_rows for p in row["placements"]}
        if not mapped_rows and abbreviation not in context_summary:
            continue
        records.append({
            "abbreviation": abbreviation,
            "n_lrabr_senses": len(lrabr.senses.get(abbreviation, {})),
            "n_unmapped_senses": len(unmapped_rows),
            "has_db_sense": bool(statuses & {"equal", "narrower"}),
            "contrastive_ready": bool(statuses & {"equal", "narrower"})
                                 and (bool(unmapped_rows) or bool(statuses & {"absent", "broader"})),
            "corpus_senses": context_summary.get(abbreviation, {}),
            "senses": mapped_rows + unmapped_rows[:max_unmapped],
        })
    return records


# ------------------------------------------------------------------ finalize
def _digest(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def surface_concepts(ontology: Ontology) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for term in ontology.live_terms():
        for text in [term.name or ""] + [s.text for s in term.synonyms]:
            if text:
                index[norm_basic(text)].add(term.id)
    return index


def names_concept(mention: str, concept: str | None, ontology: Ontology) -> bool:
    """True if the mention is the concept's label or an EXACT synonym, up to norm_loose.

    Such a string names the concept without context. It outranks a corpus
    majority ('germ cell' is annotated as spermatogonium in testis papers) and
    needs no review for weak evidence ('corneal fibroblasts', the plural of a
    keratocyte EXACT synonym, seen only through a CellLink related link).
    """
    term = ontology.terms.get(concept) if concept else None
    if term is None or term.is_obsolete:
        return False
    names = [term.name or ""] + [s.text for s in term.synonyms if s.scope == "EXACT"]
    return norm_loose(mention) in {norm_loose(name) for name in names if name}


def finalize(candidate: Candidate, index: FieldIndex, lrabr, ontology_surfaces, corpus_surfaces) -> dict:
    mention = candidate.mention
    is_name = names_concept(mention, candidate.concept, index.ontology)
    reasons = [source.get("review_reason") for source in candidate.sources]
    review_reason = None if is_name or any(reason is None for reason in reasons) \
        else "; ".join(dict.fromkeys(reasons))
    draft = index.draft(candidate.relation, review_reason)
    if candidate.forced:
        draft.update(expected_action=candidate.forced[0], basis=candidate.forced[1],
                     gold_preferred=None, gold_equal=[], gold_narrower=[], n_narrower_total=0,
                     gold_broader=[], nearest_broader_depth=None)
    category, surface = index.categorize(mention, draft, lrabr.abbreviations)
    category = candidate.fixed_category or category
    surface["names_concept"] = is_name
    surface["loose_variants"] = []  # filled by mark_loose_variants after capping
    if candidate.context is not None:
        # A glossary sentence ("DC, dendritic cell; ...") spells out the answer;
        # a table row ("OPC 1524 13 0.1067 ...") is not a usable query context.
        surface["context_contains_label"] = bool(candidate.concept_label) and \
            norm_loose(candidate.concept_label) in norm_loose(candidate.context)
        surface["context_is_table_like"] = \
            sum(c.isdigit() for c in candidate.context) > 0.15 * max(len(candidate.context), 1)
    neighbours = index.neighbours(mention)
    accepted = (set(draft["gold_equal"]) | {i["name"] for i in draft["gold_narrower"]}
                | {i["name"] for i in draft["gold_broader"]})
    trap = bool(neighbours) and neighbours[0]["score"] >= TRAP_SCORE and neighbours[0]["name"] not in accepted
    key = f"{candidate.field}|{norm_basic(mention)}|{candidate.concept}|{candidate.context or ''}"
    core = surface["abbreviation_core"]
    return {
        "id": f"{ID_PREFIX[category]}-{_digest(key)}",
        "category": category,
        "field": candidate.field,
        "mention": mention,
        "query": None,
        "context": candidate.context,
        "pair_id": candidate.pair_id,
        "intended_concept": {"id": candidate.concept, "label": candidate.concept_label,
                             "ontology": candidate.ontology},
        "draft": draft,
        "surface": surface,
        "ambiguity": {
            "lrabr_senses": len(lrabr.senses.get(core, {})) if core else 0,
            "ontology_concepts": len(ontology_surfaces.get(norm_basic(mention), ())),
            "corpus_concepts": len(corpus_surfaces.get(norm_basic(mention), ())),
            "n_concepts": 1,
            "db_senses": int(draft["expected_action"] == "resolve"),
            "concept_disagreement": False,
            "other_field_concepts": 0,
            "loose_variant_conflict": False,
        },
        "alternatives": [],
        "lexical_neighbors": neighbours,
        "lexical_trap": trap,
        "corpus_count": candidate.corpus_count,
        "sources": [{k: v for k, v in s.items() if k != "review_reason" or v} for s in candidate.sources],
        "examples": candidate.examples,
        "labels": {"annotator_a": None, "annotator_b": None, "adjudicated": None, "notes": None},
    }


def _evidence_rank(record: dict) -> int:
    """2: the concept's own name (label or EXACT synonym); 1: LRABR or other
    synonym scope; 0: corpus only."""
    if record["surface"]["names_concept"]:
        return 2
    rank = 0
    for source in record["sources"]:
        if source["source"] in ("cell_ontology", "mondo"):
            exact = source.get("kind") == "label" or source.get("scope") == "EXACT"
            rank = max(rank, 2 if exact else 1)
        elif source["source"] == "lrabr":
            rank = max(rank, 1)
    return rank


def merge_by_mention(records: list[dict]) -> list[dict]:
    """One test item per (field, string); other concepts become alternatives.

    The benchmark scores a query string, so 'OPC' must be one item whose
    alternatives show that CellLink also uses it for preosteoblasts and outer
    pillar cells. The primary concept has the strongest evidence: a concept the
    string names (ontology label/EXACT synonym), then LRABR, then corpus
    frequency.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        groups[(record["field"], norm_basic(record["mention"]))].append(record)
    concepts_per_key = Counter()
    for (field_name, key), group in groups.items():
        concepts_per_key[(field_name, key)] = len(group)
    merged = []
    for (field_name, key), group in groups.items():
        group.sort(key=lambda r: (-_evidence_rank(r), -r["corpus_count"],
                                  r["intended_concept"]["id"] or "", r["id"]))
        primary = dict(group[0])
        resolved = {r["draft"]["gold_preferred"] for r in group
                    if r["draft"]["expected_action"] == "resolve"}
        outcomes = {(r["draft"]["expected_action"], r["draft"]["gold_preferred"]) for r in group}
        other_field = "disease" if field_name == "cell_type" else "cell_type"
        primary["ambiguity"] = dict(
            primary["ambiguity"], n_concepts=len(group), db_senses=len(resolved),
            concept_disagreement=len(outcomes) > 1,
            other_field_concepts=concepts_per_key.get((other_field, key), 0))
        primary["alternatives"] = [{
            "concept": r["intended_concept"],
            "expected_action": r["draft"]["expected_action"],
            "gold_preferred": r["draft"]["gold_preferred"],
            "n_narrower_total": r["draft"]["n_narrower_total"],
            "basis": r["draft"]["basis"],
            "sources": sorted({s["source"] for s in r["sources"]}),
            "corpus_count": r["corpus_count"],
        } for r in group[1:]]
        primary["corpus_count"] = sum(r["corpus_count"] for r in group)
        primary["id"] = f"{ID_PREFIX[primary['category']]}-{_digest(f'{field_name}|{key}')}"
        merged.append(primary)
    return merged


def _draft_outcome(record: dict) -> tuple:
    draft = record["draft"]
    return (draft["expected_action"], draft["gold_preferred"],
            tuple(sorted(item["name"] for item in draft["gold_narrower"])))


def mark_loose_variants(records: list[dict]) -> None:
    """Link written context-free items that differ only by norm_loose.

    'germ cell' (synonym.jsonl) and 'germ cells' (granularity.jsonl) are one
    query to a user, so annotators must see both and label them alike, and
    step 5 must put them in the same split. Context items are grouped by
    pair_id instead and are left alone.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        if record["context"] is None:
            groups[(record["field"], record["surface"]["normalized"])].append(record)
    for group in groups.values():
        if len(group) < 2:
            continue
        conflict = len({_draft_outcome(record) for record in group}) > 1
        for record in group:
            record["surface"]["loose_variants"] = sorted(
                (other["mention"] for other in group if other is not record), key=str.casefold)
            record["ambiguity"]["loose_variant_conflict"] = conflict


def apply_caps(records: list[dict], cap: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Keep every resolve/choose draft and every context item; cap other buckets.

    Capped buckets keep lexical traps first, then frequent corpus strings,
    then a seeded draw. The draw hashes each item's id rather than consuming
    a shared random stream, so adding or removing one candidate does not
    reshuffle the rest of the sample (annotations are keyed by id).
    """
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for record in sorted(records, key=lambda r: r["id"]):
        buckets[(record["category"], record["field"], record["draft"]["expected_action"])].append(record)
    kept, stats = [], []
    for key in sorted(buckets):
        items = buckets[key]
        category, field_name, action = key
        if action in KEEP_ALL_ACTIONS or category in ("context_abbreviation", "near_miss") or len(items) <= cap:
            chosen = items
        else:
            chosen = sorted(items, key=lambda r: (not r["lexical_trap"], -r["corpus_count"],
                                                  _digest(f"{seed}:{r['id']}")))[:cap]
        kept.extend(chosen)
        stats.append({"category": category, "field": field_name, "expected_action": action,
                      "generated": len(items), "written": len(chosen)})
    return kept, stats


def write_jsonl(records: list[dict], path: Path) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                    encoding="utf-8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cap", type=int, default=300,
                        help="max review/decline candidates per (category, field, action) bucket")
    parser.add_argument("--contexts-per-sense", type=int, default=3)
    parser.add_argument("--near-miss-per-concept", type=int, default=2)
    parser.add_argument("--near-miss-min-score", type=float, default=0.5)
    parser.add_argument("--near-miss-cap", type=int, default=300)
    args = parser.parse_args(argv)

    vocab_manifest = json.loads((args.vocab / "manifest.json").read_text(encoding="utf-8"))
    cell_table = read_tsv(args.vocab / "cell_types.tsv")
    disease_table = read_tsv(args.vocab / "diseases.tsv")

    fetch = lambda key: sources.fetch(key, args.cache_dir)  # noqa: E731
    cl = Ontology.load(fetch("cell_ontology"), prefix="CL:")
    mondo = Ontology.load(fetch("mondo"), prefix="MONDO:")
    expected = vocab_manifest["inputs"]["ontologies"]
    if (cl.data_version, mondo.data_version) != (expected["cell_ontology"], expected["mondo"]):
        raise SystemExit("Ontology versions differ from the vocabulary manifest; rebuild step 1 first.")
    lrabr = load_lrabr(fetch("lrabr"))
    celllink = load_celllink(fetch("celllink"))
    medmentions = load_medmentions(fetch("medmentions"), {
        "train": fetch("medmentions_pmids_trng"), "dev": fetch("medmentions_pmids_dev"),
        "test": fetch("medmentions_pmids_test")})
    disease_mentions = [m for m in medmentions if is_disease_mention(m)]

    umls_to_mondo = mondo.xref_index("UMLS:", require_source="MONDO:equivalentTo")
    cells = FieldIndex("cell_type", cell_table, cl, "cl_ids")
    diseases = FieldIndex("disease", disease_table, mondo, "mondo_ids", umls_to_nodes=umls_to_mondo)
    indexes = {"cell_type": cells, "disease": diseases}

    def disease_label(cui):
        nodes = sorted(umls_to_mondo.get(cui, ()))
        return mondo.label(nodes[0]) if nodes else None

    def disease_key(cui):
        nodes = sorted(umls_to_mondo.get(cui, ()))
        return (nodes[0], "MONDO") if nodes else (cui, "UMLS")

    pool = Pool()
    add_ontology_forms(pool, cells, "cell_ontology", "CL")
    add_ontology_forms(pool, diseases, "mondo", "MONDO")
    add_lrabr(pool, cells, lrabr, "CL")
    add_lrabr(pool, diseases, lrabr, "MONDO")
    add_corpus(pool, cells, celllink, seed=args.seed, relate=lambda c: cells.relate([c]),
               label_of=cl.label, canonical=lambda c: (c, "CL"))
    add_corpus(pool, diseases, disease_mentions, seed=args.seed,
               relate=lambda cui: diseases.relate(umls=cui), label_of=disease_label,
               canonical=disease_key)

    context, context_summary = context_candidates(
        indexes, celllink, medmentions, lrabr, per_sense=args.contexts_per_sense, seed=args.seed,
        label_of=lambda f, c: cl.label(c) if f == "cell_type" else disease_label(c))
    near_miss = (near_miss_candidates(cells, "CL", per_concept=args.near_miss_per_concept,
                                      min_score=args.near_miss_min_score, cap=args.near_miss_cap)
                 + near_miss_candidates(diseases, "MONDO", per_concept=args.near_miss_per_concept,
                                        min_score=args.near_miss_min_score, cap=args.near_miss_cap))
    pooled_strings = {(field_name, key) for field_name, key, _ in pool.items}
    near_miss = [c for c in near_miss if (c.field, norm_basic(c.mention)) not in pooled_strings]

    ontology_surfaces = {"cell_type": surface_concepts(cl), "disease": surface_concepts(mondo)}
    corpus_surfaces = {"cell_type": defaultdict(set), "disease": defaultdict(set)}
    for mention in celllink:
        corpus_surfaces["cell_type"][norm_basic(mention.text)].add(mention.concept)
    for mention in disease_mentions:
        corpus_surfaces["disease"][norm_basic(mention.text)].add(mention.concept)

    def final(candidates):
        return [finalize(c, indexes[c.field], lrabr, ontology_surfaces[c.field], corpus_surfaces[c.field])
                for c in candidates]

    records = merge_by_mention(final(pool.items.values())) + final(context) + final(near_miss)
    duplicates = [i for i, n in Counter(r["id"] for r in records).items() if n > 1]
    if duplicates:
        raise SystemExit(f"Duplicate candidate ids: {duplicates[:5]}")
    kept, bucket_stats = apply_caps(records, args.cap, args.seed)
    mark_loose_variants(kept)

    abbreviations = {r["surface"]["abbreviation_core"] for r in kept
                     if r["category"] in ("abbreviation", "context_abbreviation")
                     and r["surface"]["abbreviation_core"]}
    inventory = sense_inventory(abbreviations | set(context_summary), lrabr, indexes,
                                {"cell_type": loose_concept_index(cl), "disease": loose_concept_index(mondo)},
                                context_summary)

    args.out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for category in CATEGORIES:
        rows = sorted((r for r in kept if r["category"] == category),
                      key=lambda r: (r["field"], r["pair_id"] or "", r["mention"].casefold(),
                                     r["intended_concept"]["id"] or "", r["id"]))
        write_jsonl(rows, args.out / f"{category}.jsonl")
        counts[category] = len(rows)
    write_jsonl(inventory, args.out / "sense_inventory.jsonl")
    write_tsv(pd.DataFrame(bucket_stats), args.out / "summary.tsv")

    manifest = {
        "step": "2 - benchmark candidates with draft labels (not gold)",
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "generator": "tools/omic_tools/tests/name_extraction/generate_candidates.py",
        "git": git_state(PROJECT_ROOT),
        "parameters": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "vocabulary_manifest_sha256": sources.sha256_file(args.vocab / "manifest.json"),
        "ontologies": {"cell_ontology": cl.data_version, "mondo": mondo.data_version},
        "sources": sources.describe(sorted(sources.SOURCES)),
        "inputs": {
            "celllink_linked_mentions": len(celllink),
            "medmentions_mentions": len(medmentions),
            "medmentions_disease_mentions": len(disease_mentions),
            "medmentions_disease_semtypes": sorted(DISEASE_SEMTYPES),
            "lrabr_abbreviations": len(lrabr.senses),
        },
        "excluded_trivial": {f"{f}/{s}": n for (f, s), n in sorted(pool.trivial.items())},
        "files": {**{f"{c}.jsonl": counts[c] for c in CATEGORIES},
                  "sense_inventory.jsonl": len(inventory)},
        "buckets": bucket_stats,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": manifest["files"], "excluded_trivial": manifest["excluded_trivial"]}, indent=2))


if __name__ == "__main__":
    main()
