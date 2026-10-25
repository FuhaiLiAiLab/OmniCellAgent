"""Relate a candidate's concept to the frozen database vocabulary and draft its label.

A draft is a machine suggestion for annotators, never a gold label. It comes
from where the concept sits in the ontology relative to database terms:

* equal     - a database term carries the same ontology id -> ``resolve``
* narrower  - database terms are is_a descendants of the concept
              (2..MAX_CHOOSE_OPTIONS and no equal term -> ``choose``; exactly
              one, or so many that the concept is a grouping class -> ``review``)
* broader   - the nearest database ancestor is at most BROADER_REVIEW_DEPTH
              is_a steps up and nothing is equal or narrower -> ``review``
              ("colon carcinoma" -> colorectal neoplasm, 3 steps)
* none      - the concept is in the ontology, nothing is equal or narrower,
              and any database ancestor is further up -> ``decline``; a
              concept that cannot be placed in the ontology -> ``review``

Whether a broader database term is an acceptable answer is a policy decision
for the annotators; drafts always list the nearest broader terms with depth.
"""

from __future__ import annotations

import difflib
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from tools.omic_tools.tests.name_extraction.obo import Ontology
from tools.omic_tools.tests.name_extraction.text import (
    abbreviation_core, has_marker_token, norm_basic, norm_loose)

LIST_SEP = " | "
BROADER_REVIEW_DEPTH = 3
MAX_CHOOSE_OPTIONS = 15
LISTED_NARROWER = 10
LISTED_BROADER = 5
NEIGHBOUR_CUTOFF = 0.5
TRAP_SCORE = 0.6


def split_list(value) -> list[str]:
    return [part.strip() for part in str(value or "").split(LIST_SEP) if part.strip()]


@dataclass
class Relation:
    nodes: list[str] = field(default_factory=list)
    equal: set[str] = field(default_factory=set)
    narrower: dict[str, int] = field(default_factory=dict)
    broader: dict[str, int] = field(default_factory=dict)


class FieldIndex:
    """One query field's database vocabulary joined to its ontology."""

    def __init__(self, field_name: str, table: pd.DataFrame, ontology: Ontology, id_column: str,
                 umls_to_nodes: dict[str, set[str]] | None = None):
        self.field = field_name
        self.ontology = ontology
        if "is_control_like" in table.columns:
            table = table.loc[table["is_control_like"].astype(str) != "True"]
        self.names = list(table["db_name"])
        self.n_rows = {name: int(n) for name, n in zip(table["db_name"], table["n_rows"])}
        self.node_to_names: dict[str, set[str]] = defaultdict(set)
        for name, ids in zip(table["db_name"], table[id_column]):
            for node in split_list(ids):
                self.node_to_names[node].add(name)
        self.umls_to_names: dict[str, set[str]] = defaultdict(set)
        if "umls_ids" in table.columns:
            for name, ids in zip(table["db_name"], table["umls_ids"]):
                for cui in split_list(ids):
                    self.umls_to_names[cui].add(name)
        self.umls_to_nodes = umls_to_nodes or {}

        self.basic_names = {norm_basic(name): name for name in self.names}
        self.loose_forms: dict[str, set[str]] = defaultdict(set)
        for name in self.names:
            self.loose_forms[norm_loose(name)].add(name)
            for component in name.split(";"):
                for form in (component, component.split(":", 1)[0]):
                    if form.strip():
                        self.loose_forms[norm_loose(form)].add(name)

        # For every ontology node: database nodes below it, with the shortest distance.
        self.db_descendants: dict[str, dict[str, int]] = defaultdict(dict)
        for node in self.node_to_names:
            for ancestor, depth in ontology.ancestors(node).items():
                known = self.db_descendants[ancestor].get(node)
                if known is None or depth < known:
                    self.db_descendants[ancestor][node] = depth
        self._neighbour_cache: dict[str, list[dict]] = {}

    # ---------------------------------------------------------------- surfaces
    def is_trivial(self, mention: str) -> bool:
        """The current exact matcher already finds it (case-insensitive equality)."""
        return norm_basic(mention) in self.basic_names

    def variant_of(self, mention: str) -> set[str]:
        return set(self.loose_forms.get(norm_loose(mention), ()))

    def neighbours(self, mention: str, n: int = 3) -> list[dict]:
        key = norm_basic(mention)
        if key not in self._neighbour_cache:
            matches = difflib.get_close_matches(key, list(self.basic_names), n=n, cutoff=NEIGHBOUR_CUTOFF)
            self._neighbour_cache[key] = [
                {"name": self.basic_names[match],
                 "score": round(difflib.SequenceMatcher(None, key, match).ratio(), 3)}
                for match in matches]
        return self._neighbour_cache[key]

    # ---------------------------------------------------------------- relation
    def relate(self, nodes=(), umls: str | None = None, db_names=()) -> Relation:
        live = []
        for node in nodes:
            current = self.ontology.current_id(node)
            if current and current not in live:
                live.append(current)
        if umls:
            for node in sorted(self.umls_to_nodes.get(umls, ())):
                if node not in live:
                    live.append(node)
        relation = Relation(nodes=live, equal=set(db_names))
        if umls:
            relation.equal |= self.umls_to_names.get(umls, set())
        for node in live:
            relation.equal |= self.node_to_names.get(node, set())
        for node in live:
            for db_node, depth in self.db_descendants.get(node, {}).items():
                for name in self.node_to_names[db_node]:
                    if name not in relation.equal:
                        relation.narrower[name] = min(depth, relation.narrower.get(name, depth))
            for ancestor, depth in self.ontology.ancestors(node).items():
                for name in self.node_to_names.get(ancestor, ()):
                    if name not in relation.equal and name not in relation.narrower:
                        relation.broader[name] = min(depth, relation.broader.get(name, depth))
        return relation

    def _rank(self, items: dict[str, int]) -> list[tuple[str, int]]:
        return sorted(items.items(), key=lambda kv: (kv[1], -self.n_rows.get(kv[0], 0), kv[0]))

    def draft(self, relation: Relation, review_reason: str | None = None) -> dict:
        equal = sorted(relation.equal, key=lambda name: (-self.n_rows.get(name, 0), name))
        narrower, broader = self._rank(relation.narrower), self._rank(relation.broader)
        nearest = broader[0][1] if broader else None
        if equal:
            action, basis = "resolve", "a database term carries the concept's id"
        elif len(narrower) > MAX_CHOOSE_OPTIONS:
            action, basis = "review", f"grouping concept: {len(narrower)} narrower database terms"
        elif len(narrower) >= 2:
            action, basis = "choose", f"no equal term; {len(narrower)} narrower database terms"
        elif narrower:
            action, basis = "review", "no equal term; exactly one narrower database term"
        elif nearest is not None and nearest <= BROADER_REVIEW_DEPTH:
            action, basis = "review", f"only broader database terms; nearest is {nearest} is_a steps up"
        elif relation.nodes:
            action, basis = "decline", "concept is in the ontology; no database term is equal or narrower"
            if nearest is not None:
                basis += f"; nearest broader database term is {nearest} is_a steps up"
        else:
            action, basis = "review", "concept could not be placed in the ontology"
        if review_reason and action != "review":
            basis = f"{review_reason}; would otherwise be {action}: {basis}"
            action = "review"
        return {
            "expected_action": action,
            "gold_preferred": equal[0] if equal else None,
            "gold_equal": equal,
            "gold_narrower": [{"name": name, "depth": depth} for name, depth in narrower[:LISTED_NARROWER]],
            "n_narrower_total": len(narrower),
            "gold_broader": [{"name": name, "depth": depth} for name, depth in broader[:LISTED_BROADER]],
            "nearest_broader_depth": nearest,
            "basis": basis,
        }

    # ---------------------------------------------------------------- category
    def categorize(self, mention: str, draft: dict, known_abbreviations) -> tuple[str, dict]:
        core = abbreviation_core(mention, known_abbreviations)
        variant = sorted(self.variant_of(mention))
        action = draft["expected_action"]
        if core:
            category = "abbreviation"
        elif variant:
            category = "surface_variant"
        elif action == "choose":
            category = "granularity"
        elif action == "decline":
            category = "absent_concept"
        else:
            category = "synonym"
        accepted = set(draft["gold_equal"]) | {item["name"] for item in draft["gold_narrower"]}
        surface = {
            "normalized": norm_loose(mention),
            "abbreviation_core": core,
            "variant_of": variant,
            "has_marker_token": has_marker_token(mention),
            "draft_conflict": bool(variant) and not set(variant) & accepted,
        }
        return category, surface
