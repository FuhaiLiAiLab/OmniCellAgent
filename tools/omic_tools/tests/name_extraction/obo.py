"""Minimal OBO 1.2 reader for the Cell Ontology and MONDO.

Only [Term] stanzas are read, and only the fields the benchmark needs:
name, synonyms (with scope and type), xrefs (with qualifier sources), is_a,
alt_id, obsolescence, and replacement. Hierarchy walks use is_a edges only.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

from tools.omic_tools.tests.name_extraction.text import norm_basic

_SYNONYM_RE = re.compile(r'^"((?:[^"\\]|\\.)*)"\s+(EXACT|BROAD|NARROW|RELATED)\b\s*([^\[\s]*)')
_QUALIFIER_SOURCE_RE = re.compile(r'source="([^"]*)"')
_ESCAPE_RE = re.compile(r"\\(.)")


@dataclass
class Synonym:
    text: str
    scope: str
    type: str | None = None


@dataclass
class Xref:
    value: str
    sources: tuple[str, ...] = ()


@dataclass
class Term:
    id: str
    name: str | None = None
    synonyms: list[Synonym] = field(default_factory=list)
    xrefs: list[Xref] = field(default_factory=list)
    is_a: list[str] = field(default_factory=list)
    alt_ids: list[str] = field(default_factory=list)
    replaced_by: list[str] = field(default_factory=list)
    is_obsolete: bool = False


def _unescape(text: str) -> str:
    return _ESCAPE_RE.sub(lambda match: match.group(1), text)


def _first_token(value: str) -> str:
    tokens = value.split("!", 1)[0].split("{", 1)[0].split()
    return tokens[0] if tokens else ""


class Ontology:
    def __init__(self, terms: dict[str, Term], data_version: str | None = None):
        self.terms = terms
        self.data_version = data_version
        self._children = None
        self._alt = {alt: term.id for term in terms.values() for alt in term.alt_ids}

    # ------------------------------------------------------------------ parsing
    @classmethod
    def parse(cls, text: str, prefix: str | None = None) -> "Ontology":
        """Parse OBO text; with ``prefix`` (e.g. "CL:"), keep only that
        ontology's own terms and drop imported ones such as HP or UBERON."""
        terms: dict[str, Term] = {}
        data_version = None
        current: Term | None = None
        in_term = False
        for raw in text.splitlines():
            line = raw.strip()
            if line.startswith("["):
                if current is not None:
                    terms[current.id] = current
                current = None
                in_term = line == "[Term]"
                continue
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            if not in_term:
                if key == "data-version" and data_version is None:
                    data_version = value
                continue
            if key == "id":
                current = Term(id=value)
            elif current is None:
                continue
            elif key == "name":
                current.name = _unescape(value)
            elif key == "synonym":
                match = _SYNONYM_RE.match(value)
                if match:
                    current.synonyms.append(Synonym(_unescape(match.group(1)), match.group(2),
                                                    match.group(3) or None))
            elif key == "xref" and _first_token(value):
                sources = tuple(_QUALIFIER_SOURCE_RE.findall(value.split("{", 1)[1])) if "{" in value else ()
                current.xrefs.append(Xref(_first_token(value), sources))
            elif key in ("is_a", "alt_id", "replaced_by") and _first_token(value):
                {"is_a": current.is_a, "alt_id": current.alt_ids,
                 "replaced_by": current.replaced_by}[key].append(_first_token(value))
            elif key == "is_obsolete":
                current.is_obsolete = value.lower() == "true"
        if current is not None:
            terms[current.id] = current
        if prefix:
            terms = {term_id: term for term_id, term in terms.items() if term_id.startswith(prefix)}
        return cls(terms, data_version)

    @classmethod
    def load(cls, path, prefix: str | None = None) -> "Ontology":
        return cls.parse(Path(path).read_text(encoding="utf-8"), prefix=prefix)

    # --------------------------------------------------------------- identity
    def current_id(self, term_id: str) -> str | None:
        """Follow alt_id and replaced_by to a live term; None if there is none."""
        term_id = self._alt.get(term_id, term_id)
        seen = set()
        while term_id in self.terms and self.terms[term_id].is_obsolete:
            if term_id in seen or not self.terms[term_id].replaced_by:
                return None
            seen.add(term_id)
            term_id = self.terms[term_id].replaced_by[0]
        return term_id if term_id in self.terms else None

    def live_terms(self):
        return (term for term in self.terms.values() if not term.is_obsolete)

    def label(self, term_id: str) -> str | None:
        term = self.terms.get(term_id)
        return term.name if term else None

    def label_index(self) -> dict[str, set[str]]:
        index: dict[str, set[str]] = defaultdict(set)
        for term in self.live_terms():
            if term.name:
                index[norm_basic(term.name)].add(term.id)
        return index

    def synonym_index(self, scopes=("EXACT",)) -> dict[str, set[str]]:
        index: dict[str, set[str]] = defaultdict(set)
        for term in self.live_terms():
            for synonym in term.synonyms:
                if synonym.scope in scopes:
                    index[norm_basic(synonym.text)].add(term.id)
        return index

    def xref_index(self, prefix: str, require_source: str | None = None) -> dict[str, set[str]]:
        """Map xref values (without prefix) to live term ids."""
        index: dict[str, set[str]] = defaultdict(set)
        for term in self.live_terms():
            for xref in term.xrefs:
                if not xref.value.startswith(prefix):
                    continue
                if require_source and require_source not in xref.sources:
                    continue
                index[xref.value[len(prefix):]].add(term.id)
        return index

    # --------------------------------------------------------------- hierarchy
    def parents(self, term_id: str) -> list[str]:
        term = self.terms.get(term_id)
        return [parent for parent in term.is_a if parent in self.terms] if term else []

    def children(self, term_id: str) -> list[str]:
        if self._children is None:
            children = defaultdict(list)
            for term in self.live_terms():
                for parent in self.parents(term.id):
                    children[parent].append(term.id)
            self._children = children
        return self._children.get(term_id, [])

    def ancestors(self, term_id: str, max_depth: int | None = None) -> dict[str, int]:
        """Shortest is_a distance to every ancestor (the term itself excluded)."""
        return self._walk(term_id, self.parents, max_depth)

    def descendants(self, term_id: str, max_depth: int | None = None) -> dict[str, int]:
        return self._walk(term_id, self.children, max_depth)

    @staticmethod
    def _walk(start: str, step, max_depth: int | None) -> dict[str, int]:
        depths: dict[str, int] = {}
        queue = deque([(start, 0)])
        while queue:
            node, depth = queue.popleft()
            if max_depth is not None and depth >= max_depth:
                continue
            for neighbour in step(node):
                if neighbour != start and neighbour not in depths:
                    depths[neighbour] = depth + 1
                    queue.append((neighbour, depth + 1))
        return depths
