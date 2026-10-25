"""Loaders for the corpora and the abbreviation list used as candidate sources.

Each corpus mention is reduced to a ``CorpusMention``: the surface string, the
concept it was annotated with, and the sentence it appeared in.
"""

from __future__ import annotations

import gzip
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from tools.omic_tools.tests.name_extraction.text import norm_loose, sentence_around, sentence_boundaries

CELLLINK_SPLITS = {"train": "splits/NLM-CellLink-corpus-train.xml",
                   "val": "splits/NLM-CellLink-corpus-val.xml"}
_CELLLINK_ID_RE = re.compile(r"\(skos:(exact|related)\)\s*(CL:\d+)")

# UMLS semantic types treated as disease-like in MedMentions (full corpus codes).
DISEASE_SEMTYPES = frozenset({
    "T019",  # Congenital Abnormality
    "T020",  # Acquired Abnormality
    "T037",  # Injury or Poisoning
    "T046",  # Pathologic Function
    "T047",  # Disease or Syndrome
    "T048",  # Mental or Behavioral Dysfunction
    "T049",  # Cell or Molecular Dysfunction
    "T050",  # Experimental Model of Disease
    "T190",  # Anatomical Abnormality
    "T191",  # Neoplastic Process
})


@dataclass(frozen=True)
class CorpusMention:
    corpus: str
    doc: str
    split: str
    text: str
    concept: str          # CL:... for CellLink, a UMLS CUI for MedMentions
    link: str             # "exact" / "related" (CellLink) or "annotated" (MedMentions)
    entity_type: str      # CellLink annotation type or MedMentions semantic types
    context: str


# ----------------------------------------------------------------------- LRABR
@dataclass
class Lrabr:
    """SPECIALIST abbreviation table: abbreviation -> sense -> expansion variants.

    A sense is the expansion's lexical record (EUI); spelling variants of one
    expansion share it ("Alzheimer disease", "Alzheimer's disease").
    """
    senses: dict[str, dict[str, list[str]]]
    by_expansion: dict[str, set[tuple[str, str]]]
    abbreviations: frozenset = frozenset()

    def __post_init__(self):
        self.abbreviations = frozenset(self.senses)


def load_lrabr(path) -> Lrabr:
    senses: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    by_expansion: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split("|")
        if len(parts) < 5 or not parts[1] or not parts[4]:
            continue
        abbreviation, expansion = parts[1], parts[4]
        sense = parts[3] or "text:" + norm_loose(expansion)
        if expansion not in senses[abbreviation][sense]:
            senses[abbreviation][sense].append(expansion)
        by_expansion[norm_loose(expansion)].add((abbreviation, sense))
    return Lrabr({abbr: dict(groups) for abbr, groups in senses.items()}, dict(by_expansion))


# -------------------------------------------------------------------- CellLink
def _celllink_split(root: ET.Element, split: str) -> list[CorpusMention]:
    mentions = []
    for document in root.iter("document"):
        doc_id = document.findtext("id") or ""
        for passage in document.iter("passage"):
            passage_offset = int(passage.findtext("offset") or 0)
            text = passage.findtext("text") or ""
            boundaries = sentence_boundaries(text)
            for annotation in passage.iter("annotation"):
                infons = {infon.get("key"): (infon.text or "") for infon in annotation.findall("infon")}
                links = _CELLLINK_ID_RE.findall(infons.get("identifier", ""))
                location = annotation.find("location")
                surface = annotation.findtext("text") or ""
                if not links or location is None or not surface.strip():
                    continue
                start = int(location.get("offset")) - passage_offset
                end = start + int(location.get("length"))
                context = (sentence_around(text, start, end, boundaries=boundaries)
                           if 0 <= start < len(text) else "")
                for link, concept in links:
                    mentions.append(CorpusMention("celllink", doc_id, split, surface.strip(), concept,
                                                  link, infons.get("type", ""), context))
    return mentions


def load_celllink(path) -> list[CorpusMention]:
    """Linked mentions from the public CellLink train and validation splits.

    The test split is released without annotations and is not used.
    """
    with zipfile.ZipFile(path) as archive:
        mentions = []
        for split, member in CELLLINK_SPLITS.items():
            mentions.extend(_celllink_split(ET.fromstring(archive.read(member)), split))
    return mentions


# ----------------------------------------------------------------- MedMentions
def load_medmentions(corpus_path, split_paths: dict[str, Path]) -> list[CorpusMention]:
    """All MedMentions (full) mentions with their sentence.

    PubTator offsets index the title, one space, then the abstract.
    """
    split_of = {}
    for split, path in split_paths.items():
        for pmid in Path(path).read_text(encoding="utf-8").split():
            split_of[pmid] = split
    with gzip.open(corpus_path, "rt", encoding="utf-8") as handle:
        blocks = handle.read().split("\n\n")
    mentions = []
    for block in blocks:
        title = abstract = ""
        rows = []
        for line in block.splitlines():
            if "|t|" in line[:20]:
                title = line.split("|t|", 1)[1]
            elif "|a|" in line[:20]:
                abstract = line.split("|a|", 1)[1]
            elif line.count("\t") == 5:
                rows.append(line.split("\t"))
        document = f"{title} {abstract}"
        boundaries = sentence_boundaries(document)
        for pmid, start, end, surface, semtypes, cui in rows:
            start, end = int(start), int(end)
            mentions.append(CorpusMention(
                "medmentions", f"PMID:{pmid}", split_of.get(pmid, "unknown"), surface.strip(),
                cui.replace("UMLS:", "").strip(), "annotated", semtypes,
                sentence_around(document, start, end, boundaries=boundaries)))
    return mentions


def is_disease_mention(mention: CorpusMention) -> bool:
    return bool(set(mention.entity_type.split(",")) & DISEASE_SEMTYPES)
