"""Surface normalisation, abbreviation detection, and sentence extraction.

Two normalisation levels are used:

* ``norm_basic`` only unifies Unicode, case, and whitespace. Two strings that
  agree here differ by nothing a case-insensitive exact matcher would notice.
* ``norm_loose`` also drops possessives and punctuation, maps Greek letters,
  marker signs (``+`` to "positive", a token-final ``-`` to "negative"), and
  small Roman numerals, and strips a plural ``s`` from tokens of five or more
  characters. It is applied to both sides of a comparison, so the crude
  stemming only has to be consistent, not correct. Signs are spelled out
  because dropping them would equate opposite phenotypes ("CD16+"/"CD16-").
"""

from __future__ import annotations

import re
import unicodedata

_APOSTROPHES = str.maketrans({char: "'" for char in "’‘´`׳ʼ′"})
_DASHES = str.maketrans({char: "-" for char in "‐‑‒–—―−"})
_GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
          "κ": "kappa", "λ": "lambda"}
_ROMAN = {"ii": "2", "iii": "3", "iv": "4"}
_ROMAN_I_CONTEXT = {"type", "class", "grade", "stage", "phase"}
_KEEP_FINAL_S = ("ss", "us", "is", "os", "as")

# Words that may follow an abbreviation without changing what it names:
# "NK cells" is the abbreviation NK, "COPD disease" is COPD.
GENERIC_HEADS = frozenset({"cell", "cells", "disease", "diseases", "syndrome", "syndromes"})

_MARKER_RE = re.compile(r"(?:\bCD\d+[A-Za-z]*\b|[A-Za-z0-9]+\+)")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|\n+")


def _unify(text) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    return text.translate(_APOSTROPHES).translate(_DASHES)


def norm_basic(text) -> str:
    return " ".join(_unify(text).casefold().split())


def norm_loose(text) -> str:
    text = norm_basic(text)
    for letter, name in _GREEK.items():
        text = text.replace(letter, f" {name} ")
    text = re.sub(r"'s\b", "", text)
    text = text.replace("'", "")
    text = re.sub(r"\s*\+", " positive ", text)
    # A minus that ends a token is a marker sign ('CD16-', 'CD4-CD8- T cell');
    # a hyphen between two word characters ('B-cell', 'CD4-positive') is not,
    # nor is one split from its word ('CD14- positive monocytes').
    text = re.sub(r"(?<=\w)-(?!\w)(?!\s*(?:positive|negative)\b)", " negative ", text)
    text = re.sub(r"[\W_]+", " ", text)
    tokens: list[str] = []
    for token in text.split():
        if token in _ROMAN:
            token = _ROMAN[token]
        elif token == "i" and tokens and tokens[-1] in _ROMAN_I_CONTEXT:
            token = "1"
        elif len(token) >= 5 and token.endswith("s") and not token.endswith(_KEEP_FINAL_S):
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens)


def _abbreviation_shaped(token: str, known) -> bool:
    if not 2 <= len(token) <= 10 or token.isdigit() or not any(c.isalpha() for c in token):
        return False
    if any(c.isspace() for c in token):
        return False
    uppers = sum(c.isupper() for c in token)
    has_digit = any(c.isdigit() for c in token)
    if uppers and token in known:
        return True
    # All-capital words longer than five letters ("ASTHMA") are usually shouted
    # words, not abbreviations; LRABR membership above still admits "CADASIL".
    letters_only_upper = token.isalpha() and token.isupper()
    if letters_only_upper and len(token) > 5:
        return False
    return uppers >= 2 or (has_digit and uppers >= 1) or any(c.isupper() for c in token[1:])


def abbreviation_core(mention, known_abbreviations=frozenset()) -> str | None:
    """Return the abbreviation a mention consists of, or None.

    'NK cells' -> 'NK', 'Tregs' -> 'Treg', 'ILC3s' -> 'ILC3', 'OPCs' -> 'OPC'.
    A lower-case plural s is stripped; an upper-case S is part of the
    abbreviation ('ALS', 'MS'). ``known_abbreviations`` (case-sensitive, e.g.
    from LRABR) admits mixed-case forms such as 'Treg' that the shape rule
    alone would miss; it never admits an all-lower-case word.
    """
    tokens = _unify(mention).split()
    while len(tokens) > 1 and tokens[-1].casefold() in GENERIC_HEADS:
        tokens.pop()
    if len(tokens) != 1:
        return None
    token = tokens[0].strip(".,;:()[]")
    token = token.rstrip("+")
    if token.endswith("s") and len(token) > 2:
        singular = token[:-1]
        if _abbreviation_shaped(singular, known_abbreviations):
            return singular
    return token if _abbreviation_shaped(token, known_abbreviations) else None


def has_marker_token(mention) -> bool:
    """Marker-style naming such as 'CD8+ T cells' or 'FOXP3+ cells'."""
    return bool(_MARKER_RE.search(_unify(mention)))


def sentence_boundaries(text: str) -> list[tuple[int, int]]:
    """(start, end) of every sentence separator; compute once per document."""
    return [(match.start(), match.end()) for match in _SENTENCE_END_RE.finditer(text)]


def sentence_around(text: str, start: int, end: int, max_chars: int = 300,
                    boundaries: list[tuple[int, int]] | None = None) -> str:
    """The sentence containing text[start:end], trimmed to max_chars around the span."""
    if boundaries is None:
        boundaries = sentence_boundaries(text)
    left, right = 0, len(text)
    for separator_start, separator_end in boundaries:
        if separator_end <= start:
            left = separator_end
        elif separator_start >= end:
            right = separator_start
            break
    sentence = text[left:right].strip()
    if len(sentence) <= max_chars:
        return sentence
    offset = text.find(sentence, left)
    centre = (start + end) // 2 - offset
    lo = max(0, centre - max_chars // 2)
    hi = min(len(sentence), lo + max_chars)
    lo = max(0, hi - max_chars)
    return ("…" if lo > 0 else "") + sentence[lo:hi].strip() + ("…" if hi < len(sentence) else "")
