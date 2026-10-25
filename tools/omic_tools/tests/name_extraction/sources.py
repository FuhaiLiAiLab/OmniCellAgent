"""Pinned external sources for the name-extraction benchmark.

Every source is fetched from a fixed release URL and checked against a
pinned SHA-256 before use, so a rerun either reproduces the same inputs or
fails loudly. Downloads live in a cache outside the repository.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CACHE_ENV = "OMNICELL_NAME_BENCH_CACHE"

_MEDMENTIONS = ("https://github.com/chanzuckerberg/MedMentions/raw/"
                "14578232bb6cdf290fea9dd7fee60a3e01ee6fd5/full/data/")


@dataclass(frozen=True)
class Source:
    key: str
    url: str
    sha256: str
    filename: str
    version: str
    license: str
    citation: str


SOURCES = {
    source.key: source for source in (
        Source(
            key="cell_ontology",
            url="https://zenodo.org/api/records/18090009/files/cl_v2025-01-08.obo/content",
            sha256="2350e0c6584f43644c56ffe3d204d715f63d06ceb3f571ba8741fdb487424d4d",
            filename="cl_v2025-01-08.obo",
            version="Cell Ontology data-version releases/2025-01-07, the copy shipped with CellLink",
            license="CC BY 4.0",
            citation="Diehl AD et al. The Cell Ontology 2016. J Biomed Semantics 7:44.",
        ),
        Source(
            key="mondo",
            url="https://github.com/monarch-initiative/mondo/releases/download/v2026-09-01/mondo.obo",
            sha256="50c8367f9bd9978321eedf84d4e2e79cb757efbed514c81effb00b9ad2e49994",
            filename="mondo_v2026-09-01.obo",
            version="MONDO v2026-09-01",
            license="CC BY 4.0",
            citation="Vasilevsky NA et al. Mondo: Unifying diseases for the world. medRxiv 2022.",
        ),
        Source(
            key="celllink",
            url="https://zenodo.org/api/records/18090009/files/data.zip/content",
            sha256="6c585fcea88b4d9624fcdf46bbb4948bc24307847d3ef6d12036b460ca65ab0c",
            filename="celllink_data.zip",
            version="NLM CellLink corpus, Zenodo record 18090009 (2025-12-29)",
            license="Public domain (United States Government Work)",
            citation="Rotenberg NH, Leaman R, Islamaj R et al. Cell phenotypes in the biomedical "
                     "literature: a systematic analysis and text mining corpus. bioRxiv 2026. "
                     "doi:10.5281/zenodo.18090009",
        ),
        Source(
            key="medmentions",
            url=_MEDMENTIONS + "corpus_pubtator.txt.gz",
            sha256="95bebda2776e4cd8f59456af3d977eb7c640fa7997db64c112a18fb09158b1ce",
            filename="medmentions_full_corpus_pubtator.txt.gz",
            version="MedMentions full, commit 14578232 (UMLS 2017AA)",
            license="CC0 1.0",
            citation="Mohan S, Li D. MedMentions: A Large Biomedical Corpus Annotated with UMLS "
                     "Concepts. AKBC 2019.",
        ),
        Source(
            key="medmentions_pmids_trng",
            url=_MEDMENTIONS + "corpus_pubtator_pmids_trng.txt",
            sha256="ab037fc5606588b4294fa1ead987a66652fe8f3c81e7832e6a02d0c1cdb1c7e6",
            filename="medmentions_pmids_trng.txt",
            version="MedMentions full, commit 14578232",
            license="CC0 1.0",
            citation="Mohan S, Li D. AKBC 2019.",
        ),
        Source(
            key="medmentions_pmids_dev",
            url=_MEDMENTIONS + "corpus_pubtator_pmids_dev.txt",
            sha256="d03508c74df795000599509b997b7280c0ff5bc53b064547e98b29be8b616abd",
            filename="medmentions_pmids_dev.txt",
            version="MedMentions full, commit 14578232",
            license="CC0 1.0",
            citation="Mohan S, Li D. AKBC 2019.",
        ),
        Source(
            key="medmentions_pmids_test",
            url=_MEDMENTIONS + "corpus_pubtator_pmids_test.txt",
            sha256="fece5bcb8c61edf8de73884980b80daf274132ef6d69f0d195c0114acbfdd4ab",
            filename="medmentions_pmids_test.txt",
            version="MedMentions full, commit 14578232",
            license="CC0 1.0",
            citation="Mohan S, Li D. AKBC 2019.",
        ),
        Source(
            key="lrabr",
            url="https://data.lhncbc.nlm.nih.gov/public/lsg/lexicon/2025/release/LEX_DOC/LRABR",
            sha256="39f0119727555d9f9dc1d8cfef7c3496360ef718d96f756f5deddb32fb867027",
            filename="LRABR_2025",
            version="SPECIALIST Lexicon 2025 release, LRABR table",
            license="NLM SPECIALIST Lexicon terms; review before redistributing the raw table",
            citation="National Library of Medicine. SPECIALIST Lexicon, 2025 release.",
        ),
    )
}


def default_cache_dir() -> Path:
    """Cache outside the repository; override with OMNICELL_NAME_BENCH_CACHE."""
    configured = os.environ.get(CACHE_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "omnicellagent" / "name_extraction"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, retries: int = 3, timeout: int = 600) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, retries + 1):
        handle, temp_name = tempfile.mkstemp(dir=destination.parent, prefix=".partial-")
        try:
            with os.fdopen(handle, "wb") as output, urllib.request.urlopen(url, timeout=timeout) as response:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    output.write(chunk)
            os.replace(temp_name, destination)
            return
        except Exception as error:  # network errors are retried, then re-raised
            last_error = error
            if os.path.exists(temp_name):
                os.remove(temp_name)
            print(f"[sources] download attempt {attempt}/{retries} failed for {url}: {error}",
                  file=sys.stderr)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not download {url}") from last_error


def fetch(key: str, cache_dir: Path | None = None) -> Path:
    """Return the cached path of a pinned source, downloading it if needed."""
    source = SOURCES[key]
    cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
    path = cache_dir / source.filename
    if not path.exists():
        print(f"[sources] downloading {key} from {source.url}")
        _download(source.url, path)
    actual = sha256_file(path)
    if actual != source.sha256:
        raise ValueError(
            f"Checksum mismatch for {key} at {path}: expected {source.sha256}, got {actual}. "
            "Delete the cached file to re-download, or update the pin deliberately."
        )
    return path


def describe(keys) -> dict:
    """Provenance block for manifests."""
    return {
        key: {
            "url": SOURCES[key].url,
            "sha256": SOURCES[key].sha256,
            "version": SOURCES[key].version,
            "license": SOURCES[key].license,
            "citation": SOURCES[key].citation,
        }
        for key in keys
    }
