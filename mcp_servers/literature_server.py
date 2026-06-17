"""Literature Search MCP Server - Port 9012.

This server focuses on literature retrieval payloads that are useful for
agent-to-agent handoff: every tool returns an LLM-readable message plus
structured metadata containing the raw abstracts, full paper text, or web page
text needed for later citation and offline review.
"""
import asyncio
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
from fastmcp import FastMCP

from tools.pubmed_tools.get_elsiver_paper import (
    format_sections_for_display,
    get_article_sections_from_xml,
)
from tools.pubmed_tools.get_pmc_paper import parse_xml_to_sections

try:
    from utils.path_config import get_path
except Exception:  # pragma: no cover - server can still use local MCP cache.
    get_path = None

try:
    from langchain_community.utilities.pubmed import PubMedAPIWrapper
except ImportError:  # pragma: no cover - optional but listed in requirements.
    PubMedAPIWrapper = None

try:
    import pymupdf
except ImportError:  # pragma: no cover - dependency is in project requirements.
    pymupdf = None


load_dotenv(os.path.join(project_root, ".env"))
mcp = FastMCP("Literature Search Services")

ELSIVER_ID = "http://www.elsevier.com/xml/"
PMC_ID = """key="article-id_pmc"""


def has_error_content(text: str) -> bool:
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")

    error_indicators = [
        "[Error] : No result can be found",
        "No result can be found",
        "<B> - https://www.ncbi.nlm.nih.gov/",
        "Functions: Extracting pubmed abstracts",
    ]
    text_lower = (text or "").lower()
    return any(indicator.lower() in text_lower for indicator in error_indicators)


def _new_session_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def _session_dir(session_id: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "literature_outputs", session_id)
    os.makedirs(path, exist_ok=True)
    return path


def _configured_pubmed_cache_dirs() -> Dict[str, str]:
    if get_path is not None:
        try:
            return {
                "jsonl": get_path("cache.pubmed_jsonl", absolute=True, create=True),
                "doi": get_path("cache.pubmed_doi", absolute=True, create=True),
            }
        except Exception:
            pass

    base_dir = os.path.join(project_root, "cache", "PudMed_DB")
    return {
        "jsonl": os.path.join(base_dir, "jsonl_cache"),
        "doi": os.path.join(base_dir, "doi_cache"),
    }


def _literature_processor() -> Any:
    try:
        from tools.pubmed_tools.query_pubmed_tool import MedicalResearchProcessor
    except ImportError as exc:
        raise ImportError(
            "Full-paper PubMed retrieval requires paperscraper and the existing "
            "tools.pubmed_tools.query_pubmed_tool dependencies."
        ) from exc

    class LiteratureResearchProcessor(MedicalResearchProcessor):
        """MedicalResearchProcessor with read-through DOI cache lookup."""

        def __init__(
            self,
            jsonl_cache_dir: str,
            doi_cache_dir: str,
            additional_cache_dirs: Optional[List[str]] = None,
        ):
            super().__init__(jsonl_cache_dir=jsonl_cache_dir, doi_cache_dir=doi_cache_dir)
            self.additional_cache_dirs = additional_cache_dirs or [doi_cache_dir]

        def _check_global_cache(self, doi: str) -> Optional[str]:
            cached_path = super()._check_global_cache(doi)
            if cached_path:
                return cached_path
            return _find_cached_file_in_dirs(self.additional_cache_dirs, doi)

    cache_dirs = _configured_pubmed_cache_dirs()
    os.makedirs(cache_dirs["doi"], exist_ok=True)
    os.makedirs(cache_dirs["jsonl"], exist_ok=True)
    additional_cache_dirs = _existing_paper_cache_dirs(cache_dirs["doi"])
    return LiteratureResearchProcessor(
        jsonl_cache_dir=cache_dirs["jsonl"],
        doi_cache_dir=cache_dirs["doi"],
        additional_cache_dirs=additional_cache_dirs,
    )


def _existing_paper_cache_dirs(primary_cache_dir: str) -> List[str]:
    candidates = [primary_cache_dir]

    if get_path is not None:
        for key in ("cache.pubmed_doi", "logs.base"):
            try:
                candidates.append(get_path(key, absolute=True, create=False))
            except Exception:
                pass

    candidates.append(os.path.join(project_root, "logs"))

    seen = set()
    existing_dirs = []
    for candidate in candidates:
        if not candidate:
            continue
        normalized = os.path.abspath(candidate)
        if normalized in seen or not os.path.isdir(normalized):
            continue
        seen.add(normalized)
        existing_dirs.append(normalized)
    return existing_dirs


def _find_cached_file_in_dirs(cache_dirs: List[str], doi: str) -> Optional[str]:
    if not doi:
        return None

    base = _doi_to_filename(doi)
    expected_names = {f"{base}.xml", f"{base}.pdf"}

    for cache_dir in cache_dirs:
        for expected_name in expected_names:
            candidate = os.path.join(cache_dir, expected_name)
            if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                return candidate

    for cache_dir in cache_dirs:
        for root, _, files in os.walk(cache_dir):
            matched = expected_names.intersection(files)
            if matched:
                candidate = os.path.join(root, sorted(matched)[0])
                if os.path.getsize(candidate) > 0:
                    return candidate

    return None


def _safe_filename(value: str, default: str = "result") -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return (safe or default)[:120]


def _json_response(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _save_payload(payload: Dict[str, Any], session_id: str, stem: str) -> str:
    output_dir = _session_dir(session_id)
    output_path = os.path.join(output_dir, f"{_safe_filename(stem)}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _paper_citation(paper: Dict[str, Any]) -> str:
    title = paper.get("title") or "Untitled paper"
    year = paper.get("year") or paper.get("date") or "n.d."
    journal = paper.get("journal") or paper.get("venue") or "Unknown journal"
    doi = paper.get("doi") or "no DOI"
    return f"{title} ({year}). {journal}. DOI: {doi}"


def _abstract_record(paper: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "index": index,
        "title": paper.get("title", ""),
        "authors": paper.get("authors", []),
        "journal": paper.get("journal", paper.get("venue", "")),
        "year": paper.get("year", ""),
        "date": paper.get("date", ""),
        "doi": paper.get("doi", ""),
        "pmid": paper.get("pmid", paper.get("paperId", "")),
        "abstract": paper.get("abstract", ""),
        "raw_pubmed_metadata": paper,
    }


def _doi_to_filename(doi: str) -> str:
    return doi.replace("/", "_")


def _find_cached_paper_file(cache_dir: str, doi: str) -> Optional[str]:
    if not doi:
        return None
    base = _doi_to_filename(doi)
    for ext in (".xml", ".pdf"):
        candidate = os.path.join(cache_dir, f"{base}{ext}")
        if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
            return candidate
    return None


def _extract_pdf_text(path: str) -> str:
    if pymupdf is None:
        raise ImportError("pymupdf is required to extract PDF full text")

    with pymupdf.open(path) as doc:
        return "\f".join(page.get_text() for page in doc)


def _extract_xml_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        xml_content = f.read()

    if has_error_content(xml_content):
        return ""

    if ELSIVER_ID in xml_content:
        sections = get_article_sections_from_xml(xml_content)
        return format_sections_for_display(sections) if sections else ""

    if PMC_ID in xml_content:
        return parse_xml_to_sections(path) or ""

    return re.sub(r"\s+", " ", xml_content).strip()


def _extract_full_text(path: str) -> str:
    if path.endswith(".pdf"):
        text = _extract_pdf_text(path)
    elif path.endswith(".xml"):
        text = _extract_xml_text(path)
    else:
        text = ""

    if has_error_content(text):
        return ""
    return text.strip()


def _truncate_for_message(text: str, limit: int = 1200) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _langchain_doc_to_record(doc: Any, index: int) -> Dict[str, Any]:
    metadata = getattr(doc, "metadata", {}) or {}
    page_content = getattr(doc, "page_content", "") or ""
    title = metadata.get("Title") or metadata.get("title") or ""
    if not title:
        title_match = re.search(r"Title:\s*(.+)", page_content)
        title = title_match.group(1).strip() if title_match else ""

    published = (
        metadata.get("Published")
        or metadata.get("pub_date")
        or metadata.get("Published Date")
        or ""
    )
    uid = metadata.get("uid") or metadata.get("PMID") or metadata.get("pmid") or ""

    return {
        "index": index,
        "title": title,
        "pmid": uid,
        "published": published,
        "metadata": metadata,
        "abstract": page_content,
        "raw_document": {
            "page_content": page_content,
            "metadata": metadata,
        },
    }


def _text_content(element: Optional[ET.Element]) -> str:
    if element is None:
        return ""
    return " ".join(text.strip() for text in element.itertext() if text and text.strip())


def _urlopen_with_retries(
    request: urllib.request.Request,
    timeout: int = 30,
    max_attempts: int = 4,
):
    for attempt in range(1, max_attempts + 1):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            retryable = exc.code in {429, 500, 502, 503, 504}
            if not retryable or attempt == max_attempts:
                raise
            time.sleep(min(8, 2 ** attempt))
        except urllib.error.URLError:
            if attempt == max_attempts:
                raise
            time.sleep(min(8, 2 ** attempt))


def _search_pubmed_eutils(query: str, top_k: int) -> List[Dict[str, Any]]:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    headers = {"User-Agent": "OmniCellAgent-literature-mcp/1.0"}
    search_params = urllib.parse.urlencode(
        {
            "db": "pubmed",
            "term": query,
            "retmax": top_k,
            "retmode": "json",
        }
    )
    search_request = urllib.request.Request(
        f"{base}/esearch.fcgi?{search_params}",
        headers=headers,
    )
    with _urlopen_with_retries(search_request, timeout=30) as response:
        search_data = json.loads(response.read().decode("utf-8"))

    pmids = search_data.get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return []

    fetch_params = urllib.parse.urlencode(
        {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
    )
    fetch_request = urllib.request.Request(
        f"{base}/efetch.fcgi?{fetch_params}",
        headers=headers,
    )
    with _urlopen_with_retries(fetch_request, timeout=30) as response:
        xml_text = response.read().decode("utf-8", errors="ignore")

    root = ET.fromstring(xml_text)
    papers: List[Dict[str, Any]] = []
    for index, article in enumerate(root.findall(".//PubmedArticle"), 1):
        medline = article.find("./MedlineCitation")
        article_node = medline.find("./Article") if medline is not None else None
        pubmed_data = article.find("./PubmedData")

        pmid = _text_content(medline.find("./PMID") if medline is not None else None)
        title = _text_content(article_node.find("./ArticleTitle") if article_node is not None else None)
        abstract_parts = [
            _text_content(abstract)
            for abstract in article.findall(".//Abstract/AbstractText")
        ]
        abstract = "\n".join(part for part in abstract_parts if part)
        journal = _text_content(article_node.find("./Journal/Title") if article_node is not None else None)
        year = _text_content(article_node.find("./Journal/JournalIssue/PubDate/Year") if article_node is not None else None)
        if not year and article_node is not None:
            year = _text_content(article_node.find("./Journal/JournalIssue/PubDate/MedlineDate"))

        authors = []
        for author in article.findall(".//AuthorList/Author"):
            last = _text_content(author.find("./LastName"))
            fore = _text_content(author.find("./ForeName"))
            collective = _text_content(author.find("./CollectiveName"))
            name = collective or " ".join(part for part in [fore, last] if part)
            if name:
                authors.append(name)

        doi = ""
        if pubmed_data is not None:
            for article_id in pubmed_data.findall(".//ArticleId"):
                if article_id.attrib.get("IdType") == "doi":
                    doi = _text_content(article_id)
                    break

        papers.append(
            {
                "index": index,
                "title": title,
                "pmid": pmid,
                "published": year,
                "year": year,
                "journal": journal,
                "authors": authors,
                "doi": doi,
                "abstract": abstract,
                "metadata": {
                    "pmid": pmid,
                    "journal": journal,
                    "year": year,
                    "doi": doi,
                },
                "raw_document": {
                    "page_content": abstract,
                    "metadata": {
                        "pmid": pmid,
                        "title": title,
                        "journal": journal,
                        "year": year,
                        "doi": doi,
                    },
                },
            }
        )

    return papers


@mcp.tool()
async def search_pubmed_abstracts(
    query: str,
    top_k: int = 32,
    session_id: Optional[str] = None,
) -> str:
    """
    Search PubMed metadata and abstracts without downloading full papers.

    Returns JSON with:
    - message: compact LLM-readable search summary
    - metadata.papers: raw PubMed records and abstracts
    - metadata.saved_file: persisted JSON payload for later reference
    """
    session_id = session_id or _new_session_id("lit_abs")
    try:
        from paperscraper.pubmed import get_and_dump_pubmed_papers
    except ImportError:
        payload = {
            "message": "paperscraper PubMed abstract search is unavailable because paperscraper is not installed. Use search_pubmed_abstracts_langchain for abstract-only PubMed search in this environment.",
            "metadata": {
                "service": "literature_search",
                "tool": "search_pubmed_abstracts",
                "query": query,
                "session_id": session_id,
                "source": "pubmed_paperscraper",
                "paper_count": 0,
                "papers": [],
                "error": "paperscraper is not installed",
            },
        }
        payload["metadata"]["saved_file"] = _save_payload(
            payload, session_id, f"pubmed_abstracts_{_safe_filename(query)}"
        )
        return _json_response(payload)

    cache_dirs = _configured_pubmed_cache_dirs()
    os.makedirs(cache_dirs["jsonl"], exist_ok=True)
    metadata_path = os.path.join(
        cache_dirs["jsonl"],
        f"{session_id}_abstracts.jsonl",
    )

    get_and_dump_pubmed_papers([query], output_filepath=metadata_path, max_results=top_k)
    raw_papers = _read_jsonl(metadata_path)
    papers = [_abstract_record(paper, i) for i, paper in enumerate(raw_papers, 1)]

    summary_lines = [
        f"PubMed abstract search for '{query}' returned {len(papers)} papers.",
    ]
    for paper in papers[: min(5, len(papers))]:
        abstract_preview = _truncate_for_message(paper.get("abstract", ""), 500)
        summary_lines.append(
            f"{paper['index']}. {_paper_citation(paper)}\n"
            f"   Abstract: {abstract_preview or 'No abstract available.'}"
        )

    payload = {
        "message": "\n".join(summary_lines),
        "metadata": {
            "service": "literature_search",
            "tool": "search_pubmed_abstracts",
            "query": query,
            "session_id": session_id,
            "source": "pubmed",
            "paper_count": len(papers),
            "pubmed_jsonl": metadata_path,
            "papers": papers,
        },
    }
    payload["metadata"]["saved_file"] = _save_payload(
        payload, session_id, f"pubmed_abstracts_{_safe_filename(query)}"
    )
    return _json_response(payload)


@mcp.tool()
async def search_pubmed_abstracts_langchain(
    query: str,
    top_k: int = 32,
    session_id: Optional[str] = None,
    doc_content_chars_max: int = 5000,
) -> str:
    """
    Search PubMed abstracts through LangChain's NCBI PubMed wrapper.

    This is the fast abstract-only fallback used by the LangGraph agent. It
    returns less publisher metadata than paperscraper, but has broad abstract
    availability and does not require DOI/full-text download eligibility.
    """
    session_id = session_id or _new_session_id("lit_lc_abs")

    source = "pubmed_langchain"
    fallback_reason = ""
    try:
        if PubMedAPIWrapper is not None:
            wrapper = PubMedAPIWrapper(
                top_k_results=top_k,
                doc_content_chars_max=doc_content_chars_max,
            )
            docs = await asyncio.to_thread(wrapper.load, query)
            papers = [_langchain_doc_to_record(doc, i) for i, doc in enumerate(docs, 1)]
        else:
            source = "pubmed_eutils"
            fallback_reason = "langchain_community is not installed; used direct NCBI E-utilities fallback"
            papers = await asyncio.to_thread(_search_pubmed_eutils, query, top_k)
    except Exception as exc:
        payload = {
            "message": f"PubMed abstract search for '{query}' failed via {source}: {exc}",
            "metadata": {
                "service": "literature_search",
                "tool": "search_pubmed_abstracts_langchain",
                "query": query,
                "session_id": session_id,
                "source": source,
                "fallback_reason": fallback_reason,
                "paper_count": 0,
                "papers": [],
                "error": str(exc),
            },
        }
        payload["metadata"]["saved_file"] = _save_payload(
            payload, session_id, f"pubmed_langchain_abstracts_{_safe_filename(query)}"
        )
        return _json_response(payload)

    summary_lines = [
        f"PubMed abstract search for '{query}' returned {len(papers)} papers via {source}.",
    ]
    if fallback_reason:
        summary_lines.append(f"Fallback: {fallback_reason}.")
    for paper in papers[: min(5, len(papers))]:
        pmid_suffix = f" PMID: {paper.get('pmid')}" if paper.get("pmid") else ""
        summary_lines.append(
            f"{paper['index']}. {paper.get('title') or 'Untitled paper'}{pmid_suffix}\n"
            f"   Abstract/record: {_truncate_for_message(paper.get('abstract', ''), 700)}"
        )

    payload = {
        "message": "\n".join(summary_lines),
        "metadata": {
            "service": "literature_search",
            "tool": "search_pubmed_abstracts_langchain",
            "query": query,
            "session_id": session_id,
            "source": source,
            "fallback_reason": fallback_reason,
            "paper_count": len(papers),
            "papers": papers,
        },
    }
    payload["metadata"]["saved_file"] = _save_payload(
        payload, session_id, f"pubmed_langchain_abstracts_{_safe_filename(query)}"
    )
    return _json_response(payload)


@mcp.tool()
async def search_pubmed_full_papers(
    query: str,
    top_k: int = 8,
    session_id: Optional[str] = None,
    max_message_chars_per_paper: int = 1200,
    min_full_text: int = 8,
) -> str:
    """
    Search PubMed, retrieve readable full papers when available, and return raw text.

    The JSON response includes an LLM-friendly message and metadata.papers where
    each paper carries full_text, text_length, source_file, DOI/PMID, abstract,
    and original PubMed metadata.

    Args:
        query: PubMed query.
        top_k: Target number of readable full-text papers to retrieve.
        session_id: Stable session ID for saved payloads.
        max_message_chars_per_paper: Preview length in message.
        min_full_text: If > 0, mark the payload as insufficient unless at
            least this many papers contain extracted full_text.
    """
    session_id = session_id or _new_session_id("lit_full")
    try:
        processor = _literature_processor()
    except Exception as exc:
        payload = {
            "message": f"PubMed full-paper retrieval is unavailable: {exc}",
            "metadata": {
                "service": "literature_search",
                "tool": "search_pubmed_full_papers",
                "query": query,
                "session_id": session_id,
                "source": "pubmed",
                "paper_count": 0,
                "min_full_text": min_full_text,
                "meets_min_full_text": min_full_text <= 0,
                "papers": [],
                "error": str(exc),
            },
        }
        payload["metadata"]["saved_file"] = _save_payload(
            payload, session_id, f"pubmed_full_papers_{_safe_filename(query)}"
        )
        return _json_response(payload)

    try:
        paper_cache_dir = await asyncio.to_thread(
            processor.cache_paper,
            [query],
            top_k,
            session_id,
        )
    except Exception as exc:
        payload = {
            "message": f"PubMed full-paper retrieval failed while caching papers: {exc}",
            "metadata": {
                "service": "literature_search",
                "tool": "search_pubmed_full_papers",
                "query": query,
                "session_id": session_id,
                "source": "pubmed",
                "paper_count": 0,
                "min_full_text": min_full_text,
                "meets_min_full_text": min_full_text <= 0,
                "papers": [],
                "error": str(exc),
            },
        }
        payload["metadata"]["saved_file"] = _save_payload(
            payload, session_id, f"pubmed_full_papers_{_safe_filename(query)}"
        )
        return _json_response(payload)
    metadata_path = os.path.join(processor.jsonl_cache_dir, f"{session_id}.jsonl")
    raw_papers = _read_jsonl(metadata_path)

    papers: List[Dict[str, Any]] = []
    for index, raw_paper in enumerate(raw_papers, 1):
        doi = raw_paper.get("doi", "")
        source_file = _find_cached_file_in_dirs(
            getattr(processor, "additional_cache_dirs", [paper_cache_dir]),
            doi,
        )
        full_text = ""
        extraction_error = ""

        if source_file:
            try:
                full_text = await asyncio.to_thread(_extract_full_text, source_file)
            except Exception as exc:  # Keep the paper metadata even if extraction fails.
                extraction_error = str(exc)

        if source_file and full_text:
            paper = _abstract_record(raw_paper, index)
            paper.update(
                {
                    "source_file": source_file,
                    "file_type": os.path.splitext(source_file)[1].lstrip("."),
                    "full_text": full_text,
                    "text_length": len(full_text),
                    "extraction_error": extraction_error,
                }
            )
            papers.append(paper)

        if len(papers) >= top_k:
            break

    meets_min_full_text = min_full_text <= 0 or len(papers) >= min_full_text
    min_full_text_error = ""
    if not meets_min_full_text:
        min_full_text_error = (
            f"Only {len(papers)} full-text papers were retrieved, "
            f"below required min_full_text={min_full_text}."
        )

    summary_lines = [
        f"PubMed full-paper search for '{query}' retrieved {len(papers)} readable papers.",
    ]
    if min_full_text > 0:
        summary_lines.append(
            f"Full-text requirement: {len(papers)}/{min_full_text} "
            f"({'met' if meets_min_full_text else 'not met'})."
        )
    for paper in papers:
        summary_lines.append(
            f"{paper['index']}. {_paper_citation(paper)}\n"
            f"   Full text: {paper['text_length']} characters from {paper['source_file']}\n"
            f"   Preview: {_truncate_for_message(paper.get('full_text', ''), max_message_chars_per_paper)}"
        )

    payload = {
        "message": "\n".join(summary_lines),
        "metadata": {
            "service": "literature_search",
            "tool": "search_pubmed_full_papers",
            "query": query,
            "session_id": session_id,
            "source": "pubmed",
            "paper_count": len(papers),
            "min_full_text": min_full_text,
            "meets_min_full_text": meets_min_full_text,
            "pubmed_jsonl": metadata_path,
            "paper_cache_dir": paper_cache_dir,
            "papers": papers,
        },
    }
    if min_full_text_error:
        payload["metadata"]["error"] = min_full_text_error
    payload["metadata"]["saved_file"] = _save_payload(
        payload, session_id, f"pubmed_full_papers_{_safe_filename(query)}"
    )
    return _json_response(payload)


@mcp.tool()
async def search_web_full_text(
    query: str,
    target_results: int = 32,
    session_id: Optional[str] = None,
    use_llm_filter: bool = False,
    max_message_chars_per_source: int = 1000,
) -> str:
    """
    Search the web with the text-browser extractor and preserve raw page text.

    Use this for non-PubMed literature, preprints, PDFs reachable by URL, lab
    pages, publisher pages, and other web sources. metadata.sources[].raw_text
    contains the extracted page text.
    """
    session_id = session_id or _new_session_id("lit_web")
    try:
        from tools.google_search_tools.google_search_w3m import google_search
    except Exception as exc:
        payload = {
            "message": f"Text-browser web search is unavailable: {exc}",
            "metadata": {
                "service": "literature_search",
                "tool": "search_web_full_text",
                "query": query,
                "session_id": session_id,
                "source": "text_browser",
                "source_count": 0,
                "sources": [],
                "error": str(exc),
            },
        }
        payload["metadata"]["saved_file"] = _save_payload(
            payload, session_id, f"web_full_text_{_safe_filename(query)}"
        )
        return _json_response(payload)

    results = await asyncio.to_thread(
        google_search,
        query,
        target_results,
        use_llm_filter,
    )

    sources: List[Dict[str, Any]] = []
    for index, result in enumerate(results, 1):
        raw_text = result.get("body", "") or ""
        sources.append(
            {
                "index": index,
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "raw_text": raw_text,
                "text_length": len(raw_text),
                "llm_summary": result.get("llm_summary", ""),
                "llm_confidence": result.get("llm_confidence", None),
                "llm_key_findings": result.get("llm_key_findings", []),
                "raw_search_result": dict(result),
            }
        )

    summary_lines = [
        f"Text-browser literature search for '{query}' returned {len(sources)} sources.",
    ]
    for source in sources[: min(5, len(sources))]:
        summary_lines.append(
            f"{source['index']}. {source['title']}\n"
            f"   URL: {source['url']}\n"
            f"   Extracted text: {source['text_length']} characters\n"
            f"   Preview: {_truncate_for_message(source.get('raw_text', ''), max_message_chars_per_source)}"
        )

    payload = {
        "message": "\n".join(summary_lines),
        "metadata": {
            "service": "literature_search",
            "tool": "search_web_full_text",
            "query": query,
            "session_id": session_id,
            "source": "text_browser",
            "source_count": len(sources),
            "sources": sources,
        },
    }
    payload["metadata"]["saved_file"] = _save_payload(
        payload, session_id, f"web_full_text_{_safe_filename(query)}"
    )
    return _json_response(payload)


@mcp.tool()
async def search_literature_bundle(
    query: str,
    abstract_top_k: int = 32,
    full_paper_top_k: int = 8,
    web_results: int = 32,
    session_id: Optional[str] = None,
    abstract_source: str = "langchain",
    min_full_text: int = 8,
) -> str:
    """
    Run abstract search, PubMed full-paper retrieval, and text-browser search.

    This is the convenience tool for agents that want a single literature
    package. The response metadata contains the parsed payloads from all three
    underlying search modes.
    """
    session_id = session_id or _new_session_id("lit_bundle")

    if abstract_source.lower() == "paperscraper":
        abstract_task = search_pubmed_abstracts(
            query=query,
            top_k=abstract_top_k,
            session_id=session_id,
        )
    else:
        abstract_task = search_pubmed_abstracts_langchain(
            query=query,
            top_k=abstract_top_k,
            session_id=session_id,
        )

    abstracts_raw, full_raw, web_raw = await asyncio.gather(
        abstract_task,
        search_pubmed_full_papers(
            query=query,
            top_k=full_paper_top_k,
            session_id=session_id,
            min_full_text=min_full_text,
        ),
        search_web_full_text(query=query, target_results=web_results, session_id=session_id),
    )

    abstracts = json.loads(abstracts_raw)
    full_papers = json.loads(full_raw)
    web_sources = json.loads(web_raw)

    payload = {
        "message": "\n\n".join(
            [
                abstracts["message"],
                full_papers["message"],
                web_sources["message"],
            ]
        ),
        "metadata": {
            "service": "literature_search",
            "tool": "search_literature_bundle",
            "query": query,
            "session_id": session_id,
            "abstract_source": abstract_source,
            "min_full_text": min_full_text,
            "abstract_search": abstracts["metadata"],
            "full_paper_search": full_papers["metadata"],
            "web_full_text_search": web_sources["metadata"],
        },
    }
    payload["metadata"]["saved_file"] = _save_payload(
        payload, session_id, f"literature_bundle_{_safe_filename(query)}"
    )
    return _json_response(payload)


if __name__ == "__main__":
    if "--sse" in sys.argv:
        mcp.run(transport="sse", port=9012)
    else:
        mcp.run()
