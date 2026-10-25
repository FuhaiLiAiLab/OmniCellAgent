"""
Fixes for paperscraper downloads, shared by the agent's PubMed search (query_pubmed_tool.py) and the
expert knowledge-base builder (scientist_rag_tools/scientist_tool.py).

Importing this module applies two process-wide changes:
  - requests connections allow TLS session tickets. NCBI's PMC servers (DOI->PMCID converter,
    full-text XML) answer 403 to urllib3's default TLS setup, which disables them, on OpenSSL 3.5
    (e.g. the omnicellagent env). paperscraper calls requests.get directly, so the change has to be
    made on requests' HTTPAdapter.
  - paperscraper's Wiley TDM fallback is replaced by one with configurable waits
    (WILEY_*_WAIT_SECONDS in get_papers_info_tools.py) that does not retry on 403/404.

It also provides per-publisher API keys: paperscraper tries the Wiley API after most failed downloads
whenever a Wiley key is present (and then never reaches Elsevier), so each paper is given only its own
publisher's key.
"""
import os
import ssl
import time
from pathlib import Path
from typing import Any, Dict

import requests
import paperscraper.pdf.fallbacks as paperscraper_fallbacks
from dotenv import dotenv_values
from paperscraper.pdf import save_pdf
from paperscraper.pdf.utils import load_api_keys
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

try:
    from .get_papers_info_tools import WILEY_RETRY_WAIT_SECONDS, WILEY_RATE_LIMIT_WAIT_SECONDS
except ImportError:
    from get_papers_info_tools import WILEY_RETRY_WAIT_SECONDS, WILEY_RATE_LIMIT_WAIT_SECONDS


# ---------- TLS session tickets for requests ----------

def _init_poolmanager_with_session_tickets(self, *args, **kwargs):
    if kwargs.get("ssl_context") is None:
        context = create_urllib3_context()
        context.options &= ~ssl.OP_NO_TICKET
        kwargs["ssl_context"] = context
    return _init_poolmanager_with_session_tickets.original(self, *args, **kwargs)


# Guard so importing this module under two names (package / script) patches only once;
# all other urllib3 TLS defaults, including certificate verification, are unchanged
if not getattr(HTTPAdapter.init_poolmanager, "allows_session_tickets", False):
    _init_poolmanager_with_session_tickets.original = HTTPAdapter.init_poolmanager
    _init_poolmanager_with_session_tickets.allows_session_tickets = True
    HTTPAdapter.init_poolmanager = _init_poolmanager_with_session_tickets


# ---------- Wiley TDM fallback ----------

def _fallback_wiley_api(paper_metadata: Dict[str, Any], output_path, api_keys: Dict[str, str],
                        max_attempts: int = 2) -> bool:
    """paperscraper's fallback_wiley_api with configurable waits (WILEY_*_WAIT_SECONDS)."""
    doi = paper_metadata["doi"]
    api_url = f"https://api.wiley.com/onlinelibrary/tdm/v1/articles/{doi.replace('/', '%2F')}"
    headers = {"Wiley-TDM-Client-Token": api_keys.get("WILEY_TDM_API_TOKEN")}
    logger = paperscraper_fallbacks.logger

    success = False
    for attempt in range(max_attempts):
        try:
            api_response = requests.get(api_url, headers=headers, allow_redirects=True, timeout=60)
            api_response.raise_for_status()
            if api_response.content[:4] != b"%PDF":
                logger.warning(f"API returned content that is not a valid PDF for {doi}.")
            else:
                with open(output_path.with_suffix(".pdf"), "wb+") as f:
                    f.write(api_response.content)
                logger.info(f"Successfully downloaded PDF via Wiley API for {doi}.")
                success = True
                break
        except Exception as e:
            logger.error(f"Could not download via Wiley API (attempt {attempt + 1}/{max_attempts}): {e}")
            # 403 = article not covered by the license, 404 = DOI unknown to Wiley; a retry cannot succeed
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (403, 404):
                break
            if attempt < max_attempts - 1:
                logger.info(f"Waiting {WILEY_RETRY_WAIT_SECONDS} seconds before retrying...")
                time.sleep(WILEY_RETRY_WAIT_SECONDS)

    logger.info(f"Waiting {WILEY_RATE_LIMIT_WAIT_SECONDS} seconds before next request to comply with Wiley API rate limits...")
    time.sleep(WILEY_RATE_LIMIT_WAIT_SECONDS)
    return success


# paperscraper looks fallbacks up in this shared dict at call time
paperscraper_fallbacks.FALLBACKS["wiley"] = _fallback_wiley_api


# ---------- Per-publisher API keys ----------

PUBLISHER_API_KEYS = {
    "Wiley": "WILEY_TDM_API_TOKEN",
    "Elsevier": "ELSEVIER_TDM_API_KEY",
}

PROJECT_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")


def load_project_api_keys(env_path: str = PROJECT_ENV_PATH) -> Dict[str, str]:
    """paperscraper's API keys, with values in the project .env taking precedence over the shell
    (load_dotenv never overrides variables the shell already exported)."""
    api_keys = load_api_keys(env_path)
    env_file_keys = dotenv_values(env_path)
    api_keys.update({k: env_file_keys[k] for k in api_keys if env_file_keys.get(k)})
    return api_keys


def publisher_for_doi(doi: str) -> str:
    """Publisher guessed from the DOI prefix (used for API-key selection and log messages)."""
    doi_lower = doi.lower()
    if "wiley" in doi_lower or "10.1002" in doi or "10.1111" in doi:
        return "Wiley"
    if "elsevier" in doi_lower or "10.1016" in doi:
        return "Elsevier"
    if "springer" in doi_lower or "10.1007" in doi:
        return "Springer"
    if "nature" in doi_lower or "10.1038" in doi:
        return "Nature"
    if "pnas" in doi_lower or "10.1073" in doi:
        return "PNAS"
    if "10.1371" in doi:
        return "PLOS"
    if "10.3389" in doi:
        return "Frontiers"
    return "unknown"


def api_keys_for_publisher(api_keys: Dict[str, str], publisher: str) -> Dict[str, str]:
    """The API keys paperscraper may use for a paper from this publisher."""
    keys = {k: v for k, v in api_keys.items() if v and k not in PUBLISHER_API_KEYS.values()}
    key_name = PUBLISHER_API_KEYS.get(publisher)
    if key_name and api_keys.get(key_name):
        keys[key_name] = api_keys[key_name]
    return keys


def download_paper(paper: Dict[str, Any], out_dir: str, api_keys: Dict[str, str]) -> str:
    """Download one paperscraper dump record to <out_dir>/<doi with / as _>.pdf (or .xml), passing only
    its publisher's API key. Same file naming and skip rule as paperscraper's save_pdf_from_dump.

    Returns "cached", "downloaded", "failed" or "no_doi".
    """
    doi = paper.get("doi")
    if not doi:
        return "no_doi"
    pdf_file = Path(out_dir) / f"{doi.replace('/', '_')}.pdf"
    xml_file = pdf_file.with_suffix(".xml")
    if pdf_file.exists() or xml_file.exists():
        return "cached"
    try:
        save_pdf(paper, str(pdf_file), api_keys=api_keys_for_publisher(api_keys, publisher_for_doi(doi)))
    except Exception as e:
        print(f"[Error] Download failed for {doi}: {e}")
        return "failed"
    return "downloaded" if (pdf_file.exists() or xml_file.exists()) else "failed"
