"""
Check that the Wiley TDM API key works, both directly and through the pipeline's download path.

1. Direct: GET the Wiley TDM API with the key the pipeline uses and confirm a PDF comes back.
2. Pipeline: download the same DOIs with MedicalResearchProcessor's paperscraper call into a
   temporary folder (not the global DOI cache, so cached papers can't hide a failure).

The key is read the same way the pipeline reads it: WILEY_TDM_API_TOKEN from the project .env,
falling back to the shell environment. The key itself is never printed.

Usage (from the project root):
    python tools/pubmed_tools/test/test_wiley_api.py
    python tools/pubmed_tools/test/test_wiley_api.py --doi 10.1002/dad2.70495
"""

import os
import sys
import json
import argparse
import tempfile
import urllib.parse

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from paperscraper.pdf import save_pdf_from_dump
from tools.pubmed_tools.query_pubmed_tool import MedicalResearchProcessor

from dotenv import load_dotenv, dotenv_values
load_dotenv()



# Wiley DOIs (10.1002 / 10.1111) the TDM API returned for our key. HTTP 403 means the
# institution's license does not cover that article, not that the key is wrong.
DEFAULT_DOIS = ["10.1002/dad2.70495", "10.1111/jnc.15248"]
WILEY_TDM_URL = "https://api.wiley.com/onlinelibrary/tdm/v1/articles/"


def check_direct(doi: str, key: str) -> bool:
    r = requests.get(WILEY_TDM_URL + urllib.parse.quote(doi, safe=""),
                     headers={"Wiley-TDM-Client-Token": key}, timeout=60)
    is_pdf = r.content[:4] == b"%PDF"
    note = "  (not covered by license)" if r.status_code == 403 else ""
    print(f"  direct   {doi:25} HTTP {r.status_code}  PDF={is_pdf}  {len(r.content)} bytes{note}")
    return r.status_code == 200 and is_pdf


def check_pipeline(doi: str, processor: MedicalResearchProcessor, out_dir: str) -> bool:
    dump_path = os.path.join(out_dir, "paper.jsonl")
    with open(dump_path, "w") as f:
        f.write(json.dumps({"doi": doi}) + "\n")

    before = set(os.listdir(out_dir))
    save_pdf_from_dump(dump_path, pdf_path=out_dir, key_to_save="doi",
                       api_keys=processor._api_keys_for_publisher("Wiley"))
    new_files = [f for f in set(os.listdir(out_dir)) - before if f.endswith((".pdf", ".xml"))]

    ok = False
    for name in new_files:
        with open(os.path.join(out_dir, name), "rb") as f:
            head = f.read(4)
        ok = ok or head == b"%PDF" or name.endswith(".xml")
    print(f"  pipeline {doi:25} files={new_files or 'none'}  OK={ok}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Check the Wiley TDM API key")
    parser.add_argument("--doi", action="append", help="Wiley DOI to test (repeatable)")
    args = parser.parse_args()
    dois = args.doi or DEFAULT_DOIS

    processor = MedicalResearchProcessor()
    key = processor.api_keys.get("WILEY_TDM_API_TOKEN")
    if not key:
        print("FAIL: no WILEY_TDM_API_TOKEN found in the project .env or the shell environment")
        sys.exit(1)
    from_env_file = key == dotenv_values(processor.api_keys_path).get("WILEY_TDM_API_TOKEN")
    print(f"Using WILEY_TDM_API_TOKEN from: {'project .env' if from_env_file else 'shell environment'}")

    results = []
    with tempfile.TemporaryDirectory() as out_dir:
        for doi in dois:
            results.append(check_direct(doi, key))
            results.append(check_pipeline(doi, processor, out_dir))

    ok = all(results)
    print(f"\n{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
