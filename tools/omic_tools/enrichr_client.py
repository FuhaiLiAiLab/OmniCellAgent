"""Bounded Enrichr requests, audited raw responses and typed legacy CSVs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
import uuid

import numpy as np
import pandas as pd
import requests

URL = "https://maayanlab.cloud/Enrichr"
DEFAULT_LIBRARIES = (
    "GO_Biological_Process_2021", "GO_Molecular_Function_2021", "GO_Cellular_Component_2021",
    "KEGG_2021_Human", "Reactome_2022", "WikiPathways_2019_Human", "MSigDB_Hallmark_2020",
    "DisGeNET", "OMIM_Disease", "OMIM_Expanded", "Human_Phenotype_Ontology", "Jensen_DISEASES",
    "GTEx_Tissue_Expression_Down", "GTEx_Tissue_Expression_Up",
)
ENRICHMENT_DTYPES = {
    "Rank": "int64", "Term": "object", "P-value": "float64", "Odds Ratio": "float64",
    "Combined Score": "float64", "Genes": "object", "Adjusted P-value": "float64",
    "Old P-value": "float64", "Old Adjusted P-value": "float64",
}


class EnrichrError(RuntimeError):
    """An HTTP, protocol or output failure; never an empty biological result."""


def write_status(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def record_enrichment_failure(directory, error):
    """Record failures at setup/summary boundaries as well as during HTTP."""
    try:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "enrichment_status.json"
        try:
            status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (ValueError, OSError):
            status = {}
        if status.get("status") != "failed":
            status.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_status(path, status)
    except OSError as status_error:
        print(f"[Enrichr] Could not persist failure status: {status_error}")


def archive_enrichment_plots(directory):
    directory = Path(directory)
    old = [path for extension in ("png", "pdf", "html") for path in directory.glob(f"*_regulated.{extension}")]
    if old:
        history = directory / "history" / uuid.uuid4().hex
        history.mkdir(parents=True)
        for path in old:
            path.replace(history / path.name)


def parse_enrichment_payload(payload, library):
    if not isinstance(payload, dict) or library not in payload or not isinstance(payload[library], list):
        raise ValueError(f"Response must contain a list for {library}; a missing key is not an empty result")
    records = []
    for index, row in enumerate(payload[library]):
        if not isinstance(row, list) or len(row) < 7:
            raise ValueError(f"{library} row {index} has fewer than seven required fields")
        if not isinstance(row[1], str) or not row[1]:
            raise ValueError(f"{library} row {index} has an invalid Term")
        if not isinstance(row[5], list) or not all(isinstance(gene, str) for gene in row[5]):
            raise ValueError(f"{library} row {index} has an invalid Genes list")
        records.append(row[:7] + [row[7] if len(row) > 7 else np.nan,
                                  row[8] if len(row) > 8 else np.nan])
    table = pd.DataFrame(records, columns=list(ENRICHMENT_DTYPES))
    for name, dtype in ENRICHMENT_DTYPES.items():
        if dtype != "object":
            table[name] = pd.to_numeric(table[name], errors="raise")
    if not table.empty:
        rank = table["Rank"].to_numpy()
        if not np.isfinite(rank).all() or (rank < 1).any() or (rank != np.floor(rank)).any():
            raise ValueError("Enrichr ranks must be positive integers")
        for column in ("P-value", "Adjusted P-value"):
            values = table[column].to_numpy()
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f"Invalid Enrichr probability column: {column}")
    return table.astype(ENRICHMENT_DTYPES)


def read_enrichment_results(path):
    """Read the legacy nine-column CSV; refuse a failed/running current run."""
    path = Path(path)
    for folder in (path.parent, path.parent.parent):
        status_path = folder / "enrichment_status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status["status"] not in ("success", "empty", "skipped"):
                raise EnrichrError(f"Cannot read {path}: enrichment status={status['status']}")
    table = pd.read_csv(path, dtype=ENRICHMENT_DTYPES, encoding="utf-8")
    if table.columns.tolist() != list(ENRICHMENT_DTYPES):
        raise ValueError(f"Unexpected enrichment CSV columns: {path}")
    return table


def fetch_enrichment(gene_list, sample_id, output_dir, databases=None, request_timeout=60):
    try:
        return _fetch_enrichment(gene_list, sample_id, output_dir, databases, request_timeout)
    except EnrichrError:
        raise
    except Exception as error:
        if output_dir is not None and Path(sample_id).name == sample_id:
            record_enrichment_failure(Path(output_dir) / sample_id, error)
        raise EnrichrError(f"Enrichr setup/output failed for {sample_id}: {error}") from error


def _fetch_enrichment(gene_list, sample_id, output_dir, databases=None, request_timeout=60):
    """Keep old outputs in history; publish only a complete current response set.

    Successful returns retain the existing {library: {library: rows}} shape.
    Empty lists write typed header-only tables. Failed requests raise and write
    status=failed, with no old table left at the current consumer path.
    """
    if not np.isfinite(request_timeout) or request_timeout <= 0:
        raise ValueError("request_timeout must be finite and positive")
    if Path(sample_id).name != sample_id:
        raise ValueError("sample_id must be one directory name")
    libraries = list(DEFAULT_LIBRARIES if databases is None else databases)
    if not libraries or len(set(libraries)) != len(libraries):
        raise ValueError("Specify at least one unique Enrichr library")
    genes = [str(gene).strip() for gene in gene_list]
    if any(not gene or "\n" in gene or "\r" in gene for gene in genes):
        raise ValueError("Gene symbols must be nonempty single-line strings")
    directory = Path(output_dir) / sample_id
    directory.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    owned = list(directory.glob("*_results.csv")) + [directory / name for name in
             ("all_enrichment_results.json", "summary.txt", "gene_list.txt", "enrichment_status.json", "raw")]
    owned = [path for path in owned if path.exists()]
    if owned:
        history = directory / "history" / run_id
        history.mkdir(parents=True)
        for path in owned:
            shutil.move(str(path), str(history / path.name))
        # Migrate audit references from older runs that used absolute paths.
        previous_status = history / "enrichment_status.json"
        if previous_status.exists():
            previous = json.loads(previous_status.read_text(encoding="utf-8"))
            for request in previous.get("requests", []):
                raw_path = Path(request.get("raw_body", ""))
                if raw_path.is_absolute():
                    try:
                        request["raw_body"] = str(raw_path.relative_to(directory.resolve()))
                    except ValueError:
                        pass
            write_status(previous_status, previous)
    raw_dir = directory / "raw"
    raw_dir.mkdir()
    gene_text = "\n".join(genes)
    (directory / "gene_list.txt").write_text(gene_text, encoding="utf-8")
    status_path = directory / "enrichment_status.json"
    status = {"run_id": run_id, "status": "running", "sample_id": sample_id,
              "submitted_genes": len(genes), "gene_list_sha256": hashlib.sha256(gene_text.encode()).hexdigest(),
              "started_at": datetime.now(timezone.utc).isoformat(), "request_timeout": request_timeout,
              "libraries": {}, "requests": []}
    write_status(status_path, status)

    def request_json(method, endpoint, stem, **kwargs):
        record = {"method": method.upper(), "endpoint": endpoint, "status": "running",
                  "timeout_seconds": request_timeout, "params": kwargs.get("params")}
        started = time.monotonic()
        try:
            response = getattr(requests, method)(f"{URL}/{endpoint}", timeout=request_timeout, **kwargs)
            record["http_status"] = response.status_code
            raw_path = raw_dir / f"{stem}.body.txt"
            raw_path.write_bytes(response.content)
            record["raw_body"] = str(raw_path.relative_to(directory))
            response.raise_for_status()
            payload = json.loads(response.text)
            if not isinstance(payload, dict):
                raise ValueError("Enrichr response is not a JSON object")
            record["status"] = "success"
            return payload
        except Exception as error:
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["elapsed_seconds"] = round(time.monotonic() - started, 4)
            status["requests"].append(record)
            write_status(status_path, status)
            print(f"[Enrichr] {sample_id} {method.upper()} {endpoint} HTTP={record.get('http_status')} status={record['status']}")

    stage = "addList"
    try:
        with tempfile.TemporaryDirectory(prefix=".new-", dir=directory) as temporary:
            staging = Path(temporary)
            if genes:
                uploaded = request_json("post", "addList", "addList", files={"list": (None, gene_text)},
                                        data={"description": f"DEA gene set - {sample_id}"})
                user_list_id = uploaded.get("userListId")
                if isinstance(user_list_id, bool) or not isinstance(user_list_id, int) or user_list_id <= 0:
                    raise ValueError("addList response lacks a positive integer userListId")
                status["user_list_id"] = user_list_id
            results = {}
            for library in libraries:
                stage = library
                safe = library.replace("/", "_")
                payload = (request_json("get", "enrich", safe,
                           params={"userListId": user_list_id, "backgroundType": library}) if genes else {library: []})
                table = parse_enrichment_payload(payload, library)
                table.sort_values("Adjusted P-value").head(50).to_csv(staging / f"{safe}_results.csv", index=False)
                results[library] = payload
                status["libraries"][library] = {
                    "status": "success" if len(table) else ("empty" if genes else "skipped"),
                    "returned_rows": len(table), "saved_rows": min(50, len(table)),
                    "significant_rows_fdr_0_05": int((table["Adjusted P-value"] < 0.05).sum()),
                    "row_widths": sorted({len(row) for row in payload[library]}),
                    "dtypes": table.dtypes.astype(str).to_dict(),
                }
                print(f"[Enrichr] {sample_id} {library}: returned={len(table)}, saved={min(50, len(table))}, status={status['libraries'][library]['status']}")
            (staging / "all_enrichment_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            for artifact in staging.iterdir():
                artifact.replace(directory / artifact.name)
        status["status"] = ("skipped" if not genes else
                            "success" if any(item["returned_rows"] for item in status["libraries"].values()) else "empty")
        write_status(status_path, status)
        return results
    except Exception as error:
        status.update(status="failed", failed_stage=stage, error=f"{type(error).__name__}: {error}")
        write_status(status_path, status)
        raise EnrichrError(f"Enrichr failed for {sample_id} at {stage}: {status['error']}") from error
