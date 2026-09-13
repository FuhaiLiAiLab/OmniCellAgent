"""Enrichr analysis of saved donor-pseudobulk DESeq2 results.

Only gene symbols are sent to the existing Enrichr ``addList``/``enrich``
endpoints. A tested-gene background is deliberately not submitted: the approved
analysis uses the service default. The service does not return its actual
universe size or membership in these responses.

``regenerate_enrichment_plots`` reads only the saved enrichment tables. Neither
entry point fits a differential-expression model or recomputes DE statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# Retain the established public entry point and private plotting helper imports.
from tools.omic_tools.pseudobulk_pipeline.enrichment_plots import (
    COMBINED_CATEGORIES,
    _direction_label,
    _read_plot_table,
    _render_plot,
    regenerate_enrichment_plots,
)


ENRICHR_URL = "https://maayanlab.cloud/Enrichr"
# Preserve perform_enrichment_analysis(..., fast_mode=False) library choices.
LIBRARIES = (
    "GO_Biological_Process_2021",
    "GO_Molecular_Function_2021",
    "GO_Cellular_Component_2021",
    "KEGG_2021_Human",
    "Reactome_2022",
    "WikiPathways_2019_Human",
    "MSigDB_Hallmark_2020",
    "DisGeNET",
    "OMIM_Disease",
    "OMIM_Expanded",
    "Human_Phenotype_Ontology",
    "Jensen_DISEASES",
    "GTEx_Tissue_Expression_Down",
    "GTEx_Tissue_Expression_Up",
)
DIRECTIONS = ("all", "up", "down")
TABLE_COLUMNS = (
    "rank", "term", "p_value", "api_score_4", "combined_score",
    "overlapping_genes", "adjusted_p_value", "old_p_value",
    "old_adjusted_p_value", "overlap_count", "library", "direction",
    "extra_api_fields",
)
REQUIRED_DE_COLUMNS = {
    "gene", "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj",
    "tested", "filter_reason",
}


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    """Replace only this component's own metadata files atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_options(alpha, log2fc_min, timeout, retries):
    if not math.isfinite(alpha) or not 0 < alpha <= 1:
        raise ValueError("alpha must be finite and in (0, 1]")
    if not math.isfinite(log2fc_min) or log2fc_min < 0:
        raise ValueError("log2fc_min must be finite and nonnegative")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("retries must be a nonnegative integer")


def _read_results(path, alpha, log2fc_min):
    frame = pd.read_csv(path, keep_default_na=False, dtype={"gene": str})
    missing = REQUIRED_DE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"results.csv lacks required columns: {sorted(missing)}")
    gene = frame["gene"].str.strip()
    if gene.eq("").any() or gene.str.contains(r"[\s\x00-\x1f\x7f]", regex=True).any():
        raise ValueError("gene symbols must be nonempty and contain no whitespace/control characters")
    if gene.duplicated().any():
        raise ValueError("results.csv has duplicate gene symbols")
    frame["gene"] = gene
    boolean = frame["tested"].astype(str).str.lower().str.strip()
    if not boolean.isin(("true", "false", "1", "0")).all():
        raise ValueError("tested must contain explicit true/false or 1/0 values")
    frame["tested"] = boolean.isin(("true", "1"))
    for column in ("padj", "log2FoldChange"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    finite_padj = np.isfinite(frame["padj"])
    if ((frame.loc[finite_padj, "padj"] < 0) | (frame.loc[finite_padj, "padj"] > 1)).any():
        raise ValueError("finite padj values must be in [0, 1]")
    significant = (
        frame["tested"] & finite_padj & np.isfinite(frame["log2FoldChange"])
        & frame["padj"].lt(alpha) & frame["log2FoldChange"].abs().ge(log2fc_min)
    )
    selected = frame.loc[significant].copy()
    return frame, {
        "all": selected,
        "up": selected.loc[selected["log2FoldChange"] > 0].copy(),
        "down": selected.loc[selected["log2FoldChange"] < 0].copy(),
    }


def _request_json(method, endpoint, raw_dir, stem, timeout, retries, **kwargs):
    """Retain the response body and request metadata for every HTTP attempt."""
    url = f"{ENRICHR_URL}/{endpoint}"
    summary = {"status": "error", "attempts": 0, "history": []}
    for attempt in range(1, retries + 2):
        record = {
            "attempt": attempt, "method": method.upper(), "url": url,
            "timestamp": _utc_now(), "timeout_seconds": timeout,
        }
        if "params" in kwargs:
            record["params"] = kwargs["params"]
        response = None
        payload = None
        retryable = True
        try:
            response = getattr(requests, method)(url, timeout=timeout, **kwargs)
            body_file = raw_dir / f"{stem}.attempt-{attempt}.body.txt"
            body_file.write_bytes(response.content)
            record["raw_body"] = str(body_file.resolve())
            record["http_status"] = response.status_code
            record["response_headers"] = {
                name: response.headers[name] for name in ("Content-Type", "Date", "Retry-After")
                if name in response.headers
            }
            response.raise_for_status()
            # Enrichr occasionally serializes Infinity/NaN. Keep exact raw text;
            # stdlib JSON parsing preserves these values for the complete CSV.
            payload = json.loads(response.text)
            if not isinstance(payload, dict):
                raise ValueError("Enrichr response must be a JSON object")
            record["status"] = "success"
        except (requests.RequestException, ValueError) as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            if response is not None and 400 <= response.status_code < 500:
                retryable = response.status_code in (408, 425, 429)
        _write_json(raw_dir / f"{stem}.attempt-{attempt}.json", record)
        summary["history"].append(record)
        summary["attempts"] = attempt
        if record["status"] == "success":
            summary["status"] = "success"
            return payload, summary
        summary["error"] = record["error"]
        if not retryable or attempt > retries:
            break
        delay = min(2 ** (attempt - 1), 30)
        if response is not None:
            try:
                delay = min(max(delay, float(response.headers.get("Retry-After", 0))), 30)
            except ValueError:
                pass
        time.sleep(delay)
    return None, summary


def _parse_library(payload, library, direction):
    if library not in payload or not isinstance(payload[library], list):
        raise ValueError(f"Enrichr response lacks a list for requested library {library}")
    records = []
    for index, row in enumerate(payload[library]):
        if not isinstance(row, list) or len(row) < 7:
            raise ValueError(f"{library} result row {index} has fewer than 7 fields")
        genes = row[5]
        if not isinstance(genes, list) or not all(isinstance(gene, str) for gene in genes):
            raise ValueError(f"{library} result row {index} lacks an overlapping-gene list")
        if not isinstance(row[1], str):
            raise ValueError(f"{library} result row {index} lacks a term name")
        records.append({
            "rank": row[0], "term": row[1], "p_value": row[2],
            "api_score_4": row[3], "combined_score": row[4],
            "overlapping_genes": json.dumps(genes, ensure_ascii=False),
            "adjusted_p_value": row[6],
            "old_p_value": row[7] if len(row) > 7 else None,
            "old_adjusted_p_value": row[8] if len(row) > 8 else None,
            "overlap_count": len(set(genes)), "library": library,
            "direction": direction, "extra_api_fields": json.dumps(row[9:]),
        })
    return pd.DataFrame(records, columns=TABLE_COLUMNS)


def _combined_status(statuses):
    meaningful = [status for status in statuses if status not in ("empty", "skipped")]
    if not meaningful:
        return "empty"
    if all(status == "success" for status in meaningful):
        return "success"
    if all(status == "error" for status in meaningful):
        return "error"
    return "partial"


def _background_metadata():
    return {
        "mode": "server_default",
        "custom_background_submitted": False,
        "tested_genes_used_as_background": False,
        "server_universe_size": None,
        "server_universe_membership": None,
        "universe_verification": "Unknown: addList/enrich responses do not report the actual server universe.",
        "documented_nominal_size": 20000,
        "documented_nominal_size_note": (
            "Official help describes 20,000 genes (or the background total) in the odds-ratio formula. "
            "This does not establish the runtime universe size or its gene membership."
        ),
        "documentation_url": f"{ENRICHR_URL}/templates/help/background-content.html",
        "documentation_checked": "2026-09-10",
        "backgroundType_parameter": "Requested annotation library name; not a submitted background gene list.",
    }


def _build_metadata(comparison_dir, source, frame, alpha, log2fc_min, timeout, retries):
    """Describe the input provenance, fixed API contract, and selection rules."""
    status_path = comparison_dir / "status.json"
    comparison_status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    contrast = comparison_status.get("contrast", {})
    metadata = {
        "schema_version": 1, "status": "running", "started_at": _utc_now(),
        "comparison": comparison_dir.name, "contrast": contrast,
        "source": {"path": str(source), "sha256": _hash_file(source), "row_count": len(frame)},
        "background": _background_metadata(), "libraries": list(LIBRARIES),
        "thresholds": {
            "alpha": alpha, "log2fc_min": log2fc_min, "tested_required": True,
            "finite_padj_required": True, "finite_log2fc_required": True,
            "padj_operator": "<", "absolute_log2fc_operator": ">=", "gene_cap": None,
        },
        "direction_definition": {
            "all": "All qualifying significant genes, regardless of fold-change sign",
            "up": "Positive log2FoldChange: greater expression in the contrast numerator",
            "down": "Negative log2FoldChange: lower expression in the contrast numerator",
        },
        "interpretation": "Overrepresentation among selected genes does not establish pathway activation or inhibition.",
        "api": {
            "base_url": ENRICHR_URL, "timeout_seconds": timeout, "retries": retries,
            "api_score_4": "Fourth returned field, preserved without reinterpretation; current API help labels it Odds ratio.",
            "score_documentation_url": f"{ENRICHR_URL}/templates/help/api-content.html",
            "documentation_checked": "2026-09-10",
            "adjusted_p_values": "Returned Enrichr adjusted p-values; no across-library adjustment is applied locally.",
            "table_row_cap": None,
            "recognized_gene_count": None,
            "recognized_gene_count_note": "addList does not report how many submitted genes are recognized per library.",
        },
        "directions": {},
    }
    if status_path.is_file():
        metadata["source"]["status_sha256"] = _hash_file(status_path)
    return metadata


def _enrich_direction(comparison_dir, output, direction, selected, metadata, timeout, retries):
    """Save one foreground, upload it, and retain every library result and failure.

    The running result is attached to shared metadata before requests begin so
    each saved checkpoint includes the direction's progress.
    """
    directory = output / direction
    raw_dir = directory / "raw"
    table_dir = directory / "tables"
    raw_dir.mkdir(parents=True)
    table_dir.mkdir()
    genes = selected["gene"].tolist()
    (directory / "genes.txt").write_text("".join(f"{gene}\n" for gene in genes))
    selected.to_csv(directory / "selected_genes.csv", index=False)
    result = {
        "status": "running", "direction": direction, "gene_count": len(genes),
        "gene_list": str(directory / "genes.txt"),
        "gene_list_sha256": _hash_file(directory / "genes.txt"), "libraries": {},
    }
    metadata["directions"][direction] = result
    _write_json(directory / "status.json", result)
    if not genes:
        result.update(status="empty", reason="No genes satisfy this direction's DE selection criteria.")
        return result

    payload, upload = _request_json(
        "post", "addList", raw_dir, "addList", timeout, retries,
        files={"list": (None, "\n".join(genes))},
        data={"description": f"Donor pseudobulk: {comparison_dir.name}; {direction}"},
    )
    result["upload"] = upload
    list_id = payload.get("userListId") if payload is not None else None
    if isinstance(list_id, bool) or not isinstance(list_id, (int, str)) or not str(list_id).isdigit():
        result.update(status="error", reason="Enrichr addList did not return a valid userListId.")
        if upload["status"] == "success":
            upload.update(status="error", error=result["reason"])
        return result

    result["user_list_id"] = list_id
    result["short_id"] = payload.get("shortId")
    for library in LIBRARIES:
        payload, request_status = _request_json(
            "get", "enrich", raw_dir, library, timeout, retries,
            params={"userListId": list_id, "backgroundType": library},
        )
        library_status = {"status": "error", "request": request_status}
        result["libraries"][library] = library_status
        if payload is None:
            library_status["error"] = request_status["error"]
        else:
            try:
                table = _parse_library(payload, library, direction)
                table_path = table_dir / f"{library}.csv"
                table.to_csv(table_path, index=False)
                library_status.update(
                    status="success", row_count=len(table),
                    table=str(table_path), sha256=_hash_file(table_path),
                )
            except (ValueError, TypeError) as exc:
                library_status["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(directory / "status.json", result)
        _write_json(output / "metadata.json", metadata)
    result["status"] = _combined_status(
        item["status"] for item in result["libraries"].values()
    )
    return result


def run_enrichment(comparison_dir, alpha=0.05, log2fc_min=0.0, timeout=30, retries=2):
    """Submit all qualifying genes separately as all/up/down and save every result.

    ``all`` uses tested genes with finite ``padj`` and log2 fold change,
    ``padj < alpha``, and ``abs(log2FoldChange) >= log2fc_min``. ``up`` and
    ``down`` additionally require positive and negative fold change. There is
    no gene cap. HTTP errors become recorded statuses and do not discard other
    libraries. ``retries`` counts retries after the initial HTTP attempt.

    A fresh ``comparison_dir/enrichment`` directory is required. Invalid inputs
    fail before network calls. Plot regeneration is available independently and
    requires neither a new upload nor differential-expression recomputation.
    """
    # Validate all local inputs before creating output or making HTTP requests.
    _validate_options(alpha, log2fc_min, timeout, retries)
    comparison_dir = Path(comparison_dir).resolve()
    output = comparison_dir / "enrichment"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing enrichment output: {output}")
    source = comparison_dir / "results.csv"
    frame, gene_sets = _read_results(source, alpha, log2fc_min)
    metadata = _build_metadata(
        comparison_dir, source, frame, alpha, log2fc_min, timeout, retries,
    )
    output.mkdir()
    _write_json(output / "metadata.json", metadata)

    # Submit and checkpoint the independent all/up/down foregrounds.
    for direction, selected in gene_sets.items():
        result = _enrich_direction(
            comparison_dir, output, direction, selected, metadata, timeout, retries,
        )
        _write_json(output / direction / "status.json", result)
        _write_json(output / "metadata.json", metadata)

    # Finish API statuses before regenerating figures from the saved tables.
    metadata["status"] = _combined_status(item["status"] for item in metadata["directions"].values())
    _write_json(output / "metadata.json", metadata)
    try:
        metadata["plots"] = regenerate_enrichment_plots(comparison_dir, alpha=alpha)
    except (ImportError, ValueError, OSError) as exc:
        metadata["plots"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        if metadata["status"] == "success":
            metadata["status"] = "partial"
    metadata["completed_at"] = _utc_now()
    _write_json(output / "metadata.json", metadata)
    return metadata
