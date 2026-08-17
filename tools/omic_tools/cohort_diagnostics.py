"""Cohort comparability diagnostics for the omic DE workflow.

Pure functions over cell metadata. No I/O, no loader import.

The gate WARNS; it never withholds results. Callers always produce a DE table
and attach these diagnostics to it.
"""
import numpy as np
import pandas as pd

UNUSABLE_DONOR_VALUES = {"", "nan", "none", "unknown", "na"}

PROTOCOL_IMBALANCE_FAIL = 0.30
DATASET_OVERLAP_CAUTION = 0.20
MIN_DONORS_FAIL = 10
DONOR_DOMINANCE_FAIL = 0.30
DONOR_UNUSABLE_FAIL = 0.20
DEPTH_RATIO_CAUTION_LOW = 0.67
DEPTH_RATIO_CAUTION_HIGH = 1.50


def _usable_donor_mask(series: pd.Series) -> np.ndarray:
    normalized = series.astype(str).str.strip().str.lower()
    return (~normalized.isin(UNUSABLE_DONOR_VALUES) & series.notna()).values


def compute_cohort_diagnostics(metadata, is_ref, is_alt, lib_sizes=None,
                               ref_name: str = "reference",
                               alt_name: str = "alternate") -> dict:
    """Assess whether two cell groups are comparable for differential expression.

    Args:
        metadata: per-cell DataFrame; may be missing any of the columns used.
        is_ref, is_alt: boolean arrays selecting the two groups.
        lib_sizes: per-cell totals BEFORE normalization. Optional.

    Returns a dict with `verdict` ("ok" | "caution" | "unreliable"),
    `failed_checks`, `caution_checks`, `checks`, and `groups`.
    """
    is_ref = np.asarray(is_ref, dtype=bool)
    is_alt = np.asarray(is_alt, dtype=bool)
    checks, failed, caution = {}, [], []

    def record(name, value, detail, status):
        checks[name] = {"check": name, "value": value, "detail": detail, "status": status}
        if status == "FAIL":
            failed.append(checks[name])
        elif status == "CAUTION":
            caution.append(checks[name])

    # --- protocol balance -------------------------------------------------
    if metadata is not None and "suspension_type" in metadata.columns:
        col = metadata["suspension_type"].astype(str)
        f_ref = float((col[is_ref] == "cell").mean()) if is_ref.any() else 0.0
        f_alt = float((col[is_alt] == "cell").mean()) if is_alt.any() else 0.0
        gap = abs(f_alt - f_ref)
        record(
            "protocol_balance", gap,
            f"whole-cell fraction {ref_name}={f_ref:.2f} vs {alt_name}={f_alt:.2f}; "
            "single-nucleus and whole-cell profiles are not interchangeable",
            "FAIL" if gap > PROTOCOL_IMBALANCE_FAIL else "OK",
        )

    # --- dataset overlap --------------------------------------------------
    if metadata is not None and "dataset_id" in metadata.columns:
        ref_sets = set(metadata.loc[is_ref, "dataset_id"].dropna())
        alt_sets = set(metadata.loc[is_alt, "dataset_id"].dropna())
        total = len(ref_sets | alt_sets)
        shared = len(ref_sets & alt_sets)
        overlap = (shared / total) if total else 0.0
        if shared == 0:
            status = "FAIL"
        elif overlap < DATASET_OVERLAP_CAUTION:
            status = "CAUTION"
        else:
            status = "OK"
        record(
            "dataset_overlap", overlap,
            f"{shared} of {total} datasets appear in both groups; "
            "study batch cannot be separated from biology without overlap",
            status,
        )

    # --- donor structure --------------------------------------------------
    if metadata is not None and "donor_id" in metadata.columns:
        usable = _usable_donor_mask(metadata["donor_id"])
        unusable_frac = float(1.0 - usable.mean()) if len(usable) else 0.0
        record(
            "donor_usability", unusable_frac,
            f"{unusable_frac:.1%} of cells have an unusable donor_id; "
            "these cannot be grouped and must not be treated as one donor",
            "FAIL" if unusable_frac > DONOR_UNUSABLE_FAIL else "OK",
        )

        counts, dominance = {}, 0.0
        for name, mask in ((ref_name, is_ref), (alt_name, is_alt)):
            sel = mask & usable
            donors = metadata.loc[sel, "donor_id"]
            counts[name] = int(donors.nunique())
            if len(donors):
                dominance = max(dominance, float(donors.value_counts().iloc[0] / len(donors)))

        min_donors = min(counts.values()) if counts else 0
        record(
            "donor_count", min_donors,
            f"donors per group: " + ", ".join(f"{k}={v}" for k, v in counts.items())
            + "; cells are tested as independent replicates",
            "FAIL" if min_donors < MIN_DONORS_FAIL else "OK",
        )
        record(
            "donor_dominance", dominance,
            f"largest single donor contributes {dominance:.1%} of its group's cells",
            "FAIL" if dominance > DONOR_DOMINANCE_FAIL else "OK",
        )

    # --- sequencing depth (pre-normalization, informational after CP10K) --
    if lib_sizes is not None and is_ref.any() and is_alt.any():
        lib_sizes = np.asarray(lib_sizes, dtype=float)
        med_ref = float(np.median(lib_sizes[is_ref]))
        med_alt = float(np.median(lib_sizes[is_alt]))
        ratio = med_alt / med_ref if med_ref > 0 else float("nan")
        outside = np.isfinite(ratio) and not (
            DEPTH_RATIO_CAUTION_LOW <= ratio <= DEPTH_RATIO_CAUTION_HIGH
        )
        record(
            "depth_ratio", ratio,
            f"median library size {ref_name}={med_ref:,.0f} vs {alt_name}={med_alt:,.0f} "
            "before normalization; CP10K corrects this, the size of the correction is noted",
            "CAUTION" if outside else "OK",
        )

    verdict = "unreliable" if failed else ("caution" if caution else "ok")
    return {
        "verdict": verdict,
        "failed_checks": failed,
        "caution_checks": caution,
        "checks": checks,
        "groups": {
            "reference": {"name": ref_name, "n_cells": int(is_ref.sum())},
            "alternate": {"name": alt_name, "n_cells": int(is_alt.sum())},
        },
    }


def format_diagnostics_text(diagnostics: dict) -> str:
    """Render diagnostics as plain text for stdout, CSV headers, and plot captions."""
    if not diagnostics:
        return "COHORT DIAGNOSTICS: unavailable"
    verdict = diagnostics.get("verdict", "unknown").upper()
    groups = diagnostics.get("groups", {})
    ref = groups.get("reference", {})
    alt = groups.get("alternate", {})
    lines = [
        f"COHORT DIAGNOSTICS: {verdict}",
        f"  contrast: {ref.get('name')} (n={ref.get('n_cells')}) "
        f"vs {alt.get('name')} (n={alt.get('n_cells')})",
    ]
    for item in diagnostics.get("failed_checks", []):
        lines.append(f"  FAIL    {item['check']}: {item['detail']}")
    for item in diagnostics.get("caution_checks", []):
        lines.append(f"  CAUTION {item['check']}: {item['detail']}")
    if verdict != "OK":
        lines.append(
            "  Cells are treated as independent replicates; FDR is anti-conservative."
        )
    return "\n".join(lines)
