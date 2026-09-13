"""Exact cohort selection and explicit donor covariate conflict handling."""

import json

import numpy as np
import pandas as pd

IDENTITY_COLUMNS = ["source", "dataset_id", "donor_id"]
POINTER_COLUMNS = ["matrix_file_path", "matrix_row_idx"]
MISSING_VALUES = {"", "nan", "none", "unknown", "na", "n/a", "null", "<na>", "unknow"}


def usable_values(series):
    return series.notna() & ~series.astype(str).str.strip().str.lower().isin(MISSING_VALUES)


def select_cohort(metadata, disease, cell_type, organ, tissue=None):
    """Return all matching known-donor metacells plus case/control selection counts.

    Matches are exact apart from case. Both groups share organ/cell/tissue
    constraints. Duplicate or ambiguous source pointers cannot be counted twice.
    """
    required = set(IDENTITY_COLUMNS + POINTER_COLUMNS + ["disease_BMG_name", "CMT_name", "tissue_general"])
    if required - set(metadata.columns):
        raise ValueError(f"Missing cohort columns: {sorted(required - set(metadata.columns))}")
    repeated = metadata.loc[metadata.duplicated(POINTER_COLUMNS, keep=False)]
    for pointer, group in repeated.groupby(POINTER_COLUMNS, sort=False):
        if len(group.drop_duplicates()) > 1:
            raise ValueError(f"Metacell pointer {pointer} has conflicting annotations; refusing to keep only the first row")
    scoped = metadata.copy()
    for column, query in [("CMT_name", cell_type), ("tissue_general", organ), ("tissue", tissue)]:
        if query is not None:
            if column not in scoped:
                raise ValueError(f"Missing query column: {column}")
            scoped = scoped.loc[scoped[column].astype("string").str.lower().eq(str(query).lower()).fillna(False)]
    audit = {"query": {"disease": disease, "cell_type": cell_type, "organ": organ, "tissue": tissue},
             "input_metacells": len(metadata), "selection": "all_eligible_known_donor_metacells"}
    parts = []
    for group, target in [("disease", disease), ("control", "normal")]:
        pool = scoped.loc[scoped.disease_BMG_name.astype("string").str.lower().eq(str(target).lower()).fillna(False)]
        known = usable_values(pool.donor_id)
        valid_context = usable_values(pool.source) & usable_values(pool.dataset_id)
        retained = pool.loc[known & valid_context].drop_duplicates(POINTER_COLUMNS).copy()
        audit[group] = {"matching_metacells": len(pool), "unknown_donor_excluded": int((~known).sum()),
                        "missing_identity_context_excluded": int((known & ~valid_context).sum()),
                        "duplicate_pointers_removed": int((known & valid_context).sum()) - len(retained),
                        "selected_metacells": len(retained),
                        "donor_keys": len(retained.drop_duplicates(IDENTITY_COLUMNS))}
        if retained.empty:
            raise ValueError(f"No eligible known-donor {group} metacells for exact query {audit['query']}; counts={audit[group]}")
        parts.append(retained)
    selected = pd.concat(parts, ignore_index=True)
    if selected.duplicated(POINTER_COLUMNS).any():
        raise ValueError("The disease and control pools contain the same metacell pointer")
    return selected, audit


def _observations(group, column):
    if column not in group:
        return []
    return sorted({str(value).strip() for value in group[column].dropna()})


def apply_donor_exclusions(metadata, exclusions):
    """Apply an explicit reviewed key list; exclude whole donors, never round counts."""
    required = IDENTITY_COLUMNS + ["reason"]
    if set(required) - set(exclusions.columns):
        raise ValueError("Donor exclusions require source, dataset_id, donor_id and reason")
    table = exclusions[required].copy()
    for column in required:
        if not usable_values(table[column]).all():
            raise ValueError(f"Exclusion {column} values must be explicit and nonempty")
        table[column] = table[column].astype(str).str.strip()
    if table.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError("Exclusion file has duplicate donor keys")
    data = metadata.copy()
    for column in IDENTITY_COLUMNS:
        data[column] = data[column].astype(str).str.strip()
    keys = list(data[IDENTITY_COLUMNS].itertuples(index=False, name=None))
    excluded_keys = list(table[IDENTITY_COLUMNS].itertuples(index=False, name=None))
    unmatched = set(excluded_keys) - set(keys)
    if unmatched:
        raise ValueError(f"Exclusion donor keys are not present in this cohort: {sorted(unmatched)}")
    reasons = dict(zip(excluded_keys, table.reason))
    mask = pd.Series([key in reasons for key in keys], index=data.index)
    removed = data.loc[mask].copy()
    removed["exclusion_reason"] = [reasons[key] for key in pd.DataFrame(removed[IDENTITY_COLUMNS]).itertuples(index=False, name=None)]
    kept = data.loc[~mask].reset_index(drop=True)
    report = {"policy": "explicit_reviewed_donor_key_list", "excluded_donor_keys": len(table),
              "excluded_metacells": len(removed), "remaining_metacells": len(kept),
              "entries": table.to_dict("records")}
    return kept, removed.reset_index(drop=True), report


def build_donor_metadata(metadata, disease_name="Alzheimer's Disease"):
    """Build donor rows, aligned metacell assignments, and an annotation-issue table.

    Identity is (source, dataset_id, donor_id). A study is (source, dataset_id).
    Conflicting donor attributes stay missing; no majority or mean is imputed.
    """
    if set(IDENTITY_COLUMNS + POINTER_COLUMNS + ["disease_BMG_name"]) - set(metadata.columns):
        raise ValueError("Donor metadata requires identity, disease and metacell pointer columns")
    data = metadata.copy().reset_index(drop=True)
    for column in IDENTITY_COLUMNS:
        if not usable_values(data[column]).all():
            raise ValueError(f"Unusable donor identity field: {column}")
        data[column] = data[column].astype(str).str.strip()
    if data.duplicated(POINTER_COLUMNS).any():
        raise ValueError("Duplicate metacell pointers cannot be aggregated twice")
    keys = list(data[IDENTITY_COLUMNS].itertuples(index=False, name=None))
    key_ids = {key: f"PB{i:06d}" for i, key in enumerate(sorted(set(keys)), start=1)}
    data["sample_id"] = [key_ids[key] for key in keys]
    records, issues = [], []
    for sample_id, group in data.groupby("sample_id", sort=True):
        record = {"sample_id": sample_id, **{c: group[c].iloc[0] for c in IDENTITY_COLUMNS},
                  "n_metacells": len(group)}
        record["study"] = json.dumps([record["source"], record["dataset_id"]], separators=(",", ":"))
        raw = {
            "sex": _observations(group, "sex_normalized"),
            "age": _observations(group, "development_stage_numeric_age"),
            "disease": _observations(group, "disease_BMG_name"),
        }
        valid = {
            "sex": sorted({value.lower() for value in raw["sex"] if value.lower() in {"female", "male"}}),
            "age": sorted({float(value) for value in pd.to_numeric(
                pd.Series(raw["age"], dtype=object), errors="coerce").dropna() if np.isfinite(value)}),
            "disease": sorted({"AD" if value.lower() == disease_name.lower() else "control"
                               for value in raw["disease"]
                               if value.lower() in {disease_name.lower(), "normal"}}),
        }
        for field in ["sex", "age", "disease"]:
            values = valid[field]
            conflict = len(values) > 1
            resolution = "conflict" if conflict else "missing" if not values else "unique_observed_value"
            record[field] = values[0] if len(values) == 1 else np.nan
            record[f"{field}_conflict"] = conflict
            record[f"{field}_values"] = json.dumps(raw[field], ensure_ascii=False)
            if resolution != "unique_observed_value":
                issues.append({"sample_id": sample_id, **{c: record[c] for c in IDENTITY_COLUMNS},
                               "field": field, "observed_values": json.dumps(raw[field], ensure_ascii=False),
                               "resolution": resolution, "n_metacells": len(group)})
        records.append(record)
    donors = pd.DataFrame(records)
    assignments = data[["sample_id"] + IDENTITY_COLUMNS + POINTER_COLUMNS].copy()
    assignments.insert(0, "metacell_index", np.arange(len(assignments)))
    issue_columns = ["sample_id"] + IDENTITY_COLUMNS + ["field", "observed_values", "resolution", "n_metacells"]
    return donors, assignments, pd.DataFrame(issues, columns=issue_columns)
