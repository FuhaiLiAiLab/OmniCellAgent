"""Select disease/control metacells while maximizing known-donor coverage."""

from collections import deque

import numpy as np
import pandas as pd

from tools.omic_tools.cohort_diagnostics import _usable_donor_mask, donor_key_columns, donor_keys


def _sample_across_donors(metadata, take, random_state):
    """Visit shuffled donors in rounds, selecting one unused metacell at a time."""
    rng = np.random.default_rng(random_state)
    groups = list(metadata.groupby(donor_keys(metadata), sort=False).indices.values())
    queues = deque(iter(rng.permutation(groups[i])) for i in rng.permutation(len(groups)))
    positions = []
    while len(positions) < take:
        donor = queues.popleft()
        try:
            positions.append(int(next(donor)))
        except StopIteration:
            continue
        queues.append(donor)
    return metadata.iloc[positions].copy()


def sample_known_donor_cohort(metadata, conditions, field_alias, sample_size=1000,
                              random_state=42):
    """Independently cap each pool while covering the maximum number of donors.

    Return selected metadata and per-group counts. A pool smaller than the cap
    contributes every eligible metacell; the other pool retains its own cap.
    Take one metacell per donor before additional rounds. Donors with exhausted
    pools drop out of later rounds. If donors exceed the cap, randomly select
    as many distinct donors as there are slots, using one metacell from each.
    """
    if sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    disease_column = field_alias.get("disease", "disease_BMG_name")
    pointer_columns = ["matrix_file_path", "matrix_row_idx"]
    required = {"donor_id", disease_column, *pointer_columns}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Known-donor sampling requires metadata columns: {', '.join(sorted(missing))}")

    resolved = {field_alias.get(key, key): value for key, value in conditions.items()
                if value is not None}
    scoped = metadata
    for column, value in resolved.items():
        if column == disease_column:
            continue
        if column not in scoped.columns:
            raise ValueError(f"Column {column!r} not found in metadata")
        values = value if isinstance(value, (list, tuple, set)) else [value]
        if pd.api.types.is_string_dtype(scoped[column]):
            mask = scoped[column].astype("string").str.lower().isin(
                [str(item).lower() for item in values]
            )
        else:
            mask = scoped[column].isin(values)
        scoped = scoped.loc[mask]

    diseases = scoped[disease_column].astype("string").str.strip().str.lower()
    case_mask = diseases.notna() & ~diseases.isin(
        {"normal", "unknown", "unclassified", "unannotated", "unannoted", "none", "nan", "na", ""}
    )
    if disease_column in resolved:
        value = resolved[disease_column]
        values = value if isinstance(value, (list, tuple, set)) else [value]
        case_mask &= diseases.isin([str(item).strip().lower() for item in values])
    pools = {
        "disease": scoped.loc[case_mask],
        "control": scoped.loc[diseases.eq("normal").fillna(False)],
    }

    selected, summary = [], {
        "requested_per_group": sample_size,
        "random_state": random_state,
        "strategy": "maximize_donors_then_round_robin",
        "donor_key_columns": donor_key_columns(metadata),
    }
    empty_groups = []
    for name, pool in pools.items():
        usable = _usable_donor_mask(pool["donor_id"])
        known = pool.loc[usable]
        unique = known.drop_duplicates(pointer_columns)
        take = min(sample_size, len(unique))
        chosen = (unique.copy() if take == len(unique)
                  else _sample_across_donors(unique, take, random_state))
        available_donors = int(donor_keys(unique).nunique())
        selected_donors = int(donor_keys(chosen).nunique())
        summary[name] = {
            "matching_rows": len(pool),
            "excluded_unknown_donor": int((~usable).sum()),
            "duplicate_rows_removed": len(known) - len(unique),
            "known_donor_available": len(unique),
            "selected": len(chosen),
            "donors_available": available_donors,
            "donors_selected": selected_donors,
            "unique_donor_ids_available": int(unique.donor_id.astype(str).str.strip().nunique()),
            "unique_donor_ids_selected": int(chosen.donor_id.astype(str).str.strip().nunique()),
        }
        action = "using all available" if take == len(unique) else "donor-balanced without replacement"
        print(
            f"[Donor sampling] {name}: matching={len(pool):,}; "
            f"excluded unknown/unusable donor={int((~usable).sum()):,}; "
            f"known-donor available={len(unique):,}; requested={sample_size:,}; "
            f"selected={take:,} ({action}); "
            f"donors selected={selected_donors:,}/{available_donors:,}; "
            f"duplicate rows removed={len(known) - len(unique):,}."
        )
        if not take:
            empty_groups.append(name)
        selected.append(chosen)

    if empty_groups:
        raise ValueError(f"No known-donor metacells available for: {', '.join(empty_groups)}")
    return pd.concat(selected, ignore_index=True), summary
