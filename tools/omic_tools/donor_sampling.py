"""Select disease/control metacells while maximizing known-donor coverage."""

from collections import deque

import numpy as np
import pandas as pd

from tools.omic_tools.cohort_diagnostics import _usable_donor_mask, donor_key_columns, donor_keys

# Maximum metacells per disease/control group; the workflow and the cell-type lookup use it too
DEFAULT_SAMPLE_SIZE = 1000
# Fewest disease and fewest control metacells a comparison may use, counted after every
# selection rule (shared datasets included); the cell-type lookup lists cell types by it too
MIN_METACELLS_PER_GROUP = 10


def _sample_across_donors(metadata, take, random_state):
    """Visit shuffled donors in rounds, selecting one unused metacell at a time.

    Metacells without a usable donor ID share one pseudo-donor per source and study,
    because their donors cannot be told apart.
    """
    rng = np.random.default_rng(random_state)
    donor_id = metadata["donor_id"].where(_usable_donor_mask(metadata["donor_id"]), "")
    groups = list(metadata.groupby(donor_keys(metadata.assign(donor_id=donor_id)),
                                   sort=False).indices.values())
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


def _scope(metadata, resolved, disease_column):
    """Filter metadata by every resolved condition except the disease column."""
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
    return scoped


def _disease_mask(values, requested=None):
    """Rows whose disease names a disease and, if given, one of the requested ones.

    Matching strips surrounding whitespace and ignores case.
    """
    diseases = values.astype("string").str.strip().str.lower()
    mask = diseases.notna() & ~diseases.isin(
        {"normal", "unknown", "unclassified", "unannotated", "unannoted", "none", "nan", "na", ""}
    )
    if requested is not None:
        requested = requested if isinstance(requested, (list, tuple, set)) else [requested]
        mask &= diseases.isin([str(item).strip().lower() for item in requested])
    return mask


def _normal_mask(values):
    """Rows whose disease is "normal", ignoring case and surrounding whitespace."""
    return values.astype("string").str.strip().str.lower().eq("normal").fillna(False)


def sample_known_donor_cohort(metadata, conditions, field_alias, sample_size=DEFAULT_SAMPLE_SIZE,
                              random_state=42, keep_unknown_donor=False,
                              control_cell_type=None, disease_cell_label=None,
                              shared_datasets_only=False, min_metacells=MIN_METACELLS_PER_GROUP):
    """Independently cap each pool while covering the maximum number of donors.

    Return selected metadata and per-group counts. A pool smaller than the cap
    contributes every eligible metacell; the other pool retains its own cap.
    Take one metacell per donor before additional rounds. Donors with exhausted
    pools drop out of later rounds. If donors exceed the cap, randomly select
    as many distinct donors as there are slots, using one metacell from each.

    By default only metacells with a usable donor ID are eligible. With
    keep_unknown_donor=True, metacells without one are eligible too (some
    studies, e.g. the PDAC study GSE202051, record no donor IDs); they share
    one sampling slot per source and study, and summary donor counts cover
    usable donor IDs only.

    Both groups share the cell-type condition unless control_cell_type is
    given: then the control group uses that cell type instead (tumour cells
    have no normal counterpart with the same name). disease_cell_label also
    requires the disease cells' original source-study label (the metadata
    `cell_type` column) to equal it, ignoring case.

    shared_datasets_only keeps only the datasets (dataset_id) that contribute
    eligible metacells to both groups, before sampling. A dataset that holds
    one group only carries its study effects into that group alone, where no
    model can separate them from the disease effect.

    Raise ValueError if either group has fewer than min_metacells eligible
    metacells after these rules (duplicate matrix rows count once).
    """
    if sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    if min_metacells < 1:
        raise ValueError("min_metacells must be a positive integer")
    disease_column = field_alias.get("disease", "disease_BMG_name")
    pointer_columns = ["matrix_file_path", "matrix_row_idx"]
    required = {"donor_id", disease_column, *pointer_columns}
    if shared_datasets_only:
        required.add("dataset_id")
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Known-donor sampling requires metadata columns: {', '.join(sorted(missing))}")

    resolved = {field_alias.get(key, key): value for key, value in conditions.items()
                if value is not None}
    case_resolved, control_resolved = dict(resolved), dict(resolved)
    if control_cell_type is not None:
        control_resolved[field_alias.get("cell_type", "cell_type")] = control_cell_type
    if disease_cell_label is not None:
        # The source study's own label; the harmonized cell type is aliased to another column.
        case_resolved["cell_type"] = disease_cell_label
    scoped = _scope(metadata, case_resolved, disease_column)
    control_scoped = (scoped if control_resolved == case_resolved
                      else _scope(metadata, control_resolved, disease_column))

    pools = {
        "disease": scoped.loc[_disease_mask(scoped[disease_column], resolved.get(disease_column))],
        "control": control_scoped.loc[_normal_mask(control_scoped[disease_column])],
    }
    usable = {name: _usable_donor_mask(pool["donor_id"]) for name, pool in pools.items()}
    eligible = {name: pool if keep_unknown_donor else pool.loc[usable[name]]
                for name, pool in pools.items()}

    shared = None
    if shared_datasets_only:
        shared = sorted(set(eligible["disease"]["dataset_id"]) & set(eligible["control"]["dataset_id"]))
        if not shared:
            raise ValueError("No dataset contributes eligible metacells to both the disease and "
                             "the control group, so no within-dataset comparison is possible")
        print(f"[Donor sampling] shared datasets only: {len(shared)} datasets hold both groups")

    selected, summary = [], {
        "requested_per_group": sample_size,
        "random_state": random_state,
        "strategy": "maximize_donors_then_round_robin",
        "donor_key_columns": donor_key_columns(metadata),
        "keep_unknown_donor": bool(keep_unknown_donor),
        "control_cell_type": control_cell_type,
        "disease_cell_label": disease_cell_label,
        "shared_datasets_only": bool(shared_datasets_only),
        "shared_datasets": shared,
        "min_metacells_per_group": min_metacells,
    }
    empty_groups, small_groups = [], []
    for name, pool in pools.items():
        rows = eligible[name]
        excluded_not_shared = 0
        if shared is not None:
            in_shared = rows["dataset_id"].isin(shared)
            excluded_not_shared = int((~in_shared).sum())
            rows = rows.loc[in_shared]
        unique = rows.drop_duplicates(pointer_columns)
        take = min(sample_size, len(unique))
        chosen = (unique.copy() if take == len(unique)
                  else _sample_across_donors(unique, take, random_state))
        unique_known = unique.loc[_usable_donor_mask(unique["donor_id"])]
        chosen_known = chosen.loc[_usable_donor_mask(chosen["donor_id"])]
        available_donors = int(donor_keys(unique_known).nunique())
        selected_donors = int(donor_keys(chosen_known).nunique())
        summary[name] = {
            "matching_rows": len(pool),
            "excluded_unknown_donor": 0 if keep_unknown_donor else int((~usable[name]).sum()),
            "excluded_not_in_shared_dataset": excluded_not_shared,
            "included_unknown_donor": len(chosen) - len(chosen_known),
            "duplicate_rows_removed": len(rows) - len(unique),
            "known_donor_available": len(unique_known),
            "selected": len(chosen),
            "donors_available": available_donors,
            "donors_selected": selected_donors,
            "unique_donor_ids_available": int(unique_known.donor_id.astype(str).str.strip().nunique()),
            "unique_donor_ids_selected": int(chosen_known.donor_id.astype(str).str.strip().nunique()),
        }
        action = "using all available" if take == len(unique) else "donor-balanced without replacement"
        print(
            f"[Donor sampling] {name}: matching={len(pool):,}; "
            f"{'kept' if keep_unknown_donor else 'excluded'} unknown/unusable donor={int((~usable[name]).sum()):,}; "
            f"excluded outside shared datasets={excluded_not_shared:,}; "
            f"known-donor available={len(unique_known):,}; requested={sample_size:,}; "
            f"selected={take:,} ({action}); "
            f"donors selected={selected_donors:,}/{available_donors:,}; "
            f"duplicate rows removed={len(rows) - len(unique):,}."
        )
        if not take:
            empty_groups.append(name)
        elif len(unique) < min_metacells:
            small_groups.append(f"{name} ({len(unique):,})")
        selected.append(chosen)

    kind = "metacells" if keep_unknown_donor else "known-donor metacells"
    if empty_groups:
        raise ValueError(f"No {kind} available for: {', '.join(empty_groups)}")
    if small_groups:
        raise ValueError(f"Fewer than {min_metacells:,} {kind} available for: {', '.join(small_groups)}")
    return pd.concat(selected, ignore_index=True), summary
