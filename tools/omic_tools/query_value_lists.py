"""Write the full CellTOSG disease and cell-type vocabularies per session."""

import json
from pathlib import Path


def save_query_value_lists(metadata, output_dir, *, disease=None, cell_type=None):
    """Save exact source values with the query's vocabulary match on line one.

    A vocabulary match does not imply that the combined query has eligible
    samples. Lists cover the full dataset, before cohort or donor filtering.
    """
    fields = [
        ("disease", "disease_BMG_name", "available_diseases.txt", disease),
        ("cell type", "CMT_name", "available_cell_types.txt", cell_type),
    ]
    missing = {column for _, column, _, _ in fields} - set(metadata.columns)
    if missing:
        raise ValueError(f"Query value lists require metadata columns: {', '.join(sorted(missing))}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for label, column, filename, requested in fields:
        values = sorted(
            {str(value) for value in metadata[column].dropna().unique() if str(value).strip()},
            key=lambda value: (value.lower(), value),
        )
        if requested:
            # Disease sampling strips surrounding whitespace; cell-type
            # equality follows CellTOSG's case-insensitive exact lookup.
            normalize = (lambda value: value.strip().lower()) if label == "disease" else str.lower
            matches = [value for value in values if normalize(value) == normalize(str(requested))]
            matched = ", ".join(json.dumps(value, ensure_ascii=False) for value in matches)
            if not matched:
                matched = f"NONE (requested: {json.dumps(str(requested), ensure_ascii=False)})"
        else:
            matched = "NONE (no filter requested)"
        path = output_dir / filename
        path.write_text(
            f"Matched {label} in list: {matched}\n" + "".join(f"{value}\n" for value in values),
            encoding="utf-8",
        )
        paths[column] = str(path)
        print(f"[Query values] Saved {len(values):,} unique {label} values to {path}")
    return paths
