"""Read the seven-column DE result contract exported by the R pipeline."""

from os import PathLike

import pandas as pd


DE_RESULT_DTYPES = {
    "Name": "object",
    "log2_fold_change": "float64",
    "effect_size": "float64",
    "p_value": "float64",
    "FDR": "float64",
    "is_significant": "bool",
    "abs_log2_fc": "float64",
}


def read_de_results(path: str | PathLike[str]) -> pd.DataFrame:
    """Read full or significant-gene results with the same seven column types.

    Header-only CSVs carry no dtype information. Specify the schema at the
    read boundary so zero significant genes still produces float64/bool
    columns. This also handles the R export's TRUE/FALSE values and optional
    UTF-8 diagnostic comments without rewriting the source file.
    """
    return pd.read_csv(
        path,
        comment="#",
        encoding="utf-8",
        dtype=DE_RESULT_DTYPES,
    )
