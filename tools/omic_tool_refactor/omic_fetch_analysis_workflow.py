"""
Omic Fetch and Analysis Workflow

DATA TYPE: This workflow operates on a COHORT of SINGLE-CELL RNA-seq (scRNA-seq)
data drawn from the OmniCellTOSG database. The expression matrix is per-cell
(samples = cells), NOT bulk RNA-seq. Downstream differential expression compares
disease and normal metacells.

This workflow performs:
1. Direct query construction from disease, cell type and organ
2. scRNA-seq retrieval using CellTOSGDataLoader with an
   HGNC-approved protein-coding gene mapping
3. Differential expression analysis between disease and normal metacells
4. KEGG pathway enrichment analysis
5. Enrichment plotting (via R script)
6. Returns top genes and results for agent reflection

Based on the working CellTOSGDataLoader example pattern.
"""

from omic_analysis_components import omic_analysis, EnrichrError, summarize_enrichment
from cohort_diagnostics import compute_cohort_diagnostics, format_diagnostics_text
from celltosg_runtime_adapter import writable_celltosg_root
from donor_sampling import DEFAULT_SAMPLE_SIZE, disease_key, sample_known_donor_cohort
from query_value_lists import save_query_value_lists
from subprocess_r import run_r_script
from de_results_io import read_de_results
from r_plotting import render_enrichment_plots, collect_plot_outputs

import sys
import os
import time
import gc
import json
import numpy as np
import pandas as pd

import settings

# Add the OmniCellTOSG directory to path to import the new loader
omnicell_root = settings.OMNICELL_ROOT
sys.path.insert(0, omnicell_root)
from CellTOSG_Loader import CellTOSGDataLoader, CellTOSGSubsetBuilder


def _build_labels_from_metadata(metadata, label_col: str, field_alias: dict,
                                priority_labels: set):
    """
    Rebuild a Y vector from a metadata column.

    Returns (Y, mapping, counts, valid_mask) or None if the column has fewer
    than two non-empty classes.
    """
    resolved = field_alias.get(label_col, label_col)
    if metadata is None or resolved not in metadata.columns:
        return None

    series = metadata[resolved]
    if label_col == "disease":
        # One class per disease as the sampler matches it, named by its most common
        # spelling: v3 holds both "Alzheimer's Disease" and "Alzheimer disease".
        keys = disease_key(series)
        spelling = series.groupby(keys).agg(lambda names: names.value_counts().index[0])
        series = keys.map(spelling).where(series.notna())
    unique_vals = series.dropna().unique().tolist()
    if len(unique_vals) < 2:
        return None

    priority_lower = {p.lower() for p in (priority_labels or set())}
    if any(str(v).lower() in priority_lower for v in unique_vals):
        sorted_labels = sorted(
            set(unique_vals),
            key=lambda x: (str(x).lower() not in priority_lower, str(x).lower()),
        )
    else:
        sorted_labels = sorted(set(unique_vals), key=lambda x: str(x).lower())

    # Match CellTOSG_Loader.build_split_labels: ALL label-zero values collapse
    # into class 0. Giving each its own index produces an arbitrary contrast
    # downstream (e.g. normal vs "Unclassified" with 10 cells).
    has_priority = any(str(v).lower() in priority_lower for v in unique_vals)
    if has_priority:
        mapping = {}
        next_idx = 1
        for label_name in sorted_labels:
            if str(label_name).lower() in priority_lower:
                mapping[label_name] = 0
            else:
                mapping[label_name] = next_idx
                next_idx += 1
    else:
        mapping = {label_name: idx for idx, label_name in enumerate(sorted_labels)}
    # Sentinel -1 for NaN/unknown so we can filter them out
    Y_mapped = series.map(lambda x: mapping.get(x, -1)).values.astype(int)
    valid_mask = Y_mapped != -1
    counts = {
        int(idx): int(np.sum((Y_mapped == idx) & valid_mask))
        for idx in sorted(set(mapping.values()))
    }
    nonempty = sum(1 for c in counts.values() if c > 0)
    if nonempty < 2:
        return None
    return Y_mapped, mapping, counts, valid_mask


def select_contrast(Y, mapping: dict):
    """Choose the two-group contrast for differential expression.

    Reference is class 0 (the collapsed label-zero group, e.g. "normal").
    The alternate is the LARGEST non-reference class, so a cohort carrying many
    rare labels yields the dominant comparison rather than an arbitrary one.

    Returns None when fewer than two non-empty classes exist.
    """
    Y = np.asarray(Y)
    counts = {int(c): int(np.sum(Y == c)) for c in np.unique(Y) if int(c) >= 0}
    if counts.get(0, 0) == 0:
        return None
    non_ref = {c: n for c, n in counts.items() if c != 0 and n > 0}
    if not non_ref:
        return None

    alt_class = max(non_ref, key=lambda c: (non_ref[c], -c))
    reverse = {}
    for name, idx in mapping.items():
        reverse.setdefault(int(idx), []).append(str(name))

    return {
        "ref_class": 0,
        "alt_class": int(alt_class),
        "ref_name": "/".join(sorted(reverse.get(0, ["class_0"]))),
        "alt_name": "/".join(sorted(reverse.get(int(alt_class), [f"class_{alt_class}"]))),
        "excluded": sorted(c for c in non_ref if c != alt_class),
        "counts": counts,
    }


def omic_fetch_with_new_loader(fetch_dict: dict, output_dir: str, label: str = "disease",
                               sample_size: int = DEFAULT_SAMPLE_SIZE):
    """
    Fetch single-cell RNA-seq data using the new CellTOSGDataLoader.

    The expression matrix returned is per-cell (scRNA-seq cohort). Y is a binary/
    categorical group assignment derived from `label`.

    Args:
        fetch_dict (dict): Dictionary containing extracted entities (cell type,
            disease, organ).
        output_dir (str): Directory to save output files.
        label (str): Column used to split samples for downstream differential
            expression: "disease" (disease vs normal), the only label here.
        sample_size (int): Maximum metacells per disease/control group. Each
            group maximizes donor coverage, then fills in rounds across donors;
            use all known-donor metacells if fewer than the limit are available.

    Both groups come only from datasets that hold eligible disease and control
    metacells (see donor_sampling.sample_known_donor_cohort). A comparison with
    fewer than donor_sampling.MIN_METACELLS_PER_GROUP eligible metacells in
    either group retrieves nothing.

    Returns:
        tuple: (X, Y, metadata, similar_terms, retrieval_success)
    """
    if sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    data_root = settings.OMNICELL_DATA_ROOT

    cell_type = fetch_dict.get("cell type", None)
    disease_name = fetch_dict.get("disease", None)
    organ = fetch_dict.get("organ", None)

    print(f"\n{'='*60}")
    print(f"[Omic Fetch] Starting scRNA-seq data retrieval")
    print(f"  Cell type: {cell_type}")
    print(f"  Disease: {disease_name}")
    print(f"  Organ: {organ}")
    print(f"  Label column: {label}")
    print(f"{'='*60}\n")

    conditions = {}
    if cell_type:
        conditions["cell_type"] = cell_type
    if disease_name:
        conditions["disease"] = disease_name
    if organ:
        conditions["tissue_general"] = organ

    # Disease comparisons select known-donor case and normal pools before
    # extraction.
    task = label
    label_column = task

    try:
        all_metadata = pd.read_parquet(os.path.join(data_root, "cell_metadata_with_mappings.parquet"))
        save_query_value_lists(
            all_metadata, output_dir, disease=disease_name, cell_type=cell_type,
        )
        sampled_metadata, sampling_summary = sample_known_donor_cohort(
            all_metadata,
            conditions,
            CellTOSGSubsetBuilder.FIELD_ALIAS,
            sample_size=sample_size,
            random_state=42,
        )
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "donor_sampling.json"), "w", encoding="utf-8") as handle:
            json.dump(sampling_summary, handle, indent=2)
        # The temporary metadata already contains precisely the selected
        # case AND control rows. Do not filter it by disease or resample it.
        loader_conditions = {}
        loader_sample_size = None
        del all_metadata

        print(f"[Omic Fetch] Creating CellTOSGDataLoader...")
        print(f"  root: {data_root}")
        print(f"  conditions: {conditions}")
        print(f"  task: {task}")
        print(f"  label_column: {label_column}")
        print("  stratified_balancing: False")
        sys.stdout.flush()

        with writable_celltosg_root(data_root, metadata=sampled_metadata) as loader_root:
            dataset = CellTOSGDataLoader(
                root=loader_root,
                conditions=loader_conditions,
                task=task,
                label_column=label_column,
                sample_ratio=None,
                sample_size=loader_sample_size,
                shuffle=False,
                stratified_balancing=False,
                extract_mode="inference",
                random_state=42,
                train_text=False,
                train_bio=False,
                correction_method=None,
                output_dir=output_dir
            )

        X = dataset.data
        Y = dataset.labels
        metadata = dataset.metadata

        print(f"[Omic Fetch] Data retrieved successfully")
        print(f"  Samples: {X.shape[0] if X is not None else 0}")
        print(f"  Features: {X.shape[1] if X is not None and len(X.shape) > 1 else 0}")
        sys.stdout.flush()

        if X is None or X.shape[0] == 0:
            print(f"\n[Omic Fetch] No samples matched the conditions")
            return None, None, None, {}, False, label, None, None

        # Capture loader constants before we delete `dataset` — needed to encode the label.
        field_alias = dict(dataset.query.FIELD_ALIAS)
        priority_by_task = dict(type(dataset).LABEL_ZERO_LABELS_BY_LABEL_COLUMN)

        # Extract similar terms from soft matching
        similar_terms = {}
        if hasattr(dataset, 'query') and hasattr(dataset.query, 'last_query_conditions_resolved'):
            resolved = dataset.query.last_query_conditions_resolved
            raw = dataset.query.last_query_conditions_raw
            reverse_alias = {v: k for k, v in dataset.query.FIELD_ALIAS.items()}
            for resolved_key, resolved_value in resolved.items():
                original_key = reverse_alias.get(resolved_key, resolved_key)
                original_value = None
                for raw_key, raw_value in raw.items():
                    if dataset.query.FIELD_ALIAS.get(raw_key, raw_key) == resolved_key:
                        original_value = raw_value
                        break
                if original_value and original_value != resolved_value:
                    similar_terms[original_key] = {
                        "original": original_value,
                        "matched": resolved_value
                    }

        if similar_terms:
            print(f"\n[Soft Matching] Terms adjusted:")
            for key, mapping in similar_terms.items():
                print(f"  {key}: '{mapping['original']}' -> '{mapping['matched']}'")

        del dataset
        gc.collect()

        actual_label = label
        fallback_message = None
        label_mapping = None

        def _count_groups(Y_arr):
            if Y_arr is None:
                return {}
            arr = np.asarray(Y_arr)
            try:
                int_arr = arr.astype(int)
            except (TypeError, ValueError):
                return {}
            return {int(c): int(np.sum(int_arr == c)) for c in np.unique(int_arr)}

        # The loader only builds a numeric Y when extract_mode="train"; under
        # "inference" it deliberately returns the metadata frame as `.labels`
        # (CellTOSG_Loader/dataset.py:112-116). Encode the *requested* label here
        # so DE gets a usable Y.
        if metadata is not None and not isinstance(Y, np.ndarray):
            encoded = _build_labels_from_metadata(
                metadata, label, field_alias, priority_by_task.get(label, set())
            )
            if encoded is None:
                Y = None
                print(f"[Label] Could not encode '{label}' from metadata (<2 non-empty classes)")
            elif X.shape[0] != len(encoded[3]):
                Y = None
                print(f"[Label] Row mismatch (X={X.shape[0]}, metadata={len(encoded[3])}); skipping encode")
            else:
                Y_enc, enc_mapping, enc_counts, enc_mask = encoded
                X = X[enc_mask]
                Y = Y_enc[enc_mask]
                metadata = metadata[enc_mask].reset_index(drop=True)
                label_mapping = enc_mapping
                print(f"[Label] Encoded '{label}' from metadata: {enc_mapping} -> counts {enc_counts}")

        current_counts = _count_groups(Y)
        print(f"[Label Check] label='{label}' class counts: {current_counts}")

        return X, Y, metadata, similar_terms, True, actual_label, fallback_message, label_mapping

    except ValueError as e:
        print(f"\n[Omic Fetch] No match: {str(e)}")
        return None, None, None, {}, False, label, None, None

    except Exception as e:
        print(f"\n[Omic Fetch] Unexpected issue: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, {}, False, label, None, None


def normalize_cp10k(X, target_sum: float = 1e4):
    """Scale each cell to `target_sum` total counts (counts per 10K).

    Depth is confounded with group membership in these cohorts, so every
    statistic downstream must run on normalized values. `log1p` is deliberately
    NOT applied: it does not change Mann-Whitney ranks, and feeding log-scale
    values into log2(mean_a / mean_b) silently turns the fold change into a
    ratio of log means. Staying linear keeps that formula correct.

    Returns the same container type it was given, so gene-symbol columns survive.
    """
    if X is None:
        return None
    is_frame = hasattr(X, "columns")
    values = (X.values if is_frame else np.asarray(X)).astype(np.float64)
    sums = values.sum(axis=1, keepdims=True)
    scaled = values * (target_sum / np.maximum(sums, 1e-12))
    if is_frame:
        return pd.DataFrame(scaled, index=X.index, columns=X.columns)

    return scaled


def compute_top_genes(X, top_k=100):
    """Compute top K genes by mean expression. Returns (indices, values)."""
    if X is None or X.shape[0] == 0:
        return [], []

    # Always work on a bare ndarray: np.mean() on a DataFrame returns a Series,
    # and indexing that Series with integer positions is deprecated in pandas.
    values = X.values if hasattr(X, "values") else np.asarray(X)
    mean_expr = np.mean(values, axis=0)
    top_k_indices = np.argsort(mean_expr)[-top_k:][::-1]
    top_k_values = mean_expr[top_k_indices]

    return top_k_indices.tolist(), top_k_values.tolist()


# Fixed number of top genes to return
TOP_K_GENES = 20

def omic_fetch_analysis_workflow(*, disease=None, cell_type=None, organ=None, session_dir=None,
                                 sample_size=DEFAULT_SAMPLE_SIZE,
                                 r_timeout=3600, enrichment_timeout=60):
    """
    Perform the complete single-cell omic analysis workflow.

    NOTE: This pipeline analyzes a COHORT of single-cell RNA-seq (scRNA-seq)
    data from OmniCellTOSG. Samples are metacells (SEACells aggregates of about
    200 cells of one cell type from one source file), not single cells, donors
    or bulk libraries.

    Query the cohort with direct, keyword-only parameters (disease, cell_type,
    organ). Disease metacells are compared with normal metacells of the same cell
    type and organ, only within datasets that hold both groups; either group
    having fewer than donor_sampling.MIN_METACELLS_PER_GROUP eligible metacells
    stops the run before any analysis. Differential expression and plotting
    always run.

    Args:
        disease, cell_type, organ: Direct query parameters
        session_dir: Directory to save all outputs (REQUIRED)
        r_timeout: Maximum duration in seconds for each R invocation.
        enrichment_timeout: Timeout for each Enrichr HTTP request.
        sample_size: Maximum metacells per disease/control group (default 1000).
            Maximize donor coverage, then fill in rounds. Use all known-donor
            metacells in a group if fewer than the limit are available.
    """
    times = {}
    times['start'] = time.time()
    if sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    # The agent tool's label for this cohort: disease vs normal metacells
    label = "disease"

    if session_dir is None:
        raise ValueError("session_dir must be provided")
    os.makedirs(session_dir, exist_ok=True)
    print(f"\n[Session] Output directory: {session_dir}")

    # ===========================================================================
    # STEP 1: Build query from direct parameters
    # ===========================================================================
    print(f"\n{'='*70}")
    print(f"STEP 1: Direct Parameter Input")
    print(f"{'='*70}")
    fetch_dict = {}
    if disease is not None:
        fetch_dict["disease"] = disease
    if cell_type is not None:
        fetch_dict["cell type"] = cell_type
    if organ is not None:
        fetch_dict["organ"] = organ

    print(f"[Input] Direct parameters: {fetch_dict}")
    times['input_end'] = time.time()

    # ===========================================================================
    # STEP 2: Data Retrieval
    # ===========================================================================
    print(f"\n{'='*70}")
    print(f"STEP 2: Data Retrieval with Soft Matching")
    print(f"{'='*70}")

    (X, Y, metadata, similar_terms, retrieval_success,
     actual_label, label_fallback_message, label_mapping) = omic_fetch_with_new_loader(
        fetch_dict, session_dir, label=label,
        sample_size=sample_size,
    )
    # The agent's result reports a control source only when controls are borrowed from other
    # studies (borrow_controls), which this cohort does not do; it reports None here
    control_source = None
    control_note = None
    # The temporary loader root filters the BMG mapping to HGNC-approved
    # protein-coding symbols before extraction, including the saved raw matrix.
    # Capture them now: np.nan_to_num() downstream returns a bare ndarray and
    # destroys column labels.
    gene_names = list(X.columns) if hasattr(X, "columns") else None

    # Record pre-normalization depth for the cohort diagnostics, then normalize.
    lib_sizes = None
    if X is not None:
        raw_values = X.values if hasattr(X, "values") else np.asarray(X)
        lib_sizes = np.asarray(raw_values).sum(axis=1).astype(np.float64)
        X = normalize_cp10k(X)
        print(f"[Normalize] CP10K applied; median library size before = {np.median(lib_sizes):,.0f}")
    times['fetch_end'] = time.time()

    if not retrieval_success or X is None:
        print(f"\n{'='*70}")
        print(f"WORKFLOW: No matching data found")
        print(f"{'='*70}")

        return {
            "success": False,
            "message": "No samples matched the specified conditions.",
            "data_type": "single-cell RNA-seq (scRNA-seq) cohort",
            "label_column": label,
            "requested_label": label,
            "actual_label": actual_label,
            "label_fallback_message": label_fallback_message,
            "similar_terms": similar_terms,
            "extracted_entities": fetch_dict,
            "retrieval_success": False,
            "timing": {
                "input": times['input_end'] - times['start'],
                "fetch": times['fetch_end'] - times['input_end'],
                "total": times['fetch_end'] - times['start']
            }
        }

    # ===========================================================================
    # STEP 3: Abundance QC — most-expressed genes
    # ===========================================================================
    # This is a data-quality check, NOT a discovery step. The list contains
    # only retained protein-coding genes, often including housekeeping genes.
    # Tissue markers appearing here (SFTPB/SFTPC in lung, SPP1/CD74 in microglia)
    # confirm the intended cell population was retrieved.
    print(f"\n{'='*70}")
    print(f"STEP 3: Abundance QC — Top {TOP_K_GENES} Genes by Mean Expression (CP10K)")
    print(f"{'='*70}")

    top_gene_indices, top_gene_values = compute_top_genes(X, TOP_K_GENES)
    top_gene_names = (
        [gene_names[i] for i in top_gene_indices] if gene_names else []
    )
    print(f"[QC] Top {len(top_gene_indices)} genes computed")
    if top_gene_names:
        print(f"  Top 5 genes: {top_gene_names[:5]}")
        print(f"  Top 5 mean values (CP10K): {[f'{v:.4f}' for v in top_gene_values[:5]]}")
        print("  NOTE: abundance reflects data characteristics, not disease signal.")

    times['gene_end'] = time.time()

    if len(top_gene_indices) > 0:
        top_genes_df = pd.DataFrame({
            "rank": range(1, len(top_gene_indices) + 1),
            "gene_index": top_gene_indices,
            "gene_name": top_gene_names or [""] * len(top_gene_indices),
            "mean_expression_cp10k": top_gene_values,
        })
        top_genes_path = os.path.join(session_dir, "top_genes_by_expression.csv")
        top_genes_df.to_csv(top_genes_path, index=False)
        print(f"[QC] Saved to: {top_genes_path}")

    # ===========================================================================
    # STEP 4: Differential Expression Analysis
    # ===========================================================================
    disease_name = fetch_dict.get("disease", None)
    analysis_success = False
    de_success = False
    enrichment_success = False
    enrichment_status = "not_run"
    enrichment_error = None
    de_plots_status = "not_run"
    de_plots_error = None
    de_plot_files = []
    analysis_paths = None
    top_genes_by_fdr = []
    enrichment_top_terms = {}
    cohort_diagnostics = None
    diagnostics_text = ""

    # Build a comparison name used for output directory / plot titles.
    comparison_name = disease_name or "comparison"

    # DE runs whenever two non-empty classes exist. The old disease-specific
    # gate is superseded: label collapsing now guarantees a meaningful class 0,
    # and select_contrast names exactly what is being compared.
    contrast = None
    if Y is not None:
        contrast = select_contrast(Y, label_mapping or {})
    de_gated = contrast is not None

    if de_gated:
        print(f"\n{'='*70}")
        print(f"STEP 4: Differential Expression Analysis (label='{actual_label}')")
        print(f"{'='*70}")

        unique_labels = np.unique(Y)

        # Count samples per label index (avoid shadowing the outer `label` arg).
        label_counts = {int(lbl): int(np.sum(Y == lbl)) for lbl in unique_labels}
        print(f"[DE] Label distribution: {label_counts}")

        # For DE analysis, we need at least 2 groups with samples
        if len(unique_labels) >= 2 and all(label_counts.get(l, 0) > 0 for l in unique_labels[:2]):
            # Get the two groups based on labels
            group0_count = int(np.sum(Y == contrast["ref_class"]))
            group1_count = int(np.sum(Y == contrast["alt_class"]))
            print(f"[DE] Contrast: '{contrast['ref_name']}' (n={group0_count}) "
                  f"vs '{contrast['alt_name']}' (n={group1_count})")
            if contrast["excluded"]:
                print(f"[DE] Excluded {len(contrast['excluded'])} other class(es) "
                      f"from this contrast: {contrast['excluded']}")

            if group0_count > 0 and group1_count > 0:
                # Split data into two groups. Group 0 is the reference class
                # ("normal" — see CellTOSGDataLoader.LABEL_ZERO_LABELS_BY_LABEL_COLUMN).
                normal_omic_feature = X[Y == contrast["ref_class"]]
                disease_omic_feature = X[Y == contrast["alt_class"]]

                data_dict = {
                    "normal_omic_feature": normal_omic_feature,
                    "disease_omic_feature": disease_omic_feature,
                    "omic_label": Y,
                    # chrY expression covariate for sex imbalance (off only for a sex contrast)
                    "adjust_chrY": True,
                }
                # Source dataset per row, so limma can model the study effect.
                if (metadata is not None
                        and "dataset_id" in metadata.columns and len(metadata) == len(Y)):
                    dataset_ids = metadata["dataset_id"].astype(str).to_numpy()
                    data_dict["normal_dataset_id"] = dataset_ids[Y == contrast["ref_class"]]
                    data_dict["disease_dataset_id"] = dataset_ids[Y == contrast["alt_class"]]

                # Diagnostics are advisory only: the gate warns, it never
                # withholds. Any failure here (bad metadata shape, a
                # read-only/full session_dir on the json.dump, a future bug
                # in compute_cohort_diagnostics) must not prevent DE from
                # running, so it gets its own non-fatal try/except separate
                # from the omic_analysis try/except below.
                try:
                    cohort_diagnostics = compute_cohort_diagnostics(
                        metadata,
                        Y == contrast["ref_class"],
                        Y == contrast["alt_class"],
                        lib_sizes=lib_sizes,
                        ref_name=contrast["ref_name"],
                        alt_name=contrast["alt_name"],
                    )
                    diagnostics_text = format_diagnostics_text(cohort_diagnostics)
                    print(f"\n{diagnostics_text}\n")

                    with open(os.path.join(session_dir, "cohort_diagnostics.json"), "w") as handle:
                        json.dump(cohort_diagnostics, handle, indent=2, default=str)
                except Exception as e:
                    print(f"[DE] WARNING: cohort diagnostics failed ({e}); proceeding without them")
                    cohort_diagnostics = None
                    diagnostics_text = ""

                try:
                    data_and_analysis_dict = omic_analysis(
                        comparison_name,
                        data_dict,
                        session_dir=session_dir,
                        gene_names=gene_names,
                        diagnostics_text=diagnostics_text,
                        ref_label=contrast["ref_name"],
                        alt_label=contrast["alt_name"],
                        input_scale="linear_cp10k",
                        r_timeout=r_timeout,
                        enrichment_timeout=enrichment_timeout,
                    )
                    analysis_success = True
                    de_success = True
                    enrichment_success = data_and_analysis_dict["enrichment_success"]
                    enrichment_status = data_and_analysis_dict["enrichment_status"]
                    de_plots_status = data_and_analysis_dict["de_plots_status"]
                    de_plots_error = data_and_analysis_dict["de_plots_error"]
                    de_plot_files = data_and_analysis_dict["de_plot_files"]
                    analysis_paths = {
                        "differential_expression_dir": data_and_analysis_dict.get('differential_expression_dir'),
                        "enrichment_results_dir": data_and_analysis_dict.get('enrichment_results_dir'),
                    }

                    # Extract top genes by FDR with full statistics
                    gene_file = os.path.join(
                        data_and_analysis_dict.get('differential_expression_dir', ''),
                        "significant_genes_by_fdr.csv"
                    )
                    if os.path.exists(gene_file):
                        gene_df = read_de_results(gene_file)
                        # Store full statistics for top genes
                        top_genes_by_fdr = []
                        for idx, row in gene_df.head(TOP_K_GENES).iterrows():
                            top_genes_by_fdr.append({
                                "rank": idx + 1,
                                "gene_name": str(row['Name']),
                                "log2_fold_change": float(row['log2_fold_change']) if 'log2_fold_change' in row else None,
                                "FDR": float(row['FDR']) if 'FDR' in row else None,
                                "p_value": float(row['p_value']) if 'p_value' in row else None,
                                "effect_size": float(row['effect_size']) if 'effect_size' in row else None,
                            })
                        print(f"[DE] Top {len(top_genes_by_fdr)} genes by FDR: {[g['gene_name'] for g in top_genes_by_fdr[:5]]}...")

                    # Significant Enrichr terms of the up and down lists, so the agents report the
                    # computed enrichment instead of annotating single genes
                    if enrichment_success and analysis_paths.get("enrichment_results_dir"):
                        enrichment_top_terms = summarize_enrichment(analysis_paths["enrichment_results_dir"])

                    del data_and_analysis_dict

                except EnrichrError as e:
                    analysis_success = False
                    de_success = True
                    enrichment_success = False
                    enrichment_status = "failed"
                    enrichment_error = str(e)
                    analysis_paths = {"differential_expression_dir": os.path.join(session_dir, "differential_expression"),
                                      "enrichment_results_dir": os.path.join(session_dir, "enrichment_results")}
                    print(f"[Enrichment] Analysis failed: {e}")
                except Exception as e:
                    analysis_success = False
                    analysis_paths = None
                    print(f"[DE] Analysis failed: {str(e)}")
                    import traceback
                    traceback.print_exc()

                del normal_omic_feature, disease_omic_feature, data_dict
            else:
                print(f"[DE] Skipped: need samples in both groups for comparison")
                print(f"     NOTE: label='{actual_label}' — DE requires two non-empty classes")
                print(f"           in this column within the queried subset.")
        else:
            print(f"[DE] Skipped: need at least 2 groups with samples")
            print(f"     Labels found: {unique_labels}")
            print(f"     NOTE: With label='{actual_label}', all returned cells share the same value.")
    else:
        print(f"\n{'='*70}")
        print(f"STEP 4: Differential Expression Analysis - SKIPPED")
        print(f"{'='*70}")
        if Y is not None and contrast is None:
            print("  Reason: fewer than two non-empty classes after label collapsing")
        elif Y is None:
            print("  Reason: no labels available")

    times['de_end'] = time.time()

    # ===========================================================================
    # STEP 5: KEGG Enrichment Plotting (via R script)
    # ===========================================================================
    kegg_success = False
    enrichment_plot_status = "not_run"
    enrichment_plot_error = None
    current_enrichment_plot_files = []
    if enrichment_status in ("empty", "skipped"):
        enrichment_plot_status = "empty"

    if analysis_success and enrichment_status == "success" and analysis_paths:
        print(f"\n{'='*70}")
        print(f"STEP 5: KEGG Pathway Enrichment Plotting")
        print(f"{'='*70}")

        try:
            enrichment_dir = analysis_paths.get('enrichment_results_dir', '')
            manifest = render_enrichment_plots(
                settings.KEGG_SCRIPT, session_dir,
                enrichment_dir, comparison_name, timeout=r_timeout, runner=run_r_script)
            current_enrichment_plot_files = manifest["files"]
            enrichment_plot_status = manifest["status"]
            kegg_success = enrichment_plot_status == "success"
        except Exception as e:
            enrichment_plot_status = "failed"
            enrichment_plot_error = f"{type(e).__name__}: {e}"
            current_enrichment_plot_files = []
            print(f"[KEGG] Plotting failed: {str(e)}")
    else:
        print(f"\n{'='*70}")
        print(f"STEP 5: KEGG Pathway Enrichment Plotting - SKIPPED")
        print(f"{'='*70}")
        if not de_success:
            print("  Reason: differential expression analysis did not complete")
        elif enrichment_status != "success":
            print(f"  Reason: enrichment status is {enrichment_status}")

    times['kegg_end'] = time.time()

    # ===========================================================================
    # Cleanup and return results
    # ===========================================================================
    # Capture the feature width before releasing X — the success return dict
    # below needs it for num_features, and X (like data_and_analysis_dict and
    # normal_omic_feature/disease_omic_feature/data_dict above) is freed via
    # `del` immediately after its last use rather than left for the frame to
    # release on return.
    n_features = int(X.shape[1]) if X is not None and len(X.shape) > 1 else 0
    del X
    if Y is not None:
        del Y
    gc.collect()

    times['end'] = time.time()

    # Print timing summary
    print(f"\n{'='*70}")
    print(f"WORKFLOW TIMING SUMMARY")
    print(f"{'='*70}")
    print(f"Step 1 - Direct Input:     {times['input_end'] - times['start']:.2f}s")
    print(f"Step 2 - Data Fetch:       {times['fetch_end'] - times['input_end']:.2f}s")
    print(f"Step 3 - Gene Analysis:    {times['gene_end'] - times['fetch_end']:.2f}s")
    print(f"Step 4 - DE Analysis:      {times['de_end'] - times['gene_end']:.2f}s")
    print(f"Step 5 - KEGG Plotting:    {times['kegg_end'] - times['de_end']:.2f}s")
    print(f"{'='*70}")
    print(f"Total Duration:            {times['end'] - times['start']:.2f}s")
    print(f"{'='*70}\n")

    # ===========================================================================
    # Collect all plot paths for UI display and report embedding
    # Returns both absolute paths and relative paths for markdown embedding
    # ===========================================================================
    plot_paths, plots_for_report = collect_plot_outputs(session_dir, de_plot_files, current_enrichment_plot_files)

    # Print summary
    print(f"[Plots] Collected plots for report:")
    print(f"  - Volcano plots: {len(plots_for_report['volcano_plots'])}")
    print(f"  - Enrichment bar plots: {len(plots_for_report['enrichment_bar_plots'])}")
    print(f"  - KEGG pathway plots: {len(plots_for_report['kegg_pathway_plots'])}")
    print(f"  - Total PNG plots: {len(plots_for_report['all_plots'])}")
    print(f"  - HTML interactive: {len(plot_paths)}")

    gc.collect()
    success_message = (f"Successfully retrieved {metadata.shape[0] if metadata is not None else 0} "
                       f"metacells (scRNA-seq; about 200 cells each)")
    return {
        "success": (analysis_success and de_plots_status != "failed" and enrichment_plot_status != "failed") if de_gated else True,
        "session_dir": session_dir,
        "data_type": "single-cell RNA-seq (scRNA-seq) cohort",
        # num_samples counts these units
        "sample_unit": "metacell (SEACells aggregate of about 200 cells of one cell type from one source file)",
        "label_column": actual_label,
        "requested_label": label,
        "actual_label": actual_label,
        "label_fallback_message": label_fallback_message,
        "comparison_name": comparison_name if de_gated else None,
        "contrast": contrast,
        "control_source": control_source,
        "control_note": control_note,
        "num_samples": metadata.shape[0] if metadata is not None else 0,
        "num_features": n_features,
        "top_gene_indices": top_gene_indices,
        "top_gene_names": top_gene_names,
        "top_gene_values": top_gene_values,
        "top_genes_by_fdr": top_genes_by_fdr,
        "enrichment_top_terms": enrichment_top_terms,
        "similar_terms": similar_terms,
        "extracted_entities": fetch_dict,
        "retrieval_success": True,
        "analysis_success": analysis_success,
        "de_success": de_success,
        "enrichment_success": enrichment_success,
        "enrichment_status": enrichment_status,
        "enrichment_error": enrichment_error,
        "cohort_diagnostics": cohort_diagnostics,
        "cohort_verdict": (cohort_diagnostics or {}).get("verdict"),
        "kegg_success": kegg_success,
        "plot_success": (de_plots_status == "success" and enrichment_plot_status in ("success", "empty")) if de_gated else None,
        "de_plots_status": de_plots_status,
        "de_plots_error": de_plots_error,
        "enrichment_plot_status": enrichment_plot_status,
        "enrichment_plot_error": enrichment_plot_error,
        "analysis_paths": analysis_paths,
        "plot_paths": plot_paths,  # HTML paths for backward compatibility
        "plots_for_report": plots_for_report,  # Categorized plots with relative paths for embedding
        "message": success_message,
        "timing": {
            "input": times['input_end'] - times['start'],
            "fetch": times['fetch_end'] - times['input_end'],
            "gene_analysis": times['gene_end'] - times['fetch_end'],
            "de_analysis": times['de_end'] - times['gene_end'],
            "kegg_plotting": times['kegg_end'] - times['de_end'],
            "total": times['end'] - times['start']
        }
    }
