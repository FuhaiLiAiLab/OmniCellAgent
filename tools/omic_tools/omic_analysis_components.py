import os
import sys
import gc
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

import json

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.path_config import get_path
if __package__:
    from .subprocess_r import run_r_script
    from .de_results_io import DE_RESULT_DTYPES, read_de_results
    from .r_plotting import render_analysis_plots
    from .enrichr_client import (EnrichrError, fetch_enrichment, read_enrichment_results,
                                 write_status, record_enrichment_failure, archive_enrichment_plots)
else:
    from subprocess_r import run_r_script
    from de_results_io import DE_RESULT_DTYPES, read_de_results
    from r_plotting import render_analysis_plots
    from enrichr_client import (EnrichrError, fetch_enrichment, read_enrichment_results,
                                write_status, record_enrichment_failure, archive_enrichment_plots)

BMG_DIR = get_path('external.biomedgraphica_dir', absolute=True)
# Use relative path but resolve it once at module load time to avoid issues with parallel processes
OUTPUT_DIR = get_path('data.dataset_outputs', absolute=True, create=True)


def _analysis_r_script():
    try:
        return get_path("analysis.r_script", absolute=True)
    except KeyError:
        return str(Path(__file__).with_name("run_casestudy.R"))


def create_directories_parallel(directories):
    """Create output directories without spawning processes for filesystem I/O."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def _resolve_gene_names(gene_names, n_features: int, session_dir: str = None) -> list:
    """Resolve gene symbols for the expression feature axis.

    The loader returns a DataFrame whose columns are HGNC symbols, so the caller
    should pass them in. Falls back to the loader's own sidecar file, then fails
    loudly rather than guessing.
    """
    if gene_names is not None:
        names = list(gene_names)
        if len(names) != n_features:
            raise ValueError(
                f"gene_names has {len(names)} entries but the matrix has "
                f"{n_features} features."
            )
        return names

    choice_path = os.path.join(session_dir or "", "bmg_to_gene_choice.csv")
    if os.path.exists(choice_path):
        names = pd.read_csv(choice_path)["gene_name"].tolist()
        if len(names) == n_features:
            return names
        raise ValueError(
            f"{choice_path} has {len(names)} genes but the matrix has "
            f"{n_features} features."
        )

    raise ValueError(
        "Cannot determine gene names: none were passed and "
        f"{choice_path} is missing. The loader emits gene symbols as the "
        "columns of `dataset.data`; capture them before np.nan_to_num()."
    )


def omic_analysis(disease_name: str, data_dict: dict, enable_plotting: bool = True, session_dir: str = None, gene_names: list = None, diagnostics_text: str = "", *,
                  ref_label="Healthy", alt_label="Diseased", input_scale="linear_cp10k", r_timeout=3600,
                  enrichment_timeout=60, enrichment_databases=None) -> dict:
    """
    Perform omic analysis on the input data dictionary.

    Args:
        disease_name (str): The name of the disease for which the analysis is performed.
        data_dict (dict): A dictionary containing omic data.
        enable_plotting (bool): Whether to enable plots (default: True).
        session_dir (str): Optional session directory path to save results. If None, uses OUTPUT_DIR.
        ref_label, alt_label: Actual group names; the R contrast is alternate minus reference.
        input_scale: Explicit expression scale supplied to R; default linear CP10K.
        r_timeout: Maximum R subprocess duration in seconds.
        enrichment_timeout: Timeout in seconds for each real Enrichr HTTP request.
        enrichment_databases: Optional library subset; None preserves existing defaults.

    Returns:
        dict: A dictionary containing the results of the omic analysis.
    """
    combined_normal_matrix = data_dict.get("normal_omic_feature", None)
    combined_disease_matrix = data_dict.get("disease_omic_feature", None)
    
    # Print memory info
    print(f"Input matrices - Disease: {combined_disease_matrix.shape}, Normal: {combined_normal_matrix.shape}")
    
    # Fill NaN values with 0
    combined_disease_matrix = np.nan_to_num(combined_disease_matrix, nan=0.0)
    combined_normal_matrix = np.nan_to_num(combined_normal_matrix, nan=0.0)
    
    n_features = combined_disease_matrix.shape[1]
    gene_names = _resolve_gene_names(gene_names, n_features, session_dir)

    # The workflow supplies an HGNC protein-coding mapping to CellTOSG, which
    # selects one representative column per retained symbol. With the bundled
    # reference and current BMG mapping this is 19,109 genes; use the supplied
    # names rather than assuming a fixed width or rebuilding the entity axis.
    print("Creating disease DataFrame...")
    combined_disease_df = pd.DataFrame(
        combined_disease_matrix.T,
        columns=[f'ds_sample_{i}' for i in range(combined_disease_matrix.shape[0])],
    )
    combined_disease_df.insert(0, 'Name', gene_names)
    del combined_disease_matrix
    gc.collect()

    print("Creating normal DataFrame...")
    combined_normal_df = pd.DataFrame(
        combined_normal_matrix.T,
        columns=[f'ns_sample_{i}' for i in range(combined_normal_matrix.shape[0])],
    )
    combined_normal_df.insert(0, 'Name', gene_names)
    del combined_normal_matrix
    gc.collect()

    # Print the shapes of the aggregated DataFrames
    print(f"Disease DataFrame shape after aggregation: {combined_disease_df.shape}")
    print(f"Normal DataFrame shape after aggregation: {combined_normal_df.shape}")

    # Resolve the base directory for all analysis outputs
    # Use session directory if provided, otherwise use the default OUTPUT_DIR
    if session_dir:
        base_dir = session_dir
        print(f"Using session directory for analysis results: {session_dir}")
    else:
        base_dir = OUTPUT_DIR
        print(f"Using default output directory for analysis results: {OUTPUT_DIR}")
    os.makedirs(base_dir, exist_ok=True)

    # The R adapter below exports these matrices once, in this same session.

    de_output_dir = os.path.join(base_dir, "differential_expression")
    volcano_dir = os.path.join(base_dir, "volcano_plots")
    enrich_output_dir = os.path.join(base_dir, "enrichment_results")
    plot_enrich_dir = os.path.join(enrich_output_dir, "enrichment_plots")
    
    directories_to_create = [de_output_dir, volcano_dir, enrich_output_dir, plot_enrich_dir]
    create_directories_parallel(directories_to_create)

    if diagnostics_text:
        try:
            with open(os.path.join(de_output_dir, "COHORT_DIAGNOSTICS.txt"), "w", encoding="utf-8") as handle:
                handle.write(diagnostics_text + "\n")
        except Exception as e:
            print(f"[WARNING] Could not write COHORT_DIAGNOSTICS.txt: {e}")

    # Perform the differential expression analysis
    significant_genes, result_df = perform_unpaired_differential_expression(
        disease_df=combined_disease_df, 
        normal_df=combined_normal_df, 
        p_value_threshold=0.05,
        log2fc_threshold=1.5,
        sig_top_n=1000,  # change to 1000
        disease=disease_name,
        de_output_dir=de_output_dir,
        n_jobs=-1,  # Retained legacy argument; DE execution now belongs to R.
        diagnostics_text=diagnostics_text,
        session_dir=base_dir,
        ref_label=ref_label,
        alt_label=alt_label,
        input_scale=input_scale,
        r_timeout=r_timeout,
    )

    enrichment_results = perform_enrichment_analysis(
        significant_genes=significant_genes, disease_name=disease_name,
        enrich_output_dir=enrich_output_dir, fast_mode=False,
        pathway_dbs=enrichment_databases,
        disease_dbs=[] if enrichment_databases is not None else None,
        request_timeout=enrichment_timeout,
    )
    enrichment_status = json.loads((Path(enrich_output_dir) / "enrichment_status.json").read_text(encoding="utf-8"))["status"]
    print(f"  - Enrichment analysis: {enrichment_status}")
    de_plots = {"status": "skipped", "files": [], "error": None}
    if enable_plotting:
        try:
            de_plots = render_analysis_plots(_analysis_r_script(), base_dir,
                                             ref_csv=Path(base_dir).resolve() / f"foranalysis_combined_normal_df_{disease_name}.csv",
                                             alt_csv=Path(base_dir).resolve() / f"foranalysis_combined_disease_df_{disease_name}.csv",
                                             input_scale=input_scale,
                                             ref_label=ref_label, alt_label=alt_label, timeout=r_timeout)
        except Exception as error:
            de_plots = {"status": "failed", "files": [], "error": f"{type(error).__name__}: {error}"}
            print(f"[DE plots] Failed; DE and enrichment results are preserved: {error}")
    
    # Clean up large DataFrames that are no longer needed
    del combined_disease_df, combined_normal_df, significant_genes, result_df
    gc.collect()
    
    # Merge data dictionary with results path dictionary
    data_and_analysis_dict = {
        "data_dir": base_dir,  # Use the actual base directory (session or default)
        "differential_expression_dir": de_output_dir,
        "volcano_plots_dir": volcano_dir,
        "enrichment_results_dir": enrich_output_dir,
        "enrichment_plots_dir": plot_enrich_dir,
        "enrichment_status": enrichment_status,
        "enrichment_success": enrichment_status in ("success", "empty"),
        "de_plots_status": de_plots["status"],
        "de_plot_files": de_plots["files"],
        "de_plots_error": de_plots.get("error"),
    }

    return data_and_analysis_dict
        

def perform_unpaired_differential_expression(disease_df, normal_df,
                                    p_value_threshold=0.05, log2fc_threshold=1.5,
                                    sig_top_n=100, n_jobs=16, disease="Disease",
                                    de_output_dir=None, diagnostics_text: str = "", *,
                                    session_dir=None, ref_label="Healthy", alt_label="Diseased",
                                    input_scale="linear_cp10k", r_timeout=3600) -> tuple[dict, pd.DataFrame]:
    """Delegate metacell-level DE to the existing R/limma implementation.

    Preserve the historical function name, positional arguments, five CSVs and
    (all/up/down subsets, full table) return structure. n_jobs and the former
    log2fc_threshold argument are retained for callers; neither alters limma.
    Input matrices must be on the explicitly declared scale.
    """
    if de_output_dir is None:
        de_output_dir = Path(session_dir or OUTPUT_DIR) / "differential_expression"
    de_dir = Path(de_output_dir).resolve()
    base_dir = Path(session_dir).resolve() if session_dir else de_dir / "inputs"
    analysis_dir = (Path(session_dir).resolve() if session_dir else de_dir) / "casestudy_R"
    base_dir.mkdir(parents=True, exist_ok=True)
    de_dir.mkdir(parents=True, exist_ok=True)
    ref_csv = base_dir / f"foranalysis_combined_normal_df_{disease}.csv"
    alt_csv = base_dir / f"foranalysis_combined_disease_df_{disease}.csv"
    normal_df.to_csv(ref_csv, index=False, encoding="utf-8")
    disease_df.to_csv(alt_csv, index=False, encoding="utf-8")

    r_script = _analysis_r_script()

    filenames = (
        "unpaired_differential_expression_results.csv",
        "significant_genes_by_fdr.csv",
        "significant_genes_by_fc.csv",
        "significant_upregulated_genes.csv",
        "significant_downregulated_genes.csv",
    )
    # Read only this invocation's outputs. A failed or empty R invocation cannot
    # accidentally succeed by reading CSVs left by an earlier session run.
    with tempfile.TemporaryDirectory(prefix=".r-de-", dir=base_dir) as temporary:
        staging = Path(temporary)
        staged_de = staging / "differential_expression"
        staged_analysis = staging / "casestudy_R"
        args = [
            "--stage", "de", str(base_dir), str(staged_analysis),
            "--ref-csv", str(ref_csv), "--alt-csv", str(alt_csv),
            "--ref-label", ref_label, "--alt-label", alt_label,
            "--input-scale", input_scale, "--de-dir", str(staged_de),
            "--final-out-dir", str(analysis_dir), "--final-de-dir", str(de_dir),
            "--de-fdr", str(p_value_threshold), "--top-n", str(sig_top_n),
        ]
        if diagnostics_text:
            diagnostic_path = staging / "diagnostics.txt"
            diagnostic_path.write_text(diagnostics_text + "\n", encoding="utf-8")
            args.extend(["--diagnostics-file", str(diagnostic_path)])
        run_r_script(r_script, args, timeout=r_timeout)
        tables = {name: read_de_results(staged_de / name) for name in filenames}
        if tables[filenames[0]].empty:
            raise ValueError("R produced an empty main DE table")
        for name, table in tables.items():
            if table.columns.tolist() != list(DE_RESULT_DTYPES):
                raise ValueError(f"Invalid DE output columns: {name}")
            numeric = table[["log2_fold_change", "effect_size", "p_value", "FDR", "abs_log2_fc"]]
            if table["Name"].isna().any() or not table["Name"].is_unique or not np.isfinite(numeric.to_numpy()).all():
                raise ValueError(f"Invalid gene names or statistics in DE output: {name}")
        required_native = [staged_analysis / "DE_results_table.csv", staged_analysis / "analysis_state.rds"]
        for path in required_native:
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"R did not produce required output: {path}")
        analysis_dir.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            os.replace(staged_de / name, de_dir / name)
        for path in required_native:
            os.replace(path, analysis_dir / path.name)

    result_df = tables[filenames[0]]
    return {"all": tables[filenames[1]], "up": tables[filenames[3]],
            "down": tables[filenames[4]]}, result_df




def perform_enrichment_analysis(significant_genes, disease_name="Disease", 
                               pathway_dbs=None, disease_dbs=None, 
                               visualize=True, enrich_top_n=10,
                               enrich_output_dir=None, fast_mode=True, request_timeout=60):
    """Keep any enrichment-stage failure distinct from DE failure or emptiness."""
    try:
        return _perform_enrichment_analysis(significant_genes, disease_name, pathway_dbs, disease_dbs,
                                            visualize, enrich_top_n, enrich_output_dir, fast_mode, request_timeout)
    except Exception as error:
        if enrich_output_dir is not None:
            record_enrichment_failure(enrich_output_dir, error)
        if isinstance(error, EnrichrError):
            raise
        raise EnrichrError(f"Enrichment stage failed: {type(error).__name__}: {error}") from error


def _perform_enrichment_analysis(significant_genes, disease_name="Disease",
                                pathway_dbs=None, disease_dbs=None,
                                visualize=True, enrich_top_n=10,
                                enrich_output_dir=None, fast_mode=True, request_timeout=60):
    """
    Performs enrichment analysis on significant genes from differential expression analysis
    with flexible database selection.
    
    Args:
        significant_genes (dict): Dictionary containing dataframes of significant genes
                                 with keys 'all', 'up', and 'down'
        disease_name (str): Name of the disease for output directory naming
        pathway_dbs (list): List of pathway databases to use (if None, uses defaults)
        disease_dbs (list): List of disease databases to use (if None, uses defaults)
        visualize (bool): Whether to create visualizations for top enriched terms
        enrich_top_n (int): Number of top terms to include in visualizations
        enrich_output_dir (str): Directory to save enrichment results and plots
        
    Returns:
        dict: Dictionary with enrichment results for each gene set
    """
    # Default database lists - use fewer databases in fast mode
    if fast_mode:
        # Reduced set for speed - only most important databases
        default_pathway_dbs = [
            'GO_Biological_Process_2021',
            'KEGG_2021_Human',
            'Reactome_2022',
        ]
        
        default_disease_dbs = [
            'DisGeNET',
            'Human_Phenotype_Ontology',
        ]
    else:
        # Full set for comprehensive analysis
        default_pathway_dbs = [
            'GO_Biological_Process_2021',
            'GO_Molecular_Function_2021',
            'GO_Cellular_Component_2021',
            'KEGG_2021_Human',
            'Reactome_2022',
            'WikiPathways_2019_Human',
            'MSigDB_Hallmark_2020',
        ]
        
        default_disease_dbs = [
            'DisGeNET',
            'OMIM_Disease',
            'OMIM_Expanded',
            'Human_Phenotype_Ontology',
            'Jensen_DISEASES',
            'GTEx_Tissue_Expression_Down',
            'GTEx_Tissue_Expression_Up',
        ]
    
    # Use provided databases or defaults
    pathway_dbs = pathway_dbs if pathway_dbs is not None else default_pathway_dbs
    disease_dbs = disease_dbs if disease_dbs is not None else default_disease_dbs
    
    # Combine all databases
    all_dbs = pathway_dbs + disease_dbs
    
    results = {}
    status_path = Path(enrich_output_dir) / "enrichment_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    run_status = {"status": "running", "groups": {}}
    write_status(status_path, run_status)
    archive_enrichment_plots(status_path.parent / "enrichment_plots")
    
    # Process each set of genes (all, up, down) in parallel
    def process_gene_set(set_name, gene_df):
        gene_list = gene_df['Name'].tolist()
        print(f"\nPerforming enrichment analysis on {len(gene_list)} {set_name}-regulated genes...")
        
        # Create a sanitized ID for the file paths
        sample_id = f"{disease_name.replace(' ', '_')}_{set_name}_regulated"
        
        # Store disease name for later use in results
        set_results = {"disease": disease_name, "gene_count": len(gene_list)}
        
        # Run enrichment analysis on this gene set with specified databases
        try:
            enrichr_results = enrichr_analysis(gene_list, sample_id, enrich_output_dir,
                                              databases=all_dbs, request_timeout=request_timeout)
        except EnrichrError as error:
            run_status["groups"][set_name] = {"status": "failed", "gene_count": len(gene_list), "error": str(error)}
            run_status["status"] = "failed"
            write_status(status_path, run_status)
            raise
        returned = sum(len(payload[db]) for db, payload in enrichr_results.items())
        run_status["groups"][set_name] = {
            "status": "skipped" if not gene_list else "success" if returned else "empty",
            "gene_count": len(gene_list), "returned_rows": returned,
        }
        write_status(status_path, run_status)
        
        # Combine enrichr results with metadata
        if enrichr_results:
            set_results.update(enrichr_results)
            return set_name, set_results
        return set_name, None
    
    # Process gene sets sequentially to avoid nested parallelism issues (which can cause segfaults)
    print("Processing enrichment analysis for gene sets...")
    parallel_results = []
    for set_name, gene_df in significant_genes.items():
        result = process_gene_set(set_name, gene_df)
        parallel_results.append(result)
    
    # Collect results
    for set_name, set_results in parallel_results:
        if set_results is not None:
            results[set_name] = set_results
    
    states = [value["status"] for value in run_status["groups"].values()]
    run_status["status"] = "success" if "success" in states else "empty" if "empty" in states else "skipped"
    write_status(status_path, run_status)
    return results

def enrichr_analysis(gene_list, sample_id, enrich_output_dir=None, databases=None, *, request_timeout=60):
    """Call the real Enrichr endpoints; errors raise instead of becoming None."""
    try:
        results = fetch_enrichment(gene_list, sample_id, enrich_output_dir, databases, request_timeout)
        libraries = list(results)
        create_enrichment_summary(gene_list, results, sample_id,
                                  str(Path(enrich_output_dir) / sample_id), libraries)
        return results
    except Exception as error:
        if enrich_output_dir is not None and Path(sample_id).name == sample_id:
            record_enrichment_failure(Path(enrich_output_dir) / sample_id, error)
        if isinstance(error, EnrichrError):
            raise
        raise EnrichrError(f"Enrichr output failed for {sample_id}: {type(error).__name__}: {error}") from error

def create_enrichment_summary(gene_list, enrichr_results, sample_id, sample_dir, databases):
    """
    Creates summary files from enrichment results
    
    Args:
        gene_list (list): List of genes analyzed
        enrichr_results (dict): Enrichr API results
        sample_id (str): Sample identifier
        sample_dir (str): Directory to save summary
        databases (list): List of databases used for enrichment
    """
    # Classify databases as pathway or disease related
    pathway_dbs = [db for db in databases if any(db_prefix in db for db_prefix in 
                  ['GO_', 'KEGG', 'Reactome', 'WikiPathways', 'MSigDB'])]
    
    disease_dbs = [db for db in databases if db not in pathway_dbs]
    
    # Create a comprehensive summary file
    summary_file = f"{sample_dir}/summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"ENRICHMENT ANALYSIS SUMMARY FOR {sample_id}\n")
        f.write(f"Number of genes analyzed: {len(gene_list)}\n\n")
        
        # Pathway section
        f.write("=" * 80 + "\n")
        f.write("PATHWAY ENRICHMENT RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for database in pathway_dbs:
            if database in enrichr_results and database in enrichr_results[database]:
                results = enrichr_results[database][database]
                if not results:
                    f.write(f"\n{database}: No returned terms.\n")
                if results:
                    top_terms = results[:5]
                    
                    f.write(f"\n{database} Top 5 Terms:\n")
                    
                    for term in top_terms:
                        term_name = term[1]
                        p_value = float(term[2])
                        adj_p = term[6] if len(term) > 6 else "N/A"
                        genes = term[5] if len(term) > 5 else "N/A"
                        
                        f.write(f"  - {term_name} (P-value: {p_value:.3e}, Adj.P: {adj_p})\n")
                        f.write(f"    Genes: {genes}\n\n")
        
        # Disease section
        f.write("\n" + "=" * 80 + "\n")
        f.write("DISEASE ENRICHMENT RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for database in disease_dbs:
            if database in enrichr_results and database in enrichr_results[database]:
                results = enrichr_results[database][database]
                if not results:
                    f.write(f"\n{database}: No returned terms.\n")
                if results:
                    top_terms = results[:5]
                    
                    f.write(f"\n{database} Top 5 Terms:\n")
                    
                    for term in top_terms:
                        term_name = term[1]
                        p_value = float(term[2])
                        adj_p = term[6] if len(term) > 6 else "N/A"
                        genes = term[5] if len(term) > 5 else "N/A"
                        
                        f.write(f"  - {term_name} (P-value: {p_value:.3e}, Adj.P: {adj_p})\n")
                        f.write(f"    Genes: {genes}\n\n")
    
    print(f"Detailed enrichment summary saved to {summary_file}")
