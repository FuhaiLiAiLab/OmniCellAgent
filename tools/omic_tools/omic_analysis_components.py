import os
import sys
import gc
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

# Set matplotlib to use non-interactive backend BEFORE importing pyplot
# This prevents 'main thread is not in main loop' errors in agent context
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import time
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
import warnings

# Suppress matplotlib threading warnings for parallel plotting
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.path_config import get_path
if __package__:
    from .subprocess_r import run_r_script
    from .de_results_io import DE_RESULT_DTYPES, read_de_results
    from .enrichr_client import (EnrichrError, fetch_enrichment, read_enrichment_results,
                                 write_status, record_enrichment_failure, archive_enrichment_plots)
else:
    from subprocess_r import run_r_script
    from de_results_io import DE_RESULT_DTYPES, read_de_results
    from enrichr_client import (EnrichrError, fetch_enrichment, read_enrichment_results,
                                write_status, record_enrichment_failure, archive_enrichment_plots)

BMG_DIR = get_path('external.biomedgraphica_dir', absolute=True)
# Use relative path but resolve it once at module load time to avoid issues with parallel processes
OUTPUT_DIR = get_path('data.dataset_outputs', absolute=True, create=True)


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

    # Create volcano plots with different thresholds in parallel
    # Define volcano plot parameters
    volcano_configs = [
        {
            'save_path': os.path.join(volcano_dir, "volcano_plot.png"),
            'p_value_threshold': 0.001,
            'log2fc_threshold': 1.5,
            'plot_title': "Differential Expression: Disease vs Control",
            'highlight_top_n': 10
        },
        {
            'save_path': os.path.join(volcano_dir, "volcano_plot_permissive.png"),
            'p_value_threshold': 0.025,
            'log2fc_threshold': 0.75,
            'plot_title': "Differential Expression: Disease vs Control (Permissive)",
            'highlight_top_n': 10
        }
    ]

    # Run volcano plots, enrichment analysis, and optionally plotting sequentially
    # (parallel matplotlib causes 'main thread is not in main loop' errors)
    def create_volcano_plots():
        print("Creating volcano plots...")
        for config in volcano_configs:
            create_volcano_plot(
                result_df,
                config['p_value_threshold'],
                config['log2fc_threshold'],
                config['save_path'],
                config['plot_title'],
                config['highlight_top_n'],
                diagnostics_text=diagnostics_text,
            )
        return "Volcano plots completed"
    
    def run_enrichment_analysis():
        return perform_enrichment_analysis(
            significant_genes=significant_genes, 
            disease_name=disease_name, 
            enrich_output_dir=enrich_output_dir,
            fast_mode=False,
            pathway_dbs=enrichment_databases,
            disease_dbs=[] if enrichment_databases is not None else None,
            request_timeout=enrichment_timeout,
        )
    
    def run_enrichment_plotting():
        if enable_plotting:
            # Small delay to ensure enrichment analysis starts first
            time.sleep(1)  # Reduced from 2 seconds
            return plot_selected_enrichment(
                disease_name=disease_name, 
                regulation_type=["all", "up", "down"], 
                databases=["Reactome_2022", "KEGG_2021_Human"],
                enrich_top_n=10,
                plot_enrich_dir=plot_enrich_dir,
                enrich_results_dir=enrich_output_dir  # Pass the enrichment results directory
            )
        else:
            return "Enrichment plotting skipped for speed"
    
    # Run tasks sequentially to avoid matplotlib threading issues
    # (nested parallelism with matplotlib causes 'main thread is not in main loop' errors)
    print("Running analysis tasks sequentially...")
    
    # Execute volcano plots first
    if enable_plotting:
        volcano_result = create_volcano_plots()
        print(f"  - Volcano plots: {volcano_result}")
    
    # Execute enrichment analysis
    enrichment_results = run_enrichment_analysis()
    enrichment_status = json.loads((Path(enrich_output_dir) / "enrichment_status.json").read_text(encoding="utf-8"))["status"]
    print(f"  - Enrichment analysis: {enrichment_status}")
    
    # Execute enrichment plotting if enabled
    if enable_plotting and enrichment_status == "success":
        plotting_result = run_enrichment_plotting()
        print(f"  - Enrichment plotting: {plotting_result}")
    
    print(f"\nAll analysis tasks completed successfully!")
    
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

    try:
        r_script = get_path("analysis.r_script", absolute=True)
    except KeyError:
        # Existing local configs need not be rewritten merely to use this checkout.
        r_script = str(Path(__file__).with_name("run_casestudy.R"))

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


def create_volcano_plot(result_df, p_value_threshold=0.025, log2fc_threshold=1.5,
                         save_path=None, plot_title=None, highlight_top_n=50,
                         diagnostics_text: str = ""):
    """
    Create a volcano plot from differential expression results.
    Saves both static PNG and interactive HTML (plotly) versions.
    
    Args:
        result_df (pd.DataFrame): DataFrame with differential expression results.
        p_value_threshold (float): FDR threshold for significance.
        log2fc_threshold (float): Log2 fold change threshold for biological significance.
        save_path (str): Path to save the plot (PNG). HTML will be saved alongside.
        plot_title (str): Title for the plot.
        highlight_top_n (int): Number of top genes to highlight by name.
        
    Returns:
        None
    """
    # Make a copy to avoid modifying original
    plot_df = result_df.copy()
    
    # Transform p-values to -log10 scale
    plot_df['neg_log10_fdr'] = -np.log10(plot_df['FDR'])
    
    # Add a column to categorize genes
    plot_df['de_category'] = 'Not Significant'
    
    # Upregulated genes (log2FC > threshold and FDR < p_value_threshold)
    plot_df.loc[(plot_df['log2_fold_change'] > log2fc_threshold) & 
                  (plot_df['FDR'] < p_value_threshold), 'de_category'] = 'Upregulated'
    
    # Downregulated genes (log2FC < -threshold and FDR < p_value_threshold)
    plot_df.loc[(plot_df['log2_fold_change'] < -log2fc_threshold) & 
                  (plot_df['FDR'] < p_value_threshold), 'de_category'] = 'Downregulated'
    
    # Count the number of genes in each category for the title
    n_up = sum(plot_df['de_category'] == 'Upregulated')
    n_down = sum(plot_df['de_category'] == 'Downregulated')
    
    # =========================================================================
    # STATIC MATPLOTLIB PLOT (PNG)
    # =========================================================================
    # Set up the figure with better aspect ratio for volcano plots
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a color map for the categories
    color_map = {'Upregulated': 'red', 'Downregulated': 'blue', 'Not Significant': 'grey'}
    
    # Create a scatter plot
    sns.scatterplot(
        data=plot_df,
        x='log2_fold_change',
        y='neg_log10_fdr',
        hue='de_category',
        palette=color_map,
        alpha=0.6,
        s=50,
        edgecolor=None,
        linewidth=0,
        ax=ax
    )
    
    # Add threshold lines
    ax.axhline(y=-np.log10(p_value_threshold), linestyle='--', color='black', alpha=0.3)
    ax.axvline(x=log2fc_threshold, linestyle='--', color='black', alpha=0.3)
    ax.axvline(x=-log2fc_threshold, linestyle='--', color='black', alpha=0.3)
    
    # Identify top significant genes to label
    sig_genes = plot_df[plot_df['FDR'] < p_value_threshold].copy()
    sig_genes['importance'] = sig_genes['neg_log10_fdr'] * abs(sig_genes['log2_fold_change'])
    
    # Get top genes to label (limit to fewer for cleaner plot)
    n_labels = min(highlight_top_n, 12)  # Limit labels to avoid crowding
    top_genes = sig_genes.sort_values('importance', ascending=False).head(n_labels)
    
    # Calculate axis limits first (needed for label constraints)
    max_y = min(np.nanmax(plot_df['neg_log10_fdr']), 50)
    x_max = min(np.nanmax(abs(plot_df['log2_fold_change'])), 10)
    
    # Set axis limits early so adjustText respects them
    ax.set_xlim(-x_max * 1.1, x_max * 1.1)
    ax.set_ylim(0, max_y * 1.1)
    
    # Simple annotation approach - more reliable than adjustText for this use case
    # Place labels with slight offset and white background for readability
    for i, (_, gene) in enumerate(top_genes.iterrows()):
        x_pos = gene['log2_fold_change']
        y_pos = gene['neg_log10_fdr']
        
        # Determine text alignment based on position
        if x_pos > 0:
            ha = 'left'
            x_offset = 0.15
        else:
            ha = 'right'
            x_offset = -0.15
        
        # Stagger y offset slightly to reduce overlap
        y_offset = 0.3 + (i % 3) * 0.2
        
        ax.annotate(
            gene['Name'],
            xy=(x_pos, y_pos),
            xytext=(x_pos + x_offset, y_pos + y_offset),
            fontsize=7,
            ha=ha,
            va='bottom',
            fontweight='normal',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.15'),
            arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5, lw=0.5,
                          connectionstyle='arc3,rad=0.1')
        )
        
    # Set plot labels and title
    ax.set_xlabel('log2 Fold Change', fontsize=12)
    ax.set_ylabel('-log10(FDR)', fontsize=12)
    ax.set_title(f"{plot_title}\n(Up: {n_up}, Down: {n_down}, FDR < {p_value_threshold}, |log2FC| > {log2fc_threshold})", 
              fontsize=12, pad=10)
    
    # Add a legend
    ax.legend(title='Differential Expression', loc='lower right', frameon=True, fontsize=9)
    
    # Customize the plot
    ax.grid(True, linestyle='--', alpha=0.3)
    
    if diagnostics_text:
        verdict_line = diagnostics_text.splitlines()[0]
        plt.figtext(0.5, 0.005, verdict_line, ha="center", fontsize=8, color="firebrick")

    # Save the PNG plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.2, facecolor='white')
    plt.close(fig)
    print(f"Volcano plot (PNG) saved to {save_path}")
    
    # =========================================================================
    # INTERACTIVE PLOTLY PLOT (HTML)
    # =========================================================================
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Color mapping for plotly
        color_discrete_map = {
            'Upregulated': '#e74c3c',  # Red
            'Downregulated': '#3498db',  # Blue
            'Not Significant': '#95a5a6'  # Grey
        }
        
        # Create interactive scatter plot
        fig = px.scatter(
            plot_df,
            x='log2_fold_change',
            y='neg_log10_fdr',
            color='de_category',
            color_discrete_map=color_discrete_map,
            hover_data={
                'Name': True,
                'log2_fold_change': ':.3f',
                'FDR': ':.2e',
                'neg_log10_fdr': ':.2f',
                'de_category': True
            },
            labels={
                'log2_fold_change': 'log2 Fold Change',
                'neg_log10_fdr': '-log10(FDR)',
                'de_category': 'Category'
            },
            title=f"{plot_title}<br><sup>Up: {n_up}, Down: {n_down} | FDR < {p_value_threshold}, |log2FC| > {log2fc_threshold}</sup>",
            opacity=0.6
        )
        
        # Add threshold lines
        fig.add_hline(y=-np.log10(p_value_threshold), line_dash="dash", line_color="black", opacity=0.3)
        fig.add_vline(x=log2fc_threshold, line_dash="dash", line_color="black", opacity=0.3)
        fig.add_vline(x=-log2fc_threshold, line_dash="dash", line_color="black", opacity=0.3)
        
        # Add annotations for top genes
        for _, gene in top_genes.head(20).iterrows():  # Limit to top 20 for readability
            fig.add_annotation(
                x=gene['log2_fold_change'],
                y=gene['neg_log10_fdr'],
                text=gene['Name'],
                showarrow=True,
                arrowhead=0,
                arrowsize=0.5,
                arrowwidth=1,
                ax=20,
                ay=-20,
                font=dict(size=9),
                bgcolor="white",
                opacity=0.8
            )
        
        # Update layout
        fig.update_layout(
            xaxis_title="log2 Fold Change",
            yaxis_title="-log10(FDR)",
            legend_title="Differential Expression",
            hovermode='closest',
            template='plotly_white',
            width=1000,
            height=800
        )
        
        # Set axis limits
        fig.update_xaxes(range=[-x_max * 1.05, x_max * 1.05])
        fig.update_yaxes(range=[0, max_y * 1.05])
        
        # Save as HTML
        html_path = save_path.replace('.png', '.html')
        fig.write_html(html_path, include_plotlyjs='cdn')
        print(f"Volcano plot (HTML) saved to {html_path}")
        
        # Also export a high-quality PNG from Plotly (better than matplotlib for this)
        # This will be used in the PDF report
        try:
            # Use kaleido for static export if available
            fig.write_image(save_path, width=1200, height=960, scale=2)
            print(f"Volcano plot (PNG from Plotly) saved to {save_path}")
        except Exception as e:
            print(f"Note: Could not export PNG from Plotly ({e}), using matplotlib version")
        
    except ImportError:
        print("Warning: plotly not installed. Skipping interactive HTML volcano plot.")
    except Exception as e:
        print(f"Warning: Failed to create interactive volcano plot: {e}")
    
    # Show plot summary statistics
    print(f"Total genes plotted: {len(plot_df)}")
    print(f"Significant upregulated genes (FDR < {p_value_threshold}, log2FC > {log2fc_threshold}): {n_up}")
    print(f"Significant downregulated genes (FDR < {p_value_threshold}, log2FC < -{log2fc_threshold}): {n_down}")


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

def plot_selected_enrichment(disease_name, regulation_type="all", 
                             databases=["Reactome_2022", "KEGG_2021_Human"], 
                             enrich_top_n=10, plot_enrich_dir=None,
                             enrich_results_dir=None):
    """
    Create enrichment plots by directly reading the CSV result files.
    
    Args:
        disease_name (str): Name of the disease (used to find file paths)
        regulation_type (str or list): "all", "up", "down", or a list of these
        databases (str or list): One or more databases to plot
        enrich_top_n (int): Number of top terms to show in each plot
        plot_enrich_dir (str): Directory to save the plots
        enrich_results_dir (str): Directory containing the enrichment results CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(plot_enrich_dir, exist_ok=True)
    
    # Determine base directory for enrichment results
    if enrich_results_dir is None:
        # Fallback to old hardcoded path if not provided (backward compatibility)
        enrich_results_base = "./fetched_data/enrichment_results"
    else:
        enrich_results_base = enrich_results_dir
    
    # Convert inputs to lists if they're not already
    if isinstance(regulation_type, str):
        regulation_types = [regulation_type]
    else:
        regulation_types = regulation_type
    
    if isinstance(databases, str):
        databases = [databases]
    
    # Display name mapping for databases
    db_name_map = {
        'GO_Biological_Process_2021': 'GO Biological Process',
        'GO_Molecular_Function_2021': 'GO Molecular Function',
        'GO_Cellular_Component_2021': 'GO Cellular Component',
        'KEGG_2021_Human': 'KEGG Pathways',
        'Reactome_2022': 'Reactome Pathways',
        'WikiPathways_2019_Human': 'WikiPathways',
        'MSigDB_Hallmark_2020': 'MSigDB Hallmark',
        'DisGeNET': 'DisGeNET',
        'OMIM_Disease': 'OMIM Disease',
        'OMIM_Expanded': 'OMIM Expanded',
        'Human_Phenotype_Ontology': 'Human Phenotype',
        'Jensen_DISEASES': 'Jensen DISEASES'
    }
    
    # Display name mapping for regulation types
    reg_name_map = {
        "all": "All",
        "up": "Upregulated",
        "down": "Downregulated"
    }
    
    # Process each combination of regulation type and database in parallel
    def create_single_plot(reg_type, db):
        # Create file path using the enrichment results base directory
        sanitized_disease = disease_name.replace(' ', '_')
        results_dir = os.path.join(enrich_results_base, f"{sanitized_disease}_{reg_type}_regulated")
        csv_file = os.path.join(results_dir, f"{db}_results.csv")
        
        # Check if file exists
        if not os.path.exists(csv_file):
            print(f"No results file found for {db} in {reg_type} regulated genes: {csv_file}")
            return None
        
        try:
            # Load the CSV file
            results_df = read_enrichment_results(csv_file)
            
            # Check if we have results
            if len(results_df) == 0:
                print(f"No results found in {csv_file}")
                return None
            
            # Sort by adjusted p-value and get top terms
            results_df = results_df.sort_values('Adjusted P-value').head(enrich_top_n)
            
            # Initialize empty list for gene counts with correct length
            gene_counts = []
            
            # Parse genes and calculate counts
            for _, row in results_df.iterrows():
                gene_str = row['Genes']
                
                # Parse the gene string - could be in multiple formats
                if isinstance(gene_str, str):
                    if gene_str.startswith('[') and gene_str.endswith(']'):
                        # Handle string representation of a list
                        # Remove brackets, split by comma and quote, filter out empty strings
                        genes = [g.strip("' \"") for g in gene_str.strip('[]').replace("'", "").split(',') if g.strip()]
                        count = len(genes)
                    else:
                        # Handle plain comma-separated string
                        count = len(gene_str.split(','))
                else:
                    # Unknown format
                    count = 0
                    print(f"Warning: Unexpected gene format for {row['Term']}")
                
                gene_counts.append(count)
            
            # Create DataFrame AFTER we have all values ready
            plot_df = pd.DataFrame({
                'Term': results_df['Term'].tolist(),
                'PValue': results_df['Adjusted P-value'].tolist(),
                'Count': gene_counts  # Now this will have the same length
            })
            
            # Reverse order so most significant is at the top
            plot_df = plot_df.iloc[::-1].reset_index(drop=True)
            
            # Create the plot
            plt.figure(figsize=(12, min(12, 2 + 0.4 * len(plot_df))))
            
            # Use log transformation for p-values
            log_transform = lambda x: -np.log10(x)
            transformed_values = log_transform(plot_df['PValue'])
            
            # Create horizontal bars
            bars = plt.barh(
                y=np.arange(len(plot_df)),
                width=transformed_values,
                height=0.65,
                color='#9e9ac8',
                edgecolor='#6a51a3',
                alpha=0.7,
                linewidth=1.5
            )
            
            # Add count annotations inside bars
            for i, (_, row) in enumerate(plot_df.iterrows()):
                count = row['Count']
                pvalue = row['PValue']
                transformed = log_transform(pvalue)
                
                # Add count as text inside the bar
                plt.text(
                    transformed/2,  # Position in the middle of the visible part
                    i,
                    f"Count: {count}",
                    ha='center',
                    va='center',
                    color='#3f007d',
                    fontweight='bold',
                    fontsize=11
                )
                
                # Add p-value at the end of the bar
                plt.text(
                    transformed * 1.05,  # Position just after the bar
                    i,
                    f"p={pvalue:.2e}",
                    ha='left',
                    va='center',
                    fontsize=10,
                    fontweight='medium',
                    color='#4a4a4a'
                )
            
            # Truncate long terms
            y_labels = []
            for term in plot_df['Term']:
                if len(term) > 60:
                    y_labels.append(term[:57] + '...')
                else:
                    y_labels.append(term)
            
            # Set y-tick labels to pathway terms with better formatting
            plt.yticks(np.arange(len(plot_df)), y_labels, fontsize=11, fontweight='medium')
            
            # Format x-axis
            plt.xlabel('-log10(Adjusted P-Value)', fontsize=12, fontweight='medium')
            plt.ylabel('Pathway', fontsize=12, fontweight='medium')
            
            # Get nice names for titles
            db_display = db_name_map.get(db, db)
            reg_display = reg_name_map.get(reg_type, reg_type)
            
            plt.title(f'Top {enrich_top_n} Enriched {db_display} Terms\n{reg_display} Genes - {disease_name}', 
                    fontsize=14, fontweight='bold', pad=15)
            
            # Remove top and right spines for cleaner look
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_linewidth(1.2)
            plt.gca().spines['bottom'].set_linewidth(1.2)
            
            # Add a grid for easier reading
            plt.grid(axis='x', linestyle='--', alpha=0.3, linewidth=0.8)
            
            # Adjust layout
            plt.tight_layout(pad=2.0)
            
            # Save the plot
            safe_db = db.replace('/', '_')
            plot_filename = f"{plot_enrich_dir}/{safe_db}_{reg_type}_regulated.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            
            # Also save as PDF for publication quality
            pdf_filename = f"{plot_enrich_dir}/{safe_db}_{reg_type}_regulated.pdf"
            plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
            
            plt.close()
            
            return f"Plot saved to {plot_filename} and {pdf_filename}"
            
        except Exception as e:
            error_msg = f"Error creating plot for {db} in {reg_type} regulated genes: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None
    
    # Generate all combinations and process in parallel
    plot_combinations = [(reg_type, db) for reg_type in regulation_types for db in databases]
    
    print(f"Creating {len(plot_combinations)} enrichment plots in parallel...")
    plot_results = Parallel(n_jobs=-1)(
        delayed(create_single_plot)(reg_type, db) 
        for reg_type, db in plot_combinations
    )
    
    # Print successful results
    successful_plots = [result for result in plot_results if result is not None]
    for result in successful_plots:
        print(result)
