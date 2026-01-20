import os
import sys
import gc
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
import scipy
import scipy.stats
from scipy import stats
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
import warnings

# Suppress matplotlib threading warnings for parallel plotting
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.path_config import get_path

BMG_DIR = get_path('external.biomedgraphica_dir', absolute=True)
# Use relative path but resolve it once at module load time to avoid issues with parallel processes
OUTPUT_DIR = get_path('data.dataset_outputs', absolute=True, create=True)


def create_directories_parallel(directories):
    """Create multiple directories in parallel."""
    def create_dir(directory):
        os.makedirs(directory, exist_ok=True)
        return directory
    
    Parallel(n_jobs=-1)(delayed(create_dir)(directory) for directory in directories)


def omic_analysis(disease_name: str, data_dict: dict, enable_plotting: bool = True, session_dir: str = None) -> dict:
    """
    Perform omic analysis on the input data dictionary.

    Args:
        disease_name (str): The name of the disease for which the analysis is performed.
        data_dict (dict): A dictionary containing omic data.
        enable_plotting (bool): Whether to enable enrichment plotting (default: True).
        session_dir (str): Optional session directory path to save results. If None, uses OUTPUT_DIR.

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
    
    # Load transcriptomics and proteomics data
    transcriptomics_data_path = os.path.join(BMG_DIR, "Entity/Transcript/BioMedGraphica_Conn_Transcript.csv")
    proteomics_data_path = os.path.join(BMG_DIR, "Entity/Protein/BioMedGraphica_Conn_Protein_Display_Name.csv")
    # Load the transcriptomics and proteomics data
    transcriptomics_data = pd.read_csv(transcriptomics_data_path).rename(columns={'HGNC_Symbol': 'Name'})
    proteomics_data = pd.read_csv(proteomics_data_path).rename(columns={'BMG_Protein_Name': 'Name'})
    omics_data = pd.concat([transcriptomics_data, proteomics_data], axis=0)
    print(f"Omics data shape: {omics_data.shape}")
    
    # Clear intermediate data
    del transcriptomics_data, proteomics_data
    gc.collect()

    # Transpose the combined matrices and convert to DataFrame and rename the columns
    # Process one at a time to reduce peak memory
    print("Creating disease DataFrame...")
    combined_disease_df = pd.DataFrame(combined_disease_matrix.T, columns=[f'ds_sample_{i}' for i in range(combined_disease_matrix.shape[0])])
    del combined_disease_matrix  # Free original matrix immediately
    gc.collect()
    
    print("Creating normal DataFrame...")
    combined_normal_df = pd.DataFrame(combined_normal_matrix.T, columns=[f'ns_sample_{i}' for i in range(combined_normal_matrix.shape[0])])
    del combined_normal_matrix  # Free original matrix immediately
    gc.collect()
    
    # Convert to list to avoid index alignment issues
    biomedgraphica_ids = omics_data['BioMedGraphica_Conn_ID'].tolist()
    # Insert a new column 'BioMedGraphica_Conn_ID' as the first column in both DataFrames
    combined_disease_df.insert(0, 'BioMedGraphica_Conn_ID', biomedgraphica_ids)
    combined_normal_df.insert(0, 'BioMedGraphica_Conn_ID', biomedgraphica_ids)
    # Filter out only the transcript data
    combined_disease_df = combined_disease_df[combined_disease_df['BioMedGraphica_Conn_ID'].str.contains("BMGC_TS")]
    combined_normal_df = combined_normal_df[combined_normal_df['BioMedGraphica_Conn_ID'].str.contains("BMGC_TS")]
    gc.collect()
    
    # Map the IDs to gene names
    mapping_dict = dict(zip(omics_data['BioMedGraphica_Conn_ID'], omics_data['Name']))
    del omics_data  # Free omics_data after extracting what we need
    gc.collect()
    
    combined_disease_df['Name'] = combined_disease_df['BioMedGraphica_Conn_ID'].map(mapping_dict)
    combined_normal_df['Name'] = combined_normal_df['BioMedGraphica_Conn_ID'].map(mapping_dict)
    combined_disease_df = combined_disease_df.drop(columns=['BioMedGraphica_Conn_ID'])
    combined_normal_df = combined_normal_df.drop(columns=['BioMedGraphica_Conn_ID'])
    combined_disease_df = combined_disease_df[['Name'] + [col for col in combined_disease_df.columns if col != 'Name']]
    combined_normal_df = combined_normal_df[['Name'] + [col for col in combined_normal_df.columns if col != 'Name']]
    
    # Aggregate with mean average across the rows with the same gene name
    combined_disease_df = combined_disease_df.groupby('Name').mean().reset_index()
    combined_normal_df = combined_normal_df.groupby('Name').mean().reset_index()
    gc.collect()
    
    # Print the shapes of the aggregated DataFrames
    print(f"Disease DataFrame shape after aggregation: {combined_disease_df.shape}")
    print(f"Normal DataFrame shape after aggregation: {combined_normal_df.shape}")

    # Save analysis results to outputs directory
    outputs_dir = get_path('outputs.omic_analysis', absolute=True, create=True)
    combined_disease_df.to_csv(os.path.join(outputs_dir, "foranalysis_combined_male_df_pdac.csv"), index=False)
    combined_normal_df.to_csv(os.path.join(outputs_dir, "foranalysis_combined_female_df_pdac.csv"), index=False)
    print(f"results saved to {outputs_dir}")
    # Create all required directories in parallel
    # Use session directory if provided, otherwise use the default OUTPUT_DIR
    if session_dir:
        base_dir = session_dir
        print(f"Using session directory for analysis results: {session_dir}")
    else:
        base_dir = OUTPUT_DIR
        print(f"Using default output directory for analysis results: {OUTPUT_DIR}")
    
    de_output_dir = os.path.join(base_dir, "differential_expression")
    volcano_dir = os.path.join(base_dir, "volcano_plots")
    enrich_output_dir = os.path.join(base_dir, "enrichment_results")
    plot_enrich_dir = os.path.join(enrich_output_dir, "enrichment_plots")
    
    directories_to_create = [de_output_dir, volcano_dir, enrich_output_dir, plot_enrich_dir]
    create_directories_parallel(directories_to_create)

    # Perform the differential expression analysis
    significant_genes, result_df = perform_unpaired_differential_expression(
        disease_df=combined_disease_df, 
        normal_df=combined_normal_df, 
        p_value_threshold=0.01,
        log2fc_threshold=1.5,
        sig_top_n=1000,  # change to 1000
        disease=disease_name,
        de_output_dir=de_output_dir,
        n_jobs=-1  # Use all available cores for maximum speed
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
                config['highlight_top_n']
            )
        return "Volcano plots completed"
    
    def run_enrichment_analysis():
        return perform_enrichment_analysis(
            significant_genes=significant_genes, 
            disease_name=disease_name, 
            enrich_output_dir=enrich_output_dir,
            fast_mode=False  # Disable fast mode by default
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
    volcano_result = create_volcano_plots()
    print(f"  - Volcano plots: {volcano_result}")
    
    # Execute enrichment analysis
    enrichment_results = run_enrichment_analysis()
    print(f"  - Enrichment analysis: completed")
    
    # Execute enrichment plotting if enabled
    if enable_plotting:
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
        "enrichment_plots_dir": plot_enrich_dir
    }

    return data_and_analysis_dict
        

def perform_unpaired_differential_expression(disease_df, normal_df, 
                                    p_value_threshold=0.05, log2fc_threshold=1.5, 
                                    sig_top_n=100, n_jobs=16, disease="Disease",
                                    de_output_dir=None) -> tuple[dict, pd.DataFrame]:
    """
    Perform unpaired non-parametric differential expression analysis using Mann–Whitney U test.

    Args:
        disease_df (pd.DataFrame): Disease group, rows = genes, columns = samples, 'Name' column for gene names.
        normal_df (pd.DataFrame): Control group, same structure as disease_df.
        p_value_threshold (float): Adjusted p-value threshold for significance.
        log2fc_threshold (float): Log2 fold change threshold for biological significance.
        sig_top_n (int): Number of top significant genes to return.
        n_jobs (int): Number of parallel jobs. -1 uses all cores.
        disease (str): Name of the disease for plot titles.
        de_output_dir (str): Directory to save differential expression results.

    Returns:
        Tuple[dict, pd.DataFrame]: Dictionary with significant genes and a DataFrame of unpaired DE results.
    """
    # Extract gene names and expression matrices
    gene_names = disease_df['Name'].tolist()
    assert gene_names == normal_df['Name'].tolist(), "Gene order mismatch between disease and control."

    disease_matrix = disease_df.drop(columns=['Name']).values
    control_matrix = normal_df.drop(columns=['Name']).values

    # Check the min and max values in the matrices
    print(f"Disease matrix shape: {disease_matrix.shape}, min: {np.min(disease_matrix)}, max: {np.max(disease_matrix)}")
    print(f"Control matrix shape: {control_matrix.shape}, min: {np.min(control_matrix)}, max: {np.max(control_matrix)}")
    # import pdb; pdb.set_trace()  # Debugging breakpoint to inspect matrices
    
    # Pre-filter genes to reduce computational load
    print("Pre-filtering genes to reduce computational load...")
    
    # Calculate basic statistics for filtering
    disease_mean = np.mean(disease_matrix, axis=1)
    control_mean = np.mean(control_matrix, axis=1)
    disease_std = np.std(disease_matrix, axis=1)
    control_std = np.std(control_matrix, axis=1)
    
    # Filter criteria:
    # 1. Remove genes with very low expression in both groups
    min_expression_threshold = 0.1
    # 2. Remove genes with no variance (constant expression)
    min_variance_threshold = 1e-6
    # 3. Keep genes with some difference between groups
    
    keep_mask = (
        ((disease_mean > min_expression_threshold) | (control_mean > min_expression_threshold)) &
        ((disease_std > min_variance_threshold) | (control_std > min_variance_threshold))
    )
    
    # Apply filtering
    filtered_gene_names = np.array(gene_names)[keep_mask]
    filtered_disease_matrix = disease_matrix[keep_mask, :]
    filtered_control_matrix = control_matrix[keep_mask, :]
    
    genes_before = len(gene_names)
    genes_after = len(filtered_gene_names)
    print(f"Filtered from {genes_before} to {genes_after} genes ({genes_before - genes_after} removed, {genes_after/genes_before*100:.1f}% retained)")
    
    # Use parallel processing to calculate p-values on filtered data
    print(f"Calculating differential expression using {n_jobs} parallel jobs...")
    p_values_filtered = parallel_mannwhitney_optimized(filtered_disease_matrix, filtered_control_matrix, n_jobs)
    
    # Create full p-values array (filtered genes get their calculated p-value, others get 1.0)
    p_values = np.ones(len(gene_names))
    p_values[keep_mask] = p_values_filtered
    
    # FDR correction
    print("Applying FDR correction...")
    fdr = multipletests(p_values, method='fdr_bh')[1]

    # Calculate fold changes and effect sizes (vectorized)
    print("Calculating fold changes and effect sizes...")
    disease_mean = np.mean(disease_matrix, axis=1)
    control_mean = np.mean(control_matrix, axis=1)
    
    # Add small constant to avoid division by zero
    epsilon = 1e-8
    fold_changes = np.log2((disease_mean + epsilon) / (control_mean + epsilon))
    
    # Vectorized Cohen's d calculation for effect size
    disease_var = np.var(disease_matrix, axis=1, ddof=1)
    control_var = np.var(control_matrix, axis=1, ddof=1)
    n_disease = disease_matrix.shape[1]
    n_control = control_matrix.shape[1]
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n_disease - 1) * disease_var + (n_control - 1) * control_var) / 
                        (n_disease + n_control - 2))
    
    # Cohen's d effect size
    effect_sizes = (disease_mean - control_mean) / (pooled_std + epsilon)

    # Result table
    result_df = pd.DataFrame({
        'Name': gene_names,
        'log2_fold_change': fold_changes,
        'effect_size': effect_sizes,
        'p_value': p_values,
        'FDR': fdr
    })
    result_df['is_significant'] = result_df['FDR'] < p_value_threshold
    result_df['abs_log2_fc'] = np.abs(result_df['log2_fold_change'])

    # Debugging for genes with NaN fold change
    nan_genes = result_df[result_df['log2_fold_change'].isna()]
    if len(nan_genes) > 0:
        print(f"Found {len(nan_genes)} genes with NaN fold change values")
        for idx, row in nan_genes.head(5).iterrows():
            gene_name = row['Name']
            gene_idx = gene_names.index(gene_name)
            d_values = disease_matrix[gene_idx, :]
            c_values = control_matrix[gene_idx, :]
            print(f"Gene {gene_name}: Disease median={np.median(d_values)}, Control median={np.median(c_values)}")
            print(f"  Disease values: min={np.min(d_values)}, max={np.max(d_values)}")
            print(f"  Control values: min={np.min(c_values)}, max={np.max(c_values)}")

    # Add this before your debugging code
    zero_fc_genes = result_df[(result_df['log2_fold_change'] == 0.0) & (result_df['p_value'] < 1e-10)]
    if len(zero_fc_genes) > 0:
        print(f"Found {len(zero_fc_genes)} genes with zero fold change but very low p-values")
        for idx, row in zero_fc_genes.head(5).iterrows():
            gene_name = row['Name']
            gene_idx = gene_names.index(gene_name)
            d_values = disease_matrix[gene_idx, :]
            c_values = control_matrix[gene_idx, :]
            print(f"Gene {gene_name}:")
            print(f"  Disease values: {d_values[:5]}... (mean={np.mean(d_values)}, median={np.median(d_values)})")
            print(f"  Control values: {c_values[:5]}... (mean={np.mean(c_values)}, median={np.median(c_values)})")
            # Count zeros in each group
            d_zeros = np.sum(d_values == 0)
            c_zeros = np.sum(c_values == 0)
            print(f"  Disease zeros: {d_zeros}/{len(d_values)} ({d_zeros/len(d_values)*100:.1f}%)")
            print(f"  Control zeros: {c_zeros}/{len(c_values)} ({c_zeros/len(c_values)*100:.1f}%)")

    # Filter significant genes
    sig_genes = result_df[result_df['is_significant']]

    # Sort by different criteria
    sig_by_fdr = sig_genes.sort_values('FDR').head(sig_top_n)
    sig_by_fc = sig_genes.sort_values('abs_log2_fc', ascending=False).head(sig_top_n)

    # Separate upregulated and downregulated genes
    sig_up = sig_genes[sig_genes['log2_fold_change'] > 0].sort_values('p_value').head(sig_top_n)
    sig_down = sig_genes[sig_genes['log2_fold_change'] < 0].sort_values('p_value').head(sig_top_n)

    # Save all result files in parallel
    save_tasks = [
        (result_df, os.path.join(de_output_dir, "unpaired_differential_expression_results.csv")),
        (sig_by_fdr, os.path.join(de_output_dir, "significant_genes_by_fdr.csv")),
        (sig_by_fc, os.path.join(de_output_dir, "significant_genes_by_fc.csv")),
        (sig_up, os.path.join(de_output_dir, "significant_upregulated_genes.csv")),
        (sig_down, os.path.join(de_output_dir, "significant_downregulated_genes.csv"))
    ]
    
    def save_dataframe(df, filepath):
        df.to_csv(filepath, index=False)
        return f"Saved {filepath}"
    
    print("Saving differential expression results in parallel...")
    Parallel(n_jobs=-1)(
        delayed(save_dataframe)(df, filepath) for df, filepath in save_tasks
    )
    
    # Return both up and down-regulated gene dataframes
    return {"all": sig_by_fdr, "up": sig_up, "down": sig_down}, result_df


def parallel_mannwhitney(disease_matrix, control_matrix, n_jobs=-1):
    """
    Optimized parallel Mann-Whitney U test with batch processing
    """
    n_genes = disease_matrix.shape[0]
    n_disease = disease_matrix.shape[1]
    n_control = control_matrix.shape[1]
    
    if n_jobs == -1:
        n_cores = os.cpu_count()
    else:
        n_cores = min(n_jobs, os.cpu_count())
    
    # Calculate optimal batch size for better load balancing
    batch_size = max(1, n_genes // (n_cores * 4))  # 4 batches per core
    
    print(f"Processing {n_genes} genes in batches of {batch_size} using {n_cores} cores...")
    
    def compute_batch_mannwhitney(batch_indices):
        """Compute Mann-Whitney U test for a batch of genes"""
        batch_p_values = []
        
        for i in batch_indices:
            disease_values = disease_matrix[i, :]
            control_values = control_matrix[i, :]
            
            # Skip if identical distributions (early termination)
            if np.array_equal(disease_values, control_values):
                batch_p_values.append(1.0)
                continue
            
            try:
                # Use asymptotic method for large samples (faster)
                if len(disease_values) > 20 and len(control_values) > 20:
                    _, p_value = stats.ranksums(disease_values, control_values)
                else:
                    _, p_value = stats.mannwhitneyu(disease_values, control_values, alternative='two-sided')
                batch_p_values.append(p_value)
            except Exception:
                # Fallback for edge cases
                batch_p_values.append(1.0)
        
        return batch_p_values
    
    # Create batches
    gene_indices = np.arange(n_genes)
    batches = [gene_indices[i:i+batch_size] for i in range(0, n_genes, batch_size)]
    
    # Run parallel computation with reduced verbosity
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        batch_results = Parallel(n_jobs=n_cores, verbose=0)(
            delayed(compute_batch_mannwhitney)(batch) for batch in batches
        )
    
    # Flatten results
    p_values = [p for batch in batch_results for p in batch]
    
    return np.array(p_values)

def fast_mannwhitney_vectorized(x, y):
    """
    Fast vectorized Mann-Whitney U test for multiple genes
    Uses ranking approach optimized with numpy for speed
    """
    n_genes, n_x = x.shape
    n_y = y.shape[1]
    n_total = n_x + n_y
    
    # Combine samples for ranking
    combined = np.concatenate([x, y], axis=1)
    
    # Get ranks for each gene across all samples
    ranks = np.empty_like(combined)
    for i in range(n_genes):
        ranks[i] = scipy.stats.rankdata(combined[i])
    
    # Sum ranks for x group
    rank_sum_x = np.sum(ranks[:, :n_x], axis=1)
    
    # Calculate U statistic
    U1 = rank_sum_x - n_x * (n_x + 1) / 2
    U2 = n_x * n_y - U1
    U = np.minimum(U1, U2)
    
    # Normal approximation for p-values (faster than exact)
    mu = n_x * n_y / 2
    sigma = np.sqrt(n_x * n_y * (n_x + n_y + 1) / 12)
    
    # Continuity correction
    z = (U - mu + 0.5) / sigma
    p_values = 2 * scipy.stats.norm.cdf(z)
    
    return p_values

def parallel_mannwhitney_optimized(disease_matrix, control_matrix, n_jobs=-1):
    """
    Optimized parallel Mann-Whitney U test with choice of algorithms
    """
    n_genes = disease_matrix.shape[0]
    n_disease = disease_matrix.shape[1]
    n_control = control_matrix.shape[1]
    
    print(f"Running optimized Mann-Whitney tests on {n_genes} genes...")
    
    # Choose algorithm based on data size
    if n_genes < 5000:
        # For smaller datasets, use the exact method
        print("Using exact Mann-Whitney method for high precision...")
        return parallel_mannwhitney(disease_matrix, control_matrix, n_jobs)
    
    # For larger datasets, try vectorized approach first
    if n_genes > 20000 and (n_disease + n_control) < 100:
        print("Using vectorized Mann-Whitney method for speed...")
        try:
            return fast_mannwhitney_vectorized(disease_matrix, control_matrix)
        except Exception as e:
            print(f"Vectorized method failed ({e}), falling back to parallel method...")
    
    # Default to the batch processing method
    print("Using batch-processed Mann-Whitney method...")
    return parallel_mannwhitney(disease_matrix, control_matrix, n_jobs)

def create_volcano_plot(result_df, p_value_threshold=0.025, log2fc_threshold=1.5, 
                         save_path=None, plot_title=None, highlight_top_n=50):
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
                               enrich_output_dir=None, fast_mode=True):
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
    
    # Process each set of genes (all, up, down) in parallel
    def process_gene_set(set_name, gene_df):
        if gene_df.empty:
            print(f"No genes in the {set_name} regulated set to analyze")
            return set_name, None
            
        gene_list = gene_df['Name'].tolist()
        print(f"\nPerforming enrichment analysis on {len(gene_list)} {set_name}-regulated genes...")
        
        # Create a sanitized ID for the file paths
        sample_id = f"{disease_name.replace(' ', '_')}_{set_name}_regulated"
        
        # Store disease name for later use in results
        set_results = {"disease": disease_name, "gene_count": len(gene_list)}
        
        # Run enrichment analysis on this gene set with specified databases
        enrichr_results = enrichr_analysis(gene_list, sample_id, enrich_output_dir, databases=all_dbs)
        
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
    
    return results

def enrichr_analysis(gene_list, sample_id, enrich_output_dir=None, databases=None):
    """
    Performs enrichment analysis using the Enrichr API with flexible database selection.
    
    Args:
        gene_list (list): A list of gene symbols
        sample_id (str): Sample identifier used for naming the output directory
        enrich_output_dir (str): Base directory to save results
        databases (list): List of databases to query (if None, uses all available)
        
    Returns:
        dict: Dictionary with enrichment results
    """
    # Create sample-specific directory
    sample_dir = os.path.join(enrich_output_dir, sample_id)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Default databases if none provided
    if databases is None:
        databases = [
            # Pathway databases
            'GO_Biological_Process_2021',
            'GO_Molecular_Function_2021',
            'GO_Cellular_Component_2021',
            'KEGG_2021_Human',
            'Reactome_2022',
            'WikiPathways_2019_Human',
            'MSigDB_Hallmark_2020',
            
            # Disease-related databases
            'DisGeNET',
            'OMIM_Disease',
            'OMIM_Expanded',
            'Human_Phenotype_Ontology',
            'Jensen_DISEASES',
            'GTEx_Tissue_Expression_Down',
            'GTEx_Tissue_Expression_Up',
        ]
    
    # Step 1: Submit gene list to Enrichr
    genes_str = '\n'.join(gene_list)
    upload_url = 'https://maayanlab.cloud/Enrichr/addList'
    
    print(f"Submitting {len(gene_list)} genes to Enrichr API...")
    try:
        response = requests.post(upload_url, files={'list': (None, genes_str)}, 
                               data={'description': f'DEA gene set - {sample_id}'})
        
        if not response.ok:
            print(f"Error submitting gene list: {response.status_code}")
            return None
        
        user_list_id = response.json().get('userListId')
        print(f"Gene list submitted successfully with ID: {user_list_id}")
        
    except Exception as e:
        print(f"Error connecting to Enrichr API: {e}")
        print("Saving gene list locally for manual analysis")
        gene_list_path = os.path.join(sample_dir, "gene_list.txt")
        with open(gene_list_path, 'w') as f:
            f.write(genes_str)
        print(f"Gene list saved to {gene_list_path}")
        return None
    
    # Step 2: Retrieve enrichment results for specified databases in parallel
    def query_database(database):
        print(f"Retrieving results for {database}...")
        
        query_url = f'https://maayanlab.cloud/Enrichr/enrich?userListId={user_list_id}&backgroundType={database}'
        try:
            # Reduce delay further to speed up API calls
            time.sleep(0.1)  # Reduced from 0.2 seconds to 0.1 seconds
            response = requests.get(query_url)
            
            if not response.ok:
                print(f"Error retrieving results for {database}: {response.status_code}")
                return database, None
                
            results = response.json()
            
            # Save individual database results as CSV
            if results and database in results:
                # Convert to DataFrame
                df = pd.DataFrame(results[database])
                
                if not df.empty:
                    # Rename columns for clarity
                    if len(df.columns) >= 9:  # Enrichr typically returns 9+ columns
                        df.columns = [
                            'Rank', 'Term', 'P-value', 'Odds Ratio', 'Combined Score',
                            'Genes', 'Adjusted P-value', 'Old P-value', 'Old Adjusted P-value'
                        ]
                    
                    # Sort by adjusted p-value
                    if 'Adjusted P-value' in df.columns:
                        df = df.sort_values('Adjusted P-value')
                    
                    # Save top 50 results
                    safe_db_name = database.replace('/', '_')
                    output_file = f"{sample_dir}/{safe_db_name}_results.csv"
                    df.head(50).to_csv(output_file, index=False)
                    print(f"  ✓ Saved top results to {output_file}")
            
            return database, results
                
        except Exception as e:
            print(f"Error processing {database}: {e}")
            return database, None
    
    # Process databases sequentially with threading to avoid nested parallelism segfaults
    print("Querying enrichment databases...")
    parallel_db_results = []
    for database in databases:
        result = query_database(database)
        parallel_db_results.append(result)
    
    # Collect results
    all_results = {}
    for database, results in parallel_db_results:
        if results is not None:
            all_results[database] = results
    
    # Save all results as JSON
    with open(f"{sample_dir}/all_enrichment_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create summary files
    create_enrichment_summary(gene_list, all_results, sample_id, sample_dir, databases)
    
    return all_results

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
    with open(summary_file, 'w') as f:
        f.write(f"ENRICHMENT ANALYSIS SUMMARY FOR {sample_id}\n")
        f.write(f"Number of genes analyzed: {len(gene_list)}\n\n")
        
        # Pathway section
        f.write("=" * 80 + "\n")
        f.write("PATHWAY ENRICHMENT RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for database in pathway_dbs:
            if database in enrichr_results and database in enrichr_results[database]:
                results = enrichr_results[database][database]
                if results:
                    top_terms = results[:5]
                    
                    f.write(f"\n{database} Top 5 Terms:\n")
                    
                    for term in top_terms:
                        term_name = term[1]
                        p_value = term[2]
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
                if results:
                    top_terms = results[:5]
                    
                    f.write(f"\n{database} Top 5 Terms:\n")
                    
                    for term in top_terms:
                        term_name = term[1]
                        p_value = term[2]
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
            results_df = pd.read_csv(csv_file)
            
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