"""
Omic Fetch and Analysis Workflow

This workflow performs:
1. Named Entity Recognition (NER) to extract disease and cell type
2. Data retrieval using CellTOSGDataLoader with soft matching
3. Differential expression analysis (for disease queries with normal/disease labels)
4. KEGG pathway enrichment analysis
5. Enrichment plotting (via R script)
6. Returns top genes and results for agent reflection

When no exact match is found, provides suggestions using:
- Fuzzy matching (difflib SequenceMatcher)
- Substring matching
- BiomedBERT semantic similarity (via microservice)

Based on the working CellTOSGDataLoader example pattern.
"""

from ner_tool import ner
from omic_analysis_components import omic_analysis
from subprocess_r import run_r_script

import sys
import os
import time
import gc
import zipfile
from difflib import get_close_matches
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.path_config import get_path

# Add the OmniCellTOSG directory to path to import the new loader
omnicell_root = get_path('external.omnicell_root', absolute=True)
sys.path.insert(0, omnicell_root)
from CellTOSG_Loader_new import CellTOSGDataLoader, CellTOSGSubsetBuilder

# Global cache for query builder
_query_builder = None

def _get_query_builder():
    """Get or create the global query builder instance."""
    global _query_builder
    if _query_builder is None:
        data_root = get_path('external.omnicell_data_root', absolute=True)
        _query_builder = CellTOSGSubsetBuilder(root=data_root)
    return _query_builder

def get_suggestions(conditions: dict, n_matches: int = 5) -> str:
    """
    Get similar terms from the database for failed query conditions.
    Uses difflib.get_close_matches for fuzzy string matching.
    Returns a formatted message with suggestions.
    """
    query_builder = _get_query_builder()
    field_map = {"disease": "disease", "cell_type": "cell_type", "tissue_general": "tissue_general"}
    suggestions = {}
    
    for field, value in conditions.items():
        resolved = field_map.get(field, field)
        try:
            available = query_builder.available_conditions(include_fields=[resolved], max_uniques=None)
            if resolved in available.get("unique_values", {}):
                values = available["unique_values"][resolved]["values"]
                values_map = {v.lower(): v for v in values}
                matches = get_close_matches(value.lower(), list(values_map.keys()), n=n_matches, cutoff=0.3)
                suggestions[field] = [values_map[m] for m in matches]
        except:
            pass
    
    # Format message
    lines = [f"  {field}: '{conditions[field]}' -> try {sugg}" for field, sugg in suggestions.items() if sugg]
    if lines:
        return "Similar terms in database:\n" + "\n".join(lines)
    return "No similar terms found."


def compress_expression_matrix(npy_path: str) -> bool:
    """
    Compress expression_matrix.npy using lrzip with ZPAQ compression (-z) and remove original.
    This should be called immediately after CellTOSGDataLoader creates the file.
    
    Args:
        npy_path: Path to the expression_matrix.npy file
        
    Returns:
        bool: True if successful, False otherwise
    """
    import subprocess
    import shutil
    
    if not os.path.exists(npy_path):
        return False
    
    lrzip_path = shutil.which('lrzip')
    if not lrzip_path:
        print(f"[Cleanup] lrzip not found, skipping compression of {os.path.basename(npy_path)}")
        return False
    
    try:
        original_size = os.path.getsize(npy_path)
        lrz_path = npy_path + '.lrz'
        
        print(f"[Cleanup] Compressing {os.path.basename(npy_path)} with lrzip ZPAQ...")
        
        cmd = ['lrzip', '-z', '-q', '-o', lrz_path, npy_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[Cleanup] lrzip failed: {result.stderr}")
            return False
        
        compressed_size = os.path.getsize(lrz_path)
        os.remove(npy_path)
        
        print(f"[Cleanup] Compressed and removed: {os.path.basename(npy_path)} "
              f"({original_size/(1024*1024):.1f}MB -> {compressed_size/(1024*1024):.1f}MB, "
              f"{compressed_size/original_size*100:.1f}%)")
        return True
        
    except Exception as e:
        print(f"[Cleanup] Error compressing {npy_path}: {e}")
        return False


def compress_and_cleanup_npy_files(session_dir: str, remove_after_zip: bool = True) -> dict:
    """
    Compress large .npy files in session directory using lrzip with ZPAQ compression (-z)
    for extreme compression ratios (~15x for expression matrices).
    
    Args:
        session_dir: Directory containing .npy files to compress
        remove_after_zip: Whether to remove the original .npy files after compression
        
    Returns:
        dict: Summary of compression results
    """
    import subprocess
    import shutil
    
    if not session_dir or not os.path.exists(session_dir):
        return {"success": False, "message": "Session directory not found"}
    
    # Check if lrzip is available
    lrzip_path = shutil.which('lrzip')
    if not lrzip_path:
        print("[Cleanup] lrzip not found, falling back to standard zip compression")
        return _compress_with_zip(session_dir, remove_after_zip)
    
    # Find all .npy files recursively (in case they're in subdirectories)
    npy_files = []
    for root, dirs, files in os.walk(session_dir):
        for f in files:
            if f.endswith('.npy'):
                npy_files.append(os.path.join(root, f))
    
    if not npy_files:
        return {"success": True, "message": "No .npy files to compress", "files_processed": 0}
    
    print(f"[Cleanup] Found {len(npy_files)} .npy files to compress")
    
    results = {
        "success": True,
        "files_processed": 0,
        "total_original_size": 0,
        "total_compressed_size": 0,
        "files": []
    }
    
    for npy_path in npy_files:
        try:
            original_size = os.path.getsize(npy_path)
            results["total_original_size"] += original_size
            
            # Create lrzip file with ZPAQ compression (-z flag)
            lrz_path = npy_path + '.lrz'
            
            print(f"[Cleanup] Compressing {os.path.basename(npy_path)} with lrzip ZPAQ (this may take a few minutes)...")
            
            # Run lrzip with ZPAQ compression (-z) and quiet mode (-q)
            cmd = ['lrzip', '-z', '-q', '-o', lrz_path, npy_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"lrzip failed: {result.stderr}")
            
            compressed_size = os.path.getsize(lrz_path)
            results["total_compressed_size"] += compressed_size
            
            file_result = {
                "file": os.path.basename(npy_path),
                "original_size_mb": round(original_size / (1024 * 1024), 2),
                "compressed_size_mb": round(compressed_size / (1024 * 1024), 2),
                "compression_ratio": round(compressed_size / original_size * 100, 1) if original_size > 0 else 0
            }
            results["files"].append(file_result)
            
            # Remove original file if requested
            if remove_after_zip:
                os.remove(npy_path)
                print(f"[Cleanup] Compressed and removed: {os.path.basename(npy_path)} "
                      f"({file_result['original_size_mb']}MB -> {file_result['compressed_size_mb']}MB, "
                      f"{file_result['compression_ratio']}%)")
            else:
                print(f"[Cleanup] Compressed: {os.path.basename(npy_path)} "
                      f"({file_result['original_size_mb']}MB -> {file_result['compressed_size_mb']}MB)")
            
            results["files_processed"] += 1
            
        except Exception as e:
            print(f"[Cleanup] Error processing {npy_path}: {e}")
            results["files"].append({
                "file": os.path.basename(npy_path),
                "error": str(e)
            })
    
    # Summary
    if results["total_original_size"] > 0:
        overall_ratio = round(results["total_compressed_size"] / results["total_original_size"] * 100, 1)
        saved_mb = round((results["total_original_size"] - results["total_compressed_size"]) / (1024 * 1024), 2)
        print(f"[Cleanup] Total: {results['files_processed']} files, "
              f"saved {saved_mb}MB ({100 - overall_ratio}% reduction)")
        results["saved_mb"] = saved_mb
        results["overall_compression_ratio"] = overall_ratio
    
    return results


def omic_fetch_with_new_loader(fetch_dict: dict, output_dir: str):
    """
    Fetch omics data using the new CellTOSGDataLoader with soft matching support.
    
    Following the working example pattern exactly.
    
    Args:
        fetch_dict (dict): Dictionary containing extracted entities
        output_dir (str): Directory to save output files
        
    Returns:
        tuple: (X, Y, metadata, similar_terms, retrieval_success)
    """
    data_root = get_path('external.omnicell_data_root', absolute=True)
    
    cell_type = fetch_dict.get("cell type", None)
    disease_name = fetch_dict.get("disease", None)
    organ = fetch_dict.get("organ", None)
    tissue = fetch_dict.get("tissue", None)
    gender = fetch_dict.get("gender", None)
    
    print(f"\n{'='*60}")
    print(f"[Omic Fetch] Starting data retrieval")
    print(f"  Cell type: {cell_type}")
    print(f"  Disease: {disease_name}")
    print(f"  Organ: {organ}")
    print(f"  Tissue: {tissue}")
    print(f"  Gender: {gender}")
    print(f"{'='*60}\n")
    
    conditions = {}
    if cell_type:
        conditions["cell_type"] = cell_type
    if disease_name:
        conditions["disease"] = disease_name
    if organ:
        conditions["tissue_general"] = organ
    if tissue:
        conditions["tissue"] = tissue
    if gender:
        conditions["gender"] = gender
    
    if not conditions:
        print("[Omic Fetch] No valid conditions extracted from NER")
        return None, None, None, {}, False
    
    # Determine task and label_column based on query
    if disease_name:
        task = "disease"
        label_column = "disease"
    elif cell_type:
        task = "cell_type"
        label_column = "cell_type"
    else:
        task = "disease"
        label_column = "disease"
    
    # Enable stratified_balancing for disease task to get matched normal samples
    use_stratified_balancing = (task == "disease")
    
    try:
        print(f"[Omic Fetch] Creating CellTOSGDataLoader...")
        print(f"  root: {data_root}")
        print(f"  conditions: {conditions}")
        print(f"  task: {task}")
        print(f"  label_column: {label_column}")
        print(f"  stratified_balancing: {use_stratified_balancing}")
        sys.stdout.flush()
        
        dataset = CellTOSGDataLoader(
            root=data_root,
            conditions=conditions,
            task=task,
            label_column=label_column,
            sample_ratio=None,
            sample_size=int(1000),
            shuffle=False,
            stratified_balancing=use_stratified_balancing,
            extract_mode="inference",
            random_state=2025,
            train_text=False,
            train_bio=False,
            dataset_correction=None,
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
            return None, None, None, {}, False
        
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
        
        # Immediately compress and remove expression_matrix.npy (large file ~4GB)
        expression_matrix_path = os.path.join(output_dir, "expression_matrix.npy")
        if os.path.exists(expression_matrix_path):
            compress_expression_matrix(expression_matrix_path)
        
        return X, Y, metadata, similar_terms, True
        
    except ValueError as e:
        print(f"\n[Omic Fetch] No match: {str(e)}")
        return None, None, None, {}, False
    
    except Exception as e:
        print(f"\n[Omic Fetch] Unexpected issue: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, {}, False


def compute_top_genes(X, top_k=100):
    """Compute top K genes by mean expression."""
    if X is None or X.shape[0] == 0:
        return [], []
    
    mean_expr = np.mean(X, axis=0)
    top_k_indices = np.argsort(mean_expr)[-top_k:][::-1]
    top_k_values = mean_expr[top_k_indices]
    
    return top_k_indices.tolist(), top_k_values.tolist()


# Fixed number of top genes to return
TOP_K_GENES = 20

def omic_fetch_analysis_workflow(text=None, disease=None, cell_type=None,
                                 organ=None, tissue=None, gender=None, session_dir=None,
                                 enable_differential_expression=True, enable_plotting=True):
    """
    Perform the complete omic analysis workflow.
    
    Supports two input modes:
    1. Natural language: Provide text parameter for NER extraction
    2. Direct parameters: Provide specific parameters (disease, cell_type, etc.)
    
    Args:
        text: Natural language query for NER extraction
        disease, cell_type, organ, tissue, gender: Direct parameters
        session_dir: Directory to save all outputs (REQUIRED)
        enable_differential_expression: Run DE analysis if disease query (default True)
        enable_plotting: Run R plotting script (default True)
    """
    times = {}
    times['start'] = time.time()
    
    if session_dir is None:
        raise ValueError("session_dir must be provided")
    os.makedirs(session_dir, exist_ok=True)
    print(f"\n[Session] Output directory: {session_dir}")
    
    # ===========================================================================
    # STEP 1: Parse input (NER or direct parameters)
    # ===========================================================================
    if text is not None:
        print(f"\n{'='*70}")
        print(f"STEP 1: Named Entity Recognition (Text Mode)")
        print(f"{'='*70}")
        fetch_dict = ner(text)
        print(f"[NER] Extracted entities: {fetch_dict}")
        times['ner_end'] = time.time()
    else:
        print(f"\n{'='*70}")
        print(f"STEP 1: Direct Parameter Input (Function Mode)")
        print(f"{'='*70}")
        fetch_dict = {}
        if disease is not None:
            fetch_dict["disease"] = disease
        if cell_type is not None:
            fetch_dict["cell type"] = cell_type
        if organ is not None:
            fetch_dict["organ"] = organ
        if tissue is not None:
            fetch_dict["tissue"] = tissue
        if gender is not None:
            fetch_dict["gender"] = gender
        
        if not fetch_dict:
            print("[Input] No input provided.")
            return {
                "success": False,
                "message": "No input provided.",
                "similar_terms": {},
                "extracted_entities": {},
                "retrieval_success": False,
                "timing": {"total": 0.0}
            }
        
        print(f"[Input] Direct parameters: {fetch_dict}")
        times['ner_end'] = time.time()
    
    # ===========================================================================
    # STEP 2: Data Retrieval
    # ===========================================================================
    print(f"\n{'='*70}")
    print(f"STEP 2: Data Retrieval with Soft Matching")
    print(f"{'='*70}")
    
    X, Y, metadata, similar_terms, retrieval_success = omic_fetch_with_new_loader(fetch_dict, session_dir)
    times['fetch_end'] = time.time()
    
    if not retrieval_success or X is None:
        print(f"\n{'='*70}")
        print(f"WORKFLOW: No matching data found - Generating suggestions")
        print(f"{'='*70}")
        
        # Build conditions dict for suggestions
        conditions_for_matching = {}
        if fetch_dict.get("disease"):
            conditions_for_matching["disease"] = fetch_dict["disease"]
        if fetch_dict.get("cell type"):
            conditions_for_matching["cell_type"] = fetch_dict["cell type"]
        if fetch_dict.get("organ"):
            conditions_for_matching["tissue_general"] = fetch_dict["organ"]
        
        suggestions_message = ""
        if conditions_for_matching:
            try:
                suggestions_message = get_suggestions(conditions_for_matching, n_matches=5)
                print(suggestions_message)
            except Exception as e:
                suggestions_message = f"Could not generate suggestions: {e}"
        
        return {
            "success": False,
            "message": "No samples matched the specified conditions.",
            "similar_terms": similar_terms,
            "extracted_entities": fetch_dict,
            "retrieval_success": False,
            "suggestions": suggestions_message,
            "timing": {
                "ner": times['ner_end'] - times['start'],
                "fetch": times['fetch_end'] - times['ner_end'],
                "total": times['fetch_end'] - times['start']
            }
        }
    
    # ===========================================================================
    # STEP 3: Compute top genes by mean expression
    # ===========================================================================
    print(f"\n{'='*70}")
    print(f"STEP 3: Computing Top {TOP_K_GENES} Genes by Mean Expression")
    print(f"{'='*70}")
    
    top_gene_indices, top_gene_values = compute_top_genes(X, TOP_K_GENES)
    print(f"[Genes] Top {len(top_gene_indices)} genes computed")
    if len(top_gene_indices) > 0:
        print(f"  Top 5 gene indices: {top_gene_indices[:5]}")
        print(f"  Top 5 mean values: {[f'{v:.4f}' for v in top_gene_values[:5]]}")
    
    times['gene_end'] = time.time()
    
    if len(top_gene_indices) > 0:
        top_genes_df = pd.DataFrame({
            "rank": range(1, len(top_gene_indices) + 1),
            "gene_index": top_gene_indices,
            "mean_expression": top_gene_values
        })
        top_genes_path = os.path.join(session_dir, "top_genes_by_expression.csv")
        top_genes_df.to_csv(top_genes_path, index=False)
        print(f"[Genes] Saved to: {top_genes_path}")
    
    # ===========================================================================
    # STEP 4: Differential Expression Analysis (if disease query)
    # ===========================================================================
    disease_name = fetch_dict.get("disease", None)
    analysis_success = False
    analysis_paths = None
    top_genes_by_fdr = []
    
    if enable_differential_expression and disease_name and Y is not None:
        print(f"\n{'='*70}")
        print(f"STEP 4: Differential Expression Analysis")
        print(f"{'='*70}")
        
        # Check if we have both normal (0) and disease (1) labels
        unique_labels = np.unique(Y)
        has_normal = 0 in unique_labels
        has_disease = 1 in unique_labels
        
        # Count samples per label
        label_counts = {int(label): int(np.sum(Y == label)) for label in unique_labels}
        print(f"[DE] Label distribution: {label_counts}")
        
        # For DE analysis, we need at least 2 groups with samples
        if len(unique_labels) >= 2 and all(label_counts.get(l, 0) > 0 for l in unique_labels[:2]):
            # Get the two groups based on labels
            group0_count = np.sum(Y == 0)
            group1_count = np.sum(Y == 1) if 1 in unique_labels else 0
            
            print(f"[DE] Found {group0_count} samples in group 0 and {group1_count} samples in group 1")
            
            if group0_count > 0 and group1_count > 0:
                # Split data into two groups
                normal_omic_feature = X[Y == 0]
                disease_omic_feature = X[Y == 1]
                
                data_dict = {
                    "normal_omic_feature": normal_omic_feature,
                    "disease_omic_feature": disease_omic_feature,
                    "omic_label": Y
                }
                
                try:
                    data_and_analysis_dict = omic_analysis(
                        disease_name, 
                        data_dict, 
                        enable_plotting=enable_plotting,
                        session_dir=session_dir
                    )
                    analysis_success = True
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
                        gene_df = pd.read_csv(gene_file)
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
                    
                    del data_and_analysis_dict
                    
                except Exception as e:
                    print(f"[DE] Analysis failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                del normal_omic_feature, disease_omic_feature, data_dict
            else:
                print(f"[DE] Skipped: need samples in both groups for comparison")
                print(f"     NOTE: When querying a specific disease, all returned samples")
                print(f"           have that disease. DE analysis requires both disease")
                print(f"           AND normal samples from the same tissue context.")
        else:
            print(f"[DE] Skipped: need at least 2 groups with samples")
            print(f"     Labels found: {unique_labels}")
            print(f"     NOTE: For single-disease queries, all samples share the same label.")
    else:
        print(f"\n{'='*70}")
        print(f"STEP 4: Differential Expression Analysis - SKIPPED")
        print(f"{'='*70}")
        if not enable_differential_expression:
            print("  Reason: disabled by parameter")
        elif not disease_name:
            print("  Reason: no disease specified (cell type query)")
        elif Y is None:
            print("  Reason: no labels available")
    
    times['de_end'] = time.time()
    
    # ===========================================================================
    # STEP 5: KEGG Enrichment Plotting (via R script)
    # ===========================================================================
    kegg_success = False
    
    if enable_plotting and analysis_success and analysis_paths:
        print(f"\n{'='*70}")
        print(f"STEP 5: KEGG Pathway Enrichment Plotting")
        print(f"{'='*70}")
        
        try:
            enrichment_dir = analysis_paths.get('enrichment_results_dir', '')
            # Sanitize disease name (spaces to underscores) to match enrichment directory naming
            sanitized_disease = disease_name.replace(' ', '_')
            enrichment_all_regulated = os.path.join(enrichment_dir, f"{sanitized_disease}_all_regulated")
            plot_dir = os.path.join(session_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            kegg_script_path = get_path('enrichment.kegg_script', absolute=True)
            
            if os.path.exists(enrichment_all_regulated):
                print(f"[KEGG] Running R script on: {enrichment_all_regulated}")
                r_result = run_r_script(kegg_script_path, [enrichment_all_regulated, plot_dir])
                print(f"[KEGG] R script completed: {r_result}")
                kegg_success = True
            else:
                print(f"[KEGG] Enrichment directory not found: {enrichment_all_regulated}")
        except Exception as e:
            print(f"[KEGG] Plotting failed: {str(e)}")
    else:
        print(f"\n{'='*70}")
        print(f"STEP 5: KEGG Pathway Enrichment Plotting - SKIPPED")
        print(f"{'='*70}")
        if not enable_plotting:
            print("  Reason: disabled by parameter")
        elif not analysis_success:
            print("  Reason: differential expression analysis did not complete")
    
    times['kegg_end'] = time.time()
    
    # ===========================================================================
    # Cleanup and return results
    # ===========================================================================
    del X
    if Y is not None:
        del Y
    gc.collect()
    
    times['end'] = time.time()
    
    # Print timing summary
    print(f"\n{'='*70}")
    print(f"WORKFLOW TIMING SUMMARY")
    print(f"{'='*70}")
    print(f"Step 1 - NER/Input:        {times['ner_end'] - times['start']:.2f}s")
    print(f"Step 2 - Data Fetch:       {times['fetch_end'] - times['ner_end']:.2f}s")
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
    plot_paths = []  # Absolute paths for backward compatibility
    plots_for_report = {
        "volcano_plots": [],
        "enrichment_bar_plots": [],
        "kegg_pathway_plots": [],
        "all_plots": []
    }
    
    # Collect volcano plots
    volcano_dir = os.path.join(session_dir, "volcano_plots")
    if os.path.exists(volcano_dir):
        for f in sorted(os.listdir(volcano_dir)):
            abs_path = os.path.join(volcano_dir, f)
            rel_path = os.path.join("volcano_plots", f)
            if f.endswith('.html'):
                plot_paths.append(abs_path)
            if f.endswith('.png'):
                plots_for_report["volcano_plots"].append({
                    "name": f.replace('.png', '').replace('_', ' ').title(),
                    "filename": f,
                    "relative_path": rel_path,
                    "absolute_path": abs_path,
                    "type": "volcano",
                    "format": "png"
                })
                plots_for_report["all_plots"].append(rel_path)
    
    # Collect enrichment bar plots from enrichment_results/enrichment_plots
    enrichment_plots_dir = os.path.join(session_dir, "enrichment_results", "enrichment_plots")
    if os.path.exists(enrichment_plots_dir):
        for f in sorted(os.listdir(enrichment_plots_dir)):
            abs_path = os.path.join(enrichment_plots_dir, f)
            rel_path = os.path.join("enrichment_results", "enrichment_plots", f)
            if f.endswith('.html'):
                plot_paths.append(abs_path)
            if f.endswith('.png'):
                # Parse plot type from filename (e.g., KEGG_2021_Human_all_regulated.png)
                plot_name = f.replace('.png', '').replace('_', ' ')
                plots_for_report["enrichment_bar_plots"].append({
                    "name": plot_name,
                    "filename": f,
                    "relative_path": rel_path,
                    "absolute_path": abs_path,
                    "type": "enrichment_bar",
                    "format": "png"
                })
                plots_for_report["all_plots"].append(rel_path)
    
    # Collect KEGG pathway plots (dotplot, lollipop, combined)
    kegg_plot_dir = os.path.join(session_dir, "plots")
    if os.path.exists(kegg_plot_dir):
        for f in sorted(os.listdir(kegg_plot_dir)):
            abs_path = os.path.join(kegg_plot_dir, f)
            rel_path = os.path.join("plots", f)
            if f.endswith('.html'):
                plot_paths.append(abs_path)
            if f.endswith('.png'):
                plot_name = f.replace('.png', '').replace('_', ' ').title()
                plots_for_report["kegg_pathway_plots"].append({
                    "name": plot_name,
                    "filename": f,
                    "relative_path": rel_path,
                    "absolute_path": abs_path,
                    "type": "kegg_pathway",
                    "format": "png"
                })
                plots_for_report["all_plots"].append(rel_path)
    
    # Print summary
    print(f"[Plots] Collected plots for report:")
    print(f"  - Volcano plots: {len(plots_for_report['volcano_plots'])}")
    print(f"  - Enrichment bar plots: {len(plots_for_report['enrichment_bar_plots'])}")
    print(f"  - KEGG pathway plots: {len(plots_for_report['kegg_pathway_plots'])}")
    print(f"  - Total PNG plots: {len(plots_for_report['all_plots'])}")
    print(f"  - HTML interactive: {len(plot_paths)}")
    
    # ===========================================================================
    # Compress and cleanup large .npy files in background (don't block workflow)
    # ===========================================================================
    import threading
    
    def background_cleanup():
        print(f"\n[Cleanup] Background thread: Compressing .npy files in {session_dir}...")
        cleanup_result = compress_and_cleanup_npy_files(session_dir, remove_after_zip=True)
        print(f"[Cleanup] Background thread: Completed - {cleanup_result.get('files_processed', 0)} files processed")
    
    cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
    cleanup_thread.start()
    print(f"[Cleanup] Started background compression thread (not blocking workflow)")
    
    gc.collect()
    return {
        "success": True,
        "session_dir": session_dir,
        "num_samples": metadata.shape[0] if metadata is not None else 0,
        "num_features": len(top_gene_indices) if top_gene_indices else 0,
        "top_gene_indices": top_gene_indices,
        "top_gene_values": top_gene_values,
        "top_genes_by_fdr": top_genes_by_fdr,
        "similar_terms": similar_terms,
        "extracted_entities": fetch_dict,
        "retrieval_success": True,
        "analysis_success": analysis_success,
        "kegg_success": kegg_success,
        "analysis_paths": analysis_paths,
        "plot_paths": plot_paths,  # HTML paths for backward compatibility
        "plots_for_report": plots_for_report,  # Categorized plots with relative paths for embedding
        "message": f"Successfully retrieved {metadata.shape[0] if metadata is not None else 0} samples",
        "timing": {
            "ner": times['ner_end'] - times['start'],
            "fetch": times['fetch_end'] - times['ner_end'],
            "gene_analysis": times['gene_end'] - times['fetch_end'],
            "de_analysis": times['de_end'] - times['gene_end'],
            "kegg_plotting": times['kegg_end'] - times['de_end'],
            "total": times['end'] - times['start']
        }
    }


# ==============================================================================
# CLI INTERFACE FOR DIRECT TESTING
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Omic Fetch and Analysis Workflow - Direct Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze lung cancer with organ filter (recommended)
  python omic_fetch_analysis_workflow.py --disease "lung adenocarcinoma" --organ "lung" --session-id "lung_cancer_test"
  
  # Analyze Alzheimer's disease
  python omic_fetch_analysis_workflow.py --disease "Alzheimer disease" --organ "brain" --session-id "alzheimer_test"
  
  # Analyze specific cell type
  python omic_fetch_analysis_workflow.py --cell-type "microglial cell" --organ "brain" --session-id "microglia_test"
  
  # Combination query
  python omic_fetch_analysis_workflow.py --disease "pancreatic ductal adenocarcinoma" --cell-type "acinar cell" --session-id "pdac_acinar"
  
  # Run full test suite
  python omic_fetch_analysis_workflow.py --run-tests
        """
    )
    
    parser.add_argument("--disease", type=str, help="Disease name (e.g., 'lung adenocarcinoma', 'Alzheimer disease')")
    parser.add_argument("--cell-type", type=str, help="Cell type (e.g., 'microglial cell', 'T cell')")
    parser.add_argument("--organ", type=str, help="Organ filter for memory efficiency (e.g., 'lung', 'brain')")
    parser.add_argument("--tissue", type=str, help="Specific tissue filter")
    parser.add_argument("--text", type=str, help="Free-form text query (uses NER extraction)")
    parser.add_argument("--session-id", type=str, default="test_session", help="Session ID for output directory")
    parser.add_argument("--no-de", action="store_true", help="Skip differential expression analysis")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument("--run-tests", action="store_true", help="Run the full test suite")
    
    args = parser.parse_args()
    
    # Get sessions base directory
    sessions_base = get_path('sessions.base', absolute=True, create=True)
    
    if args.run_tests:
        # =====================================================================
        # FULL TEST SUITE
        # =====================================================================
        print("\n" + "="*80)
        print("OMIC FETCH AND ANALYSIS WORKFLOW - TEST SUITE")
        print("="*80)
        
        TEST_SESSION_DIR = os.path.join(sessions_base, 'test_suite')
        os.makedirs(TEST_SESSION_DIR, exist_ok=True)
        print(f"Test session directory: {TEST_SESSION_DIR}")
        
        # Test configurations
        TESTS = [
            {"name": "lung_adenocarcinoma", "disease": "lung adenocarcinoma", "organ": "lung"},
            {"name": "alzheimer_disease", "disease": "Alzheimer disease", "organ": "brain"},
            {"name": "breast_cancer", "disease": "breast cancer", "organ": "breast"},
            {"name": "leukemia", "disease": "leukemia"},
            {"name": "microglia_brain", "cell_type": "microglial cell", "organ": "brain"},
        ]
        
        all_results = []
        
        for test in TESTS:
            test_name = test.pop("name")
            print(f"\n{'='*60}")
            print(f"TEST: {test_name}")
            print(f"Parameters: {test}")
            print(f"{'='*60}")
            
            test_output_dir = os.path.join(TEST_SESSION_DIR, test_name)
            os.makedirs(test_output_dir, exist_ok=True)
            
            try:
                result = omic_fetch_analysis_workflow(
                    session_dir=test_output_dir,
                    enable_differential_expression=True,
                    enable_plotting=True,
                    **test
                )
                
                success = result.get('success', False)
                num_samples = result.get('num_samples', 0)
                
                print(f"Result: {'SUCCESS' if success else 'FAILED'}")
                print(f"  Samples: {num_samples}")
                print(f"  DE Analysis: {result.get('analysis_success', False)}")
                print(f"  KEGG Plot: {result.get('kegg_success', False)}")
                print(f"  Plot files: {len(result.get('plot_paths', []))}")
                
                all_results.append({
                    "test": test_name,
                    "success": success,
                    "samples": num_samples,
                    "de_success": result.get('analysis_success', False),
                    "kegg_success": result.get('kegg_success', False),
                    "plots": len(result.get('plot_paths', [])),
                    "time": result.get('timing', {}).get('total', 0)
                })
                
            except Exception as e:
                print(f"ERROR: {e}")
                all_results.append({
                    "test": test_name,
                    "success": False,
                    "samples": 0,
                    "de_success": False,
                    "kegg_success": False,
                    "plots": 0,
                    "time": 0
                })
            
            gc.collect()
        
        # Print summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        summary_df = pd.DataFrame(all_results)
        print(summary_df.to_string(index=False))
        
        summary_path = os.path.join(TEST_SESSION_DIR, "test_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")
        
    else:
        # =====================================================================
        # SINGLE RUN MODE
        # =====================================================================
        if not args.disease and not args.cell_type and not args.text:
            parser.print_help()
            print("\nError: Provide at least --disease, --cell-type, or --text")
            sys.exit(1)
        
        # Create session directory
        session_dir = os.path.join(sessions_base, args.session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        print("\n" + "="*80)
        print("OMIC FETCH AND ANALYSIS WORKFLOW")
        print("="*80)
        print(f"Session ID: {args.session_id}")
        print(f"Session Directory: {session_dir}")
        print(f"Disease: {args.disease}")
        print(f"Cell Type: {args.cell_type}")
        print(f"Organ: {args.organ}")
        print(f"Tissue: {args.tissue}")
        print(f"Text Query: {args.text}")
        print(f"DE Analysis: {not args.no_de}")
        print(f"Plotting: {not args.no_plot}")
        print("="*80 + "\n")
        
        # Build parameters
        params = {
            "session_dir": session_dir,
            "enable_differential_expression": not args.no_de,
            "enable_plotting": not args.no_plot
        }
        
        if args.disease:
            params["disease"] = args.disease
        if args.cell_type:
            params["cell_type"] = args.cell_type
        if args.organ:
            params["organ"] = args.organ
        if args.tissue:
            params["tissue"] = args.tissue
        if args.text:
            params["text"] = args.text
        
        # Run the workflow
        result = omic_fetch_analysis_workflow(**params)
        
        # Print results
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Success: {result.get('success', False)}")
        print(f"Samples Retrieved: {result.get('num_samples', 0)}")
        print(f"DE Analysis Success: {result.get('analysis_success', False)}")
        print(f"KEGG Plotting Success: {result.get('kegg_success', False)}")
        
        if result.get('top_genes_by_fdr'):
            print(f"\nTop 10 Genes by FDR:")
            for i, gene in enumerate(result.get('top_genes_by_fdr', [])[:10], 1):
                print(f"  {i}. {gene}")
        
        if result.get('plot_paths'):
            print(f"\nGenerated Plots ({len(result.get('plot_paths', []))}):")
            for p in result.get('plot_paths', []):
                print(f"  - {os.path.basename(p)}")
        
        if result.get('timing'):
            print(f"\nTiming:")
            for step, duration in result.get('timing', {}).items():
                print(f"  {step}: {duration:.2f}s")
        
        print(f"\nAll outputs saved to: {session_dir}")
        print("="*80)
