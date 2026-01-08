#!/usr/bin/env python3
"""
Simplified Microservice for OmicX Analysis Workflow - NO MATRIX PRE-LOADING
Uses on-demand matrix loading only when needed for specific queries.
"""

import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import pandas as pd
from time import time
import traceback
import psutil
import uvicorn

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('CellTOSG_Loader')

# Add project root to path for path config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.path_config import get_path

# Import workflow components
from ner_tool import ner
from omic_fetch_tool import omic_fetch
from omic_analysis_tool import omic_analysis
from subprocess_r import run_r_script

# Import the simple memmap loader (supports both .npy and .dat+.json formats)
from CellTOSG_Loader import simple_memmap_loader

# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    text: str
    top_k: Optional[int] = 20

class AnalysisResponse(BaseModel):
    top_genes: list
    disease_name: str
    plot_paths: list
    request_metadata: dict

class HealthResponse(BaseModel):
    status: str
    memory_usage: dict
    uptime_seconds: Optional[float] = None

class StatusResponse(BaseModel):
    service: str
    status: str
    memory_usage: dict
    startup_time: Optional[float] = None
    uptime_seconds: Optional[float] = None

# FastAPI app
app = FastAPI(
    title="OmicX Analysis Workflow API (Simplified)",
    description="Memory-efficient microservice - matrices loaded on-demand per query",
    version="2.0.0"
)

# Global variables
_startup_time = None
_service_ready = False

def initialize_service():
    """Initialize the microservice (NO matrix loader - pure on-demand loading)"""
    global _startup_time, _service_ready
    
    print("🚀 Initializing OmicX Analysis Microservice (Pure On-Demand Mode)")
    print("📊 Using original simple mmap loader - ZERO pre-loading")
    
    _startup_time = time()
    
    try:
        # Verify that required paths exist
        MATRIX_DIR = get_path('external.expression_matrix_dir', absolute=True)
        DATA_ROOT = get_path('external.omnicell_data_root', absolute=True)
        OUTPUT_DIR = get_path('data.dataset_outputs', absolute=True, create=True)
        
        print(f"📁 Matrix directory: {MATRIX_DIR}")
        print(f"📁 Data root: {DATA_ROOT}")
        print(f"📁 Output directory: {OUTPUT_DIR}")
        
        # Check if paths exist
        if not os.path.exists(MATRIX_DIR):
            raise FileNotFoundError(f"Matrix directory not found: {MATRIX_DIR}")
        if not os.path.exists(DATA_ROOT):
            raise FileNotFoundError(f"Data root not found: {DATA_ROOT}")
        
        # Count available matrices (.npy files)
        matrix_files = [f for f in os.listdir(MATRIX_DIR) if f.endswith('.npy')]
        print(f"📊 Found {len(matrix_files)} .npy matrix files available")
        
        # Get system memory info
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        print(f"💾 System: {total_memory_gb:.1f}GB RAM, {available_gb:.1f}GB available")
        
        # Patch the fast_data_loader module to use the original simple method
        import CellTOSG_Loader.fast_data_loader as fast_loader
        
        # Replace with simple memmap version (supports .npy and .dat+.json)
        def simple_load_expression(df_meta):
            """Simple on-demand loading using memmap - NO pre-loading"""
            print(f"[Simple Memmap Loader] Loading expression data for {len(df_meta)} samples")
            return simple_memmap_loader.load_expression_by_metadata_simple_memmap(df_meta, MATRIX_DIR)
        
        # Monkey-patch BOTH functions to avoid any confusion
        fast_loader.load_expression_by_metadata = simple_load_expression
        fast_loader.load_expression_by_metadata_fast = simple_load_expression
        
        # Also create a dummy global loader to prevent errors
        class DummyLoader:
            def __init__(self):
                pass
        fast_loader._global_loader = DummyLoader()
        
        print(f"✅ Patched fast_data_loader to use simple mmap (reads from disk, NO RAM)")
        
        _service_ready = True
        startup_duration = time() - _startup_time
        
        print(f"✅ Service ready! Startup took {startup_duration:.2f} seconds")
        print(f"🔥 Zero memory footprint - matrices loaded ONLY when needed per query")
        
    except Exception as e:
        print(f"❌ Failed to initialize: {str(e)}")
        print(traceback.format_exc())
        _service_ready = False

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    initialize_service()

def omic_fetch_analysis_workflow_core(text: str, top_k: int = 20) -> dict:
    """
    Core workflow function - matrices loaded on-demand during omic_fetch.
    
    Args:
        text (str): The input text containing information about the omics data to fetch.
        top_k (int, optional): The number of top genes to return based on FDR. Defaults to 20.
    Returns:
        dict: A dictionary containing processing times and top k FDR genes.
    """
    times = {}
    times['start'] = time()
    
    # Step 1: Perform Named Entity Recognition (NER) on the input text
    print(f"\n🔬 Step 1: NER - Extracting entities from text...")
    fetch_dict = ner(text)
    times['ner_end'] = time()
    print(f"✅ NER completed in {times['ner_end'] - times['start']:.2f}s")
    print(f"   Entities: {fetch_dict}")
    
    # Step 2: Fetch the omics data based on the recognized entities
    # THIS IS WHERE MATRICES ARE LOADED (on-demand for matching cells only)
    print(f"\n🔬 Step 2: Fetching omic data (matrices loaded on-demand)...")
    data_dict, file_path_dict, disease_name = omic_fetch(fetch_dict)
    times['fetch_end'] = time()
    print(f"✅ Data fetch completed in {times['fetch_end'] - times['ner_end']:.2f}s")
    
    # Step 3: Perform analysis on the fetched omics data
    print(f"\n🔬 Step 3: Differential expression analysis...")
    data_and_analysis_dict = omic_analysis(disease_name, data_dict)
    times['analysis_end'] = time()
    print(f"✅ Analysis completed in {times['analysis_end'] - times['fetch_end']:.2f}s")
    
    # Step 4: Read topk fdr genes and finally passed to the agent for further analysis 
    print(f"\n🔬 Step 4: Extracting top {top_k} genes...")
    gene_dataset = pd.read_csv(f"{data_and_analysis_dict['differential_expression_dir']}/significant_genes_by_fdr.csv")
    top_genes = gene_dataset['Name'].astype(str).values.tolist()[:top_k]
    times['gene_processing_end'] = time()
    print(f"✅ Gene processing completed in {times['gene_processing_end'] - times['analysis_end']:.2f}s")
    print(f"   Top genes: {top_genes[:5]}...")

    # Step 5: Run R script for enrichment analysis - use paths from config
    print(f"\n🔬 Step 5: Running R script for enrichment analysis...")
    enrichment_dir = get_path('data.enrichment_results', absolute=True)
    enrichment_all_regulated = os.path.join(enrichment_dir, f"{disease_name}_all_regulated")
    output_dir = get_path('data.plot', absolute=True, create=True)
    kegg_script_path = get_path('enrichment.kegg_script', absolute=True)

    r_result = run_r_script(kegg_script_path, [enrichment_all_regulated, output_dir])
    times['r_script_end'] = time()
    print(f"✅ R script completed in {times['r_script_end'] - times['gene_processing_end']:.2f}s")

    # Calculate durations
    durations = {
        'ner_duration': times['ner_end'] - times['start'],
        'fetch_duration': times['fetch_end'] - times['ner_end'],
        'analysis_duration': times['analysis_end'] - times['fetch_end'],
        'gene_processing_duration': times['gene_processing_end'] - times['analysis_end'],
        'r_script_duration': times['r_script_end'] - times['gene_processing_end'],
        'total_duration': times['r_script_end'] - times['start']
    }
    
    print(f"\n✅ WORKFLOW COMPLETE in {durations['total_duration']:.2f}s")

    return {
        'top_genes': top_genes,
        'disease_name': disease_name,
        'plot_paths': ["/assets/my_interactive_plot.html",
                       "/assets/my_interactive_plot_1.html"],
        'request_metadata': {
            'text': text,
            'top_k': top_k,
            'processing_times': durations
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    memory = psutil.virtual_memory()
    
    return HealthResponse(
        status='healthy' if _service_ready else 'initializing',
        memory_usage={
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'percent_used': memory.percent
        },
        uptime_seconds=time() - _startup_time if _startup_time else None
    )

@app.get("/status", response_model=StatusResponse)
async def status():
    """Detailed status endpoint"""
    if not _service_ready:
        raise HTTPException(status_code=503, detail="Service not ready yet")
    
    memory = psutil.virtual_memory()
    
    return StatusResponse(
        service='OmicX Analysis Workflow Microservice (Simplified)',
        status='ready',
        memory_usage={
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2),
            'percent_used': memory.percent
        },
        startup_time=_startup_time,
        uptime_seconds=round(time() - _startup_time, 2) if _startup_time else None
    )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """
    Main analysis endpoint - processes text query and returns top genes.
    Matrices are loaded on-demand during processing.
    """
    if not _service_ready:
        raise HTTPException(status_code=503, detail="Service not ready yet")
    
    try:
        print(f"\n{'='*80}")
        print(f"📥 NEW REQUEST: {request.text}")
        print(f"{'='*80}")
        
        # Track memory before
        memory_before = psutil.virtual_memory()
        print(f"💾 Memory before: {memory_before.percent}% used ({memory_before.used / (1024**3):.1f}GB)")
        
        result = omic_fetch_analysis_workflow_core(request.text, request.top_k)
        
        # Track memory after
        memory_after = psutil.virtual_memory()
        print(f"💾 Memory after: {memory_after.percent}% used ({memory_after.used / (1024**3):.1f}GB)")
        print(f"💾 Memory delta: +{(memory_after.used - memory_before.used) / (1024**3):.1f}GB")
        
        return AnalysisResponse(**result)
        
    except Exception as e:
        error_msg = f"Analysis failed: {str(e)}"
        print(f"\n❌ ERROR: {error_msg}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    print("🚀 Starting OmicX Analysis Workflow Microservice (Simplified)")
    print("=" * 80)
    uvicorn.run(app, host="0.0.0.0", port=8010, log_level="info")
