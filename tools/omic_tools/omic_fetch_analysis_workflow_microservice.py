#!/usr/bin/env python3
"""
Microservice for OmicX Analysis Workflow
Provides REST API endpoints for the omic_fetch_analysis_workflow with persistent matrix loading.
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

# Import matrix loader
from CellTOSG_Loader.memmap_matrix_loader import AdaptiveMatrixLoader
from CellTOSG_Loader.fast_data_loader import set_global_loader

# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    text: str
    top_k: Optional[int] = 20

class AnalysisResponse(BaseModel):
    top_genes: list
    disease_name: str
    #fetch_dict: dict
    #processing_times: dict
    #r_script_result: Any
    plot_paths: list
    request_metadata: dict

class HealthResponse(BaseModel):
    status: str
    matrices_loaded: bool
    startup_time: Optional[float]
    memory_usage: dict
    uptime_seconds: Optional[float]

class StatusResponse(BaseModel):
    service: str
    status: str
    matrices_loaded: int
    memory_usage: dict
    startup_time: Optional[float]
    uptime_seconds: Optional[float]

# Initialize FastAPI app
app = FastAPI(
    title="OmicX Analysis Workflow Microservice",
    description="Fast microservice for OmicX analysis with persistent matrix loading",
    version="1.0.0"
)

# Global variables
_global_matrix_loader = None
_startup_time = None
_matrices_loaded = False

def initialize_matrices():
    """Initialize the global matrix loader on startup with memory-efficient settings"""
    global _global_matrix_loader, _startup_time, _matrices_loaded
    
    print("🚀 Initializing OmicX Analysis Microservice...")
    print("📊 Setting up lazy-loading matrix loader (memory-efficient mode)...")
    
    _startup_time = time()
    
    try:
        MATRIX_DIR = get_path('external.expression_matrix_dir', absolute=True)
        print(f"📁 Matrix directory: {MATRIX_DIR}")
        
        # Get system memory
        import psutil
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        
        print(f"💾 System has {total_memory_gb:.1f}GB RAM")
        
        # Create the AdaptiveMatrixLoader with conservative settings for 128GB system
        # Use aggressive_loading=False to keep matrices as memmaps (no full load into RAM)
        _global_matrix_loader = AdaptiveMatrixLoader(
            matrix_dir=MATRIX_DIR,
            max_workers=4,  # Conservative worker count to prevent memory spikes
            memory_threshold=0.70,  # Conservative threshold (70% max usage)
            safety_factor=3.5,  # High safety factor for 128GB system
            aggressive_loading=False,  # Keep as memmap - don't load into RAM
            batch_size=2  # Load only 2 matrices at a time
        )
        
        # Check if matrices were loaded successfully
        loaded_matrices = _global_matrix_loader.list_matrices()
        loaded_count = len(loaded_matrices)
        print(f"✅ Successfully initialized {loaded_count} matrices (lazy-loading mode)")
        
        if loaded_count == 0:
            print("⚠️  Warning: No matrices were found - check matrix directory path")
        else:
            print(f"📊 Sample matrices: {loaded_matrices[:5]}...")
        
        print(f"🔥 Memory-efficient mode: Matrices will be accessed via memmap (disk-backed)")
        print(f"   This prevents loading all data into RAM at once")
        
        # Set the global loader for fast_data_loader module
        set_global_loader(_global_matrix_loader)
        
        _matrices_loaded = True
        startup_duration = time() - _startup_time
        
        print(f"✅ Matrix loader ready! Startup took {startup_duration:.2f} seconds")
        print(f"🔥 Microservice ready to serve workflow requests!")
        
    except Exception as e:
        print(f"❌ Failed to initialize matrices: {str(e)}")
        print(traceback.format_exc())
        _matrices_loaded = False

def omic_fetch_analysis_workflow_core(text: str, top_k: Optional[int] = 20) -> Dict[str, Any]:
    """
    Core workflow function extracted for microservice use.
    
    Args:
        text (str): The input text containing information about the omics data to fetch.
        top_k (int, optional): The number of top genes to return based on FDR. Defaults to 20.
    Returns:
        dict: A dictionary containing processing times and top k FDR genes.
    """
    times = {}
    times['start'] = time()
    
    # Step 1: Perform Named Entity Recognition (NER) on the input text
    fetch_dict = ner(text)
    times['ner_end'] = time()
    
    # Step 2: Fetch the omics data based on the recognized entities
    data_dict, file_path_dict, disease_name = omic_fetch(fetch_dict)
    times['fetch_end'] = time()
    
    # Step 3: Perform analysis on the fetched omics data
    data_and_analysis_dict = omic_analysis(disease_name, data_dict)
    times['analysis_end'] = time()
    
    # Step 4: Read topk fdr genes and finally passed to the agent for further analysis 
    gene_dataset = pd.read_csv(f"{data_and_analysis_dict['differential_expression_dir']}/significant_genes_by_fdr.csv")
    top_genes = gene_dataset['Name'].astype(str).values.tolist()[:top_k]
    times['gene_processing_end'] = time()

    # Step 5: Run R script for enrichment analysis - use paths from config
    enrichment_dir = get_path('data.enrichment_results', absolute=True)
    enrichment_all_regulated = os.path.join(enrichment_dir, f"{disease_name}_all_regulated")
    output_dir = get_path('data.plot', absolute=True, create=True)
    kegg_script_path = get_path('enrichment.kegg_script', absolute=True)  # change to kegg_copy_script for HTML output

    r_result = run_r_script(kegg_script_path, [enrichment_all_regulated, output_dir])
    times['r_script_end'] = time()

    # Calculate durations
    durations = {
        'ner_duration': times['ner_end'] - times['start'],
        'fetch_duration': times['fetch_end'] - times['ner_end'],
        'analysis_duration': times['analysis_end'] - times['fetch_end'],
        'gene_processing_duration': times['gene_processing_end'] - times['analysis_end'],
        'r_script_duration': times['r_script_end'] - times['gene_processing_end'],
        'total_duration': times['r_script_end'] - times['start']
    }

    return {
        'top_genes': top_genes,
        'disease_name': disease_name,
        #'fetch_dict': fetch_dict,
        #'processing_times': durations,
        #'r_script_result': r_result,
        'plot_paths': ["/assets/my_interactive_plot.html",
                       "/assets/my_interactive_plot_1.html"] # temporary solution
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    memory = psutil.virtual_memory()
    
    return HealthResponse(
        status='healthy' if _matrices_loaded else 'initializing',
        matrices_loaded=_matrices_loaded,
        startup_time=_startup_time,
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
    if not _matrices_loaded:
        raise HTTPException(status_code=503, detail="Matrices not loaded yet")
    
    memory = psutil.virtual_memory()
    
    # Get matrix loader info
    matrix_count = len(_global_matrix_loader.list_matrices()) if _global_matrix_loader else 0
    
    # Print debug info
    if _global_matrix_loader:
        loaded_matrices = _global_matrix_loader.list_matrices()
        print(f"[Status Debug] Loaded matrices: {loaded_matrices[:5]}..." if len(loaded_matrices) > 5 else f"[Status Debug] Loaded matrices: {loaded_matrices}")
    
    return StatusResponse(
        service='OmicX Analysis Workflow Microservice',
        status='ready',
        matrices_loaded=matrix_count,
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
    """Main analysis endpoint"""
    if not _matrices_loaded:
        raise HTTPException(status_code=503, detail="Service not ready - matrices still loading")
    
    try:
        # Validate top_k
        if request.top_k <= 0:
            raise HTTPException(status_code=400, detail="top_k must be a positive integer")
        
        # Run the workflow
        start_time = time()
        result = omic_fetch_analysis_workflow_core(request.text, request.top_k)
        request_duration = time() - start_time
        
        # Add request metadata
        result['request_metadata'] = {
            'request_duration': round(request_duration, 2),
            'input_text': request.text,
            'top_k': request.top_k,
            'timestamp': time()
        }
        
        return AnalysisResponse(**result)
        
    except ValueError as e:
        # Handle data/query related errors with 400 Bad Request
        error_message = str(e)
        print(f"❌ Data/Query Error in /analyze endpoint: {error_message}")
        
        # Return a clean error response that FastAPI can handle
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=400,
            content={
                "error": "Query validation failed",
                "message": error_message,
                "type": "ValueError"
            }
        )
        
    except Exception as e:
        # Handle other internal server errors with 500
        error_details = {
            'error': 'Internal server error',
            'message': str(e),
            'type': type(e).__name__,
            'traceback': traceback.format_exc()
        }
        print(f"❌ Internal Error in /analyze endpoint: {error_details}")
        
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": str(e),
                "type": type(e).__name__
            }
        )

@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        'service': 'OmicX Analysis Workflow Microservice',
        'version': '1.0.0',
        'status': 'ready' if _matrices_loaded else 'initializing',
        'endpoints': {
            'GET /': 'This documentation',
            'GET /health': 'Health check',
            'GET /status': 'Detailed status',
            'POST /analyze': 'Run workflow analysis',
            'GET /docs': 'Interactive API documentation',
            'GET /redoc': 'Alternative API documentation'
        },
        'analyze_endpoint': {
            'method': 'POST',
            'content_type': 'application/json',
            'required_fields': ['text'],
            'optional_fields': ['top_k (default: 20)'],
            'example_request': {
                'text': 'What are the key dysfunctional signaling targets in microglia in Alzheimer\'s disease?',
                'top_k': 20
            }
        }
    }

if __name__ == '__main__':
    # Initialize matrices on startup
    initialize_matrices()
    
    # Start FastAPI server with uvicorn
    print(f"🌐 Starting FastAPI microservice on http://0.0.0.0:8010")
    print(f"📚 Interactive docs available at http://0.0.0.0:8010/docs")
    print(f"📖 Alternative docs available at http://0.0.0.0:8010/redoc")
    
    uvicorn.run(app, host="0.0.0.0", port=8010, log_level="info")
