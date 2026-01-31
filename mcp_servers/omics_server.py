"""Omics Analysis MCP Server - Port 9005"""
import os
import sys
import asyncio
from typing import Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from fastmcp import FastMCP
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(project_root, 'tools', 'omic_tools'))
from omic_fetch_analysis_workflow import omic_fetch_analysis_workflow

load_dotenv(os.path.join(project_root, '.env'))
# Initialize MCP server
mcp = FastMCP("Omics Analysis Tools 📊")


@mcp.tool()
async def analyze_omics_data(
    query: str,
    session_id: Optional[str] = None
) -> str:
    """
    Perform comprehensive single-cell omics data analysis workflow.
    
    This tool provides end-to-end single-cell RNA-seq analysis:
    
    **Pipeline Steps:**
    1. Named Entity Recognition (NER): Extracts disease, cell type, tissue from query
    2. Data Retrieval: Fetches relevant data from OmniCellTOSG database
    3. Differential Expression: Identifies significantly changed genes (DESeq2)
    4. Pathway Enrichment: Performs KEGG pathway analysis (R)
    5. Visualization: Generates volcano plots and enrichment plots
    6. Results Summary: Returns top genes, pathways, and statistics
    
    **Automatic Features:**
    - Fuzzy matching for disease/tissue names
    - BiomedBERT semantic similarity for entity matching
    - Statistical analysis with multiple testing correction
    - Publication-quality visualizations
    - Comprehensive session reports
    
    **Files Generated:**
    - webapp/sessions/{session_id}/dataset_outputs/differential_expression/
      - top_genes.csv
      - all_results.csv
      - volcano_plot.png
    - webapp/sessions/{session_id}/dataset_outputs/enrichment/
      - kegg_enrichment.csv
      - kegg_dotplot.png
    
    Args:
        query: Natural language query describing the analysis
               Examples:
               - "Analyze lung adenocarcinoma single-cell data"
               - "Find key genes in pancreatic cancer"
               - "Compare T cells in breast cancer vs normal"
        session_id: Session identifier for file organization (default: auto-generated)
    
    Returns:
        Comprehensive analysis report containing:
        - Sample statistics (n_samples, n_cells, etc.)
        - Top differentially expressed genes (symbol, log2FC, padj)
        - Enriched KEGG pathways (pathway name, p-value, gene ratio)
        - File paths to generated plots
        - Analysis metadata (runtime, parameters)
    
    Example:
        result = await analyze_omics_data(
            query="Analyze pancreatic ductal adenocarcinoma",
            session_id="pdac_analysis_001"
        )
    
    Note:
        - Requires OmniCellTOSG database installation
        - Requires R with DESeq2 and clusterProfiler packages
        - Analysis takes 5-15 minutes for typical datasets
        - Large expression matrices are compressed with lrzip
        - Requires GLiNER and BioBERT services (ports 8002, 8003)
    """
    if session_id is None:
        import uuid
        session_id = f"omics_{uuid.uuid4().hex[:8]}"
    
    # Run the workflow in thread pool (CPU-intensive)
    result = await asyncio.to_thread(
        omic_fetch_analysis_workflow,
        query=query,
        session_id=session_id
    )
    
    return result


if __name__ == "__main__":
    import sys
    
    if "--sse" in sys.argv:
        mcp.run(transport="sse", port=9005)
    else:
        mcp.run()
