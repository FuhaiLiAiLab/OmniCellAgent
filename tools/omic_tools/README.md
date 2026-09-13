# OmniCell Omic Analysis Tools

This directory contains tools for analyzing single-cell omics data using the OmniCellTOSG dataset.

## Exact query value lists

Each session receives `available_diseases.txt` and `available_cell_types.txt`.
These are sorted, unique, nonempty values from the full dataset's
`disease_BMG_name` and `CMT_name` columns, respectively. Source spelling is
preserved. Each file starts with the requested value's match in that list:

```text
Matched disease in list: "Alzheimer's Disease"
```

The remaining lines contain one available value each. The header explicitly
reports `NONE` when the requested value is absent or no filter was requested.
Matching is exact and case-insensitive, as in the query path. A vocabulary match
does not guarantee that the combined query has eligible samples. Both lists
are written before sampling, including when retrieval finds no matching cohort.

## Disease/control sampling

The direct `omic_fetch_analysis_workflow.py` pipeline samples disease and normal
metacells independently from rows with usable donor IDs. It first applies the
disease, cell type, organ, tissue and optional sex/protocol query constraints,
excludes unknown donors, and removes repeated matrix pointers. Each group
contributes at most `--sample-size` metacells (default 1000), without replacement
and with random seed 42. Sampling takes one metacell per donor first, then fills
remaining slots in rounds across donors with unused rows. This maximizes donor
coverage to `min(sample limit, available donors)` and balances contributions
where donor pools permit. If fewer metacells than the limit are available, all
are retained. A group with zero eligible metacells stops retrieval; unknown
donors are not used as fallback.

For AD astrocytes in brain, the current dataset has 671 eligible disease rows
and 4,044 eligible controls. The default therefore returns 671 disease rows and
1,000 controls, covering all 54 AD and 244 control donor keys. Donor identity
uses `(source, dataset_id, donor_id)` so reused labels from different studies
are not merged. These keys do not establish whether the same biological donor
appears in multiple studies. Diagnostics use the same definition and also
report counts of the raw donor labels.

The previous age/sex balancing is bypassed, so groups may differ in size and
covariate composition. The existing cohort diagnostics still assess the cohort.
Maximizing donor coverage does not make the downstream metacell-level DE test
donor-aware; a donor-level pseudobulk or suitable mixed model is a separate step.

Metacell and available/selected donor counts are printed for both groups and
saved to `donor_sampling.json` in the
session directory. The loader receives only the selected metadata, ensuring
that saved labels and expression matrices refer to the same cohort. The shared
dataset remains unchanged. Other label tasks retain their existing sampling.

```bash
python tools/omic_tools/omic_fetch_analysis_workflow.py \
  --disease "Alzheimer's Disease" --organ brain --cell-type astrocyte \
  --sample-size 1000 --session-id alzheimer_known_donors
```

## Quick Start

### 1. Download the OmniCellTOSG Dataset

The dataset uses `.npy` format files that work with memory-efficient loading:

```bash
# Download to a specific directory (recommended)
cd /path/to/your/data/location
./tools/omic_tools/download_omnicell_dataset.sh /home/hao/BioProtocol/OmniCellTOSG/OmniCellTOSG_Dataset

# Or download to current directory
./tools/omic_tools/download_omnicell_dataset.sh ./OmniCellTOSG_Dataset
```

Requirements:
- `huggingface-cli` installed: `pip install huggingface_hub[cli]`
- Sufficient disk space (~500GB for full dataset)

### 2. Update Configuration

Edit `configs/paths.yaml` to point to your downloaded dataset:

```yaml
external:
  # Point to the directory containing cell_metadata_with_mappings.parquet
  omnicell_data_root: "/home/hao/BioProtocol/OmniCellTOSG/OmniCellTOSG_Dataset"
  
  # Point to the expression_matrix subdirectory
  expression_matrix_dir: "/home/hao/BioProtocol/OmniCellTOSG/OmniCellTOSG_Dataset/expression_matrix"
  
  # Point to BioMedGraphica data
  biomedgraphica_dir: "/home/hao/BioProtocol/OmniCellTOSG/OmniCellTOSG_Dataset/BioMedGraphica-Conn"
```

### 3. Start the Microservice

The microservice uses **memory-efficient on-demand loading** - no pre-loading of matrices:

```bash
# Activate your environment
conda activate autogen-dev

# Start the microservice (port 8010)
cd /home/hao/BioProtocol/OmniCellAgent
python tools/omic_tools/omic_fetch_analysis_workflow_microservice_simple.py
```

Memory usage:
- **Startup**: < 1 second, ~200MB RAM
- **During query**: Only loads matrices for matching cells
- **No memory explosion**: Uses numpy mmap (disk-backed arrays)

### 4. Test the Service

```python
from tools.omic_tools.omic_fetch_analysis_workflow_client import omic_fetch_analysis_workflow

# Simple query
result = omic_fetch_analysis_workflow("pancreatic cancer genes", top_k=20)

if isinstance(result, dict):
    print(f"Disease: {result['disease_name']}")
    print(f"Top genes: {result['top_genes']}")
else:
    print(f"Error: {result}")
```

Or test with curl:
```bash
curl -X POST http://localhost:8010/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "diabetes genes", "top_k": 10}'
```

## Architecture

### Memory-Efficient Design

The system uses **on-demand loading** instead of pre-loading all matrices:

1. **NER**: Extract disease/cell type from text query
2. **Metadata Query**: Find matching cells in metadata (~100MB)
3. **Matrix Loading**: Load ONLY expression data for matching cells using mmap
4. **Analysis**: Differential expression & enrichment

**Key insight**: You never need ALL 99 matrices (>1TB) loaded - only the tiny subset matching each specific query!

### Files

- `download_omnicell_dataset.sh` - Download script for HuggingFace dataset
- `omic_fetch_analysis_workflow_microservice_simple.py` - Memory-efficient microservice
- `omic_fetch_analysis_workflow_client.py` - Python client
- `ner_tool.py` - Named entity recognition
- `omic_fetch_tool.py` - Data fetching from metadata
- `omic_analysis_tool.py` - Differential expression analysis
- `subprocess_r.py` - R script execution for enrichment

### Data Loader

The system uses the **simple data loader** (`CellTOSG_Loader/data_loader.py`):
- Loads matrices with `np.load(mmap_mode="r")` - disk-backed, no RAM usage
- Only reads the specific rows needed for the query
- No pre-loading, no memory explosion

## Troubleshooting

### Memory Issues

If you experience memory issues:

1. **Check you're using the simple microservice**:
   ```bash
   ps aux | grep omic_fetch_analysis_workflow_microservice_simple.py
   ```

2. **Verify dataset format**: Should have `.npy` files, not `.dat`+`.json`
   ```bash
   ls expression_matrix/ | head -5
   # Should show: *.npy files
   ```

3. **Monitor memory during query**:
   ```bash
   watch -n 1 'free -h'
   ```

### Dataset Format

The HuggingFace dataset uses `.npy` format which is optimal:
- ✅ `.npy` - NumPy native format, works with `np.load(mmap_mode="r")`
- ❌ `.dat`+`.json` - Custom format, requires special loader

If you have `.dat`+`.json` files, download the correct version from HuggingFace.

### Port Already in Use

```bash
# Kill any process on port 8010
lsof -ti:8010 | xargs kill -9
```

## Performance

Typical query performance:
- **NER**: 0.1-0.5s
- **Data fetch**: 2-10s (depends on query specificity)
- **Differential expression**: 5-30s (depends on sample size)
- **Enrichment analysis**: 5-15s (R KEGG analysis)
- **Total**: 15-60s per query

Memory usage stays under 10GB for most queries, even on 128GB systems.
