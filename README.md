# OmniCellAgent: Towards AI Co-Scientists for Scientific Discovery in Precision Medicine

![](webapp/assets/dash-logo-stripe.svg)

## 🚀 Quick Start

### 1. Start All Services

```bash
# Start Neo4j, RAG tools, and microservices
bash scripts/startup.sh

# Test all services are running
bash scripts/test_services.sh

# Stop all services when done
bash scripts/stop_services.sh
```

### 2. Run AI Agent Analysis

**Simple Query (Literature Research):**
```bash
conda run -n autogen-dev python agent/simple_magentic_agent.py \
  --query "What are the key therapeutic targets for Alzheimer's Disease?"
```

**Full Analysis Pipeline (with Omic Data):**
```bash
conda run -n autogen-dev python agent/simple_magentic_agent.py \
  --query "Analyze lung cancer: find relevant genes, perform differential expression analysis, and identify therapeutic targets. Use Omni cell mining agent to do enrichment" \
  --session-id "lung_cancer_analysis"
```

Analyze lung cancer: identify relevant genes, perform differential expression analysis, and discover therapeutic targets using the OmniCell mining agent for pathway enrichment


Results will be saved in `webapp/sessions/lung_cancer_analysis/` including:
- Differential expression analysis
- Volcano plots
- Enrichment analysis plots
- Gene lists and pathway information

### 3. Launch Web UI

```bash
conda run -n autogen-dev python webapp/index.py
# Access at http://localhost:8050
```

The Web UI provides:
- **Session Management**: Each conversation creates a unique session ID (e.g., `session_20251218_143022_a1b2c3d4`)
- **Real-time Progress**: See step-by-step agent reasoning and tool calls
- **Visualization**: Plots and figures from analysis are displayed inline
- **Output Storage**: All session outputs saved in `webapp/sessions/<session_id>/`

### 4. Installation

```bash
# Create conda environment
conda create -n autogen-dev python=3.10
conda activate autogen-dev

# Install dependencies
pip install -r requirements.txt --no-deps

# Configure paths in configs/paths.yaml
# Set Neo4j path, OmniCellTOSG path, etc.
```

---

## 📂 Session Management

Each analysis session is stored in its own directory under `webapp/sessions/`:

```
webapp/sessions/
├── lung_cancer_analysis/           # Named session from CLI
│   ├── differential_expression/    # DE analysis results
│   ├── volcano_plots/              # Volcano plot images
│   ├── enrichment_results/         # Enrichment CSV files
│   ├── enrichment_plots/           # Enrichment visualizations
│   ├── plots/                      # KEGG pathway plots
│   └── top_genes_by_expression.csv
├── session_20251218_143022_a1b2c3d4/  # Auto-generated UI session
│   └── ...
```

**Session ID Formats:**
- **CLI**: Use `--session-id "your_name"` for custom names
- **Web UI**: Auto-generated as `session_YYYYMMDD_HHMMSS_<random>`

---

## 🔧 Microservices

The system runs several microservices that provide different capabilities:

| Service | Port | Description | Test Command |
|---------|------|-------------|--------------|
| **Neo4j** | 7474, 7687 | Graph database for biomedical knowledge | `curl http://localhost:7474` |
| **Scientist RAG** | 8000 | Author-specific literature knowledge base | `curl http://localhost:8000/health` |
| **GRetriever** | 8001 | Knowledge graph query service | `curl http://localhost:8001/health` |
| **GLiNER** | - | Named entity recognition | Process check |
| **BioBERT** | - | Biomedical text embeddings | Process check |
| **Webapp** | 8050 | Web interface for the agent | `curl http://localhost:8050` |

**Service Management:**
```bash
# Check service status
ps aux | grep python | grep -E "(scientist_tool|gretriever|webapp)"

# View logs
tail -f logs/service-logs/scientist_tool.log
tail -f logs/service-logs/gretriever_service_output.log

# Check GPU usage (for GRetriever)
nvidia-smi
```

See [scripts/README.md](scripts/README.md) for detailed service management documentation.

---

## 📦 Data Requirements

If you need the specialized tools *OmniCellTOSG*, download from  https://huggingface.co/datasets/FuhaiLiAiLab/OmniCellTOSG_Dataset OR use the download script inside https://github.com/FuhaiLiAiLab/OmniCellTOSG?tab=readme-ov-file and paste the expression folder path into the config file in the config folder.

# Paper

See 

**OmniCellAgent: Towards AI Co-Scientists for Scientific Discovery in Precision Medicine**
(https://www.biorxiv.org/content/10.1101/2025.07.31.667797v1)


If you used the enrichment study part, please also cite OmniCellTOSG https://arxiv.org/abs/2504.02148 

# Troubleshoot

## Todo
- add autogen_ext.memory.canvas to store memory
- line 487-488, **tool call** messages are ignored. source: autogen_agentchat\teams\_group_chat\_magentic_one\_magentic_one_orchestrator.py 
- line 493, **tool call summary*** is added to the message thread instead


## Neo4j Installation
### The database & dataset
Install the Neo4j database (and relevant JDK) by following [official instructions](https://neo4j.com/docs/operations-manual/current/installation/linux/debian/#debian-installation).
You'll also need the Neo4j [GenAI plugin](https://neo4j.com/docs/cypher-manual/current/genai-integrations/#_installation) and the Neo4j [Graph Data Science library](https://neo4j.com/docs/graph-data-science/current/installation/).

With the database installed and running, you can load the STaRK-Prime dataset by running the python notebook in `data-loading/stark_prime_neo4j_loading.ipynb`.
Alternatively, obtain a database dump at AWS S3 (bucket at gds-public-dataset/stark-prime-neo4j523) for database version 5.23.


## Prerequisite


Data downloading


Use sftp to connect to server





### Biomedical Tool Setup

To Install keggtools require **graphviz**, you need to use conda to install first:

```
conda install anaconda::graphviz
```

### Python Environment Setup
Once the **graphviz** is installed, you can set up the rest of the Python environment by running the following commands in the terminal:
```
pip install -r requirement.txt
```

### API Key Setup


Place a .env file in the Folder
```

from dotenv import load_dotenv

load_dotenv()  

```

## Run the agent

### First step: Start the Scientist RAG

```
python tools/scientist_tool.py
```

or use nohup to run it in the background
```
nohup python tools/scientist_tool.py > logs/scientist_tool_output.log 2>&1 &
```

### Second step: Start the G-Retriever Tool

#### First, Start Neo4j Database
Go to the directory where Neo4j is installed, and run the following command to start the database (supposed to be installed in `/neo4j-community-2025.03.0`):

```
nohup /bin/neo4j console > /logs/neo4j_log.out 2>&1 &
```
#### Second, Start the G-Retriever Service

```
python tools/gretriever_service.py
```


### Third step: Start the Magentic Agent

```
python magentic_agent.py --task "What are the key dysfunctional signaling targets in microglia of AD?" --task_id "1" --mode magentic > logs/results.txt
```

Or put in UI:
```
What are the key dysfunctional signaling targets in microglia of AD, based on the internal database?
```



### Note for Hyper-RAG
Three query mode: Hyper, Hyper-lite, Navie

In "hyper" mode:
First LLM call: Extract keywords from the query using keywords_extraction prompt
Second LLM call: Generate the final response using the retrieved context

In "hyper-lite" mode:
Similar to hyper mode, makes 2 LLM calls

In "naive" mode:
Makes only 1 LLM call to generate the final response
The llm_model_max_async parameter (default: 16) controls how many concurrent LLM calls can be processed at once. This means:
For naive query mode: The system can handle up to 16 concurrent user queries (since each makes 1 LLM call)
For hyper/hyper-lite modes: The system can handle up to 8 concurrent user queries (since each makes 2 LLM calls)

When set QueryParam(only_need_context=True) in HyperRAG:
The first call still made. (extracts both low-level keywords (entities) and high-level keywords)
The second call is skipped. 

-----Entities-----
```csv
id, entity, type, description, additional properties, rank
```

### Note for OmicellTOSG dataseet
df_all in CellTOSGSubsetBuilder stores the field (e.g., tissue, tissue_general, disease, cell type). 

---

## 🏗️ Architecture: Omic Analysis Pipeline

The omic analysis tool follows a multi-stage pipeline architecture:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         simple_magentic_agent.py                            │
│                                                                             │
│  create_omic_analysis_tool(session_id)                                      │
│         │                                                                   │
│         └── omic_analysis_tool() [FunctionTool wrapper for LLM]             │
│                    │                                                        │
│                    └── calls: _omic_workflow(session_dir, disease, ...)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     omic_fetch_analysis_workflow.py                         │
│                                                                             │
│  omic_fetch_analysis_workflow()                                             │
│         │                                                                   │
│         ├── STEP 1: NER extraction (ner_tool.py)                            │
│         │                                                                   │
│         ├── STEP 2: Data retrieval (CellTOSGDataLoader)                     │
│         │                                                                   │
│         ├── STEP 3: Compute top genes by expression                         │
│         │                                                                   │
│         ├── STEP 4: Differential Expression Analysis ──────────────────┐    │
│         │                                                              │    │
│         │         ┌─────────────────────────────────────────────────────────┤
│         │         │              omic_analysis.py                      │    │
│         │         │                                                    │    │
│         │         │  omic_analysis(disease_name, data_dict, ...)       │    │
│         │         │         │                                          │    │
│         │         │         ├── DE analysis (Mann-Whitney U test)      │    │
│         │         │         │                                          │    │
│         │         │         ├── Volcano plots (matplotlib + plotly)    │    │
│         │         │         │   └── saves to: volcano_plots/           │    │
│         │         │         │       ├── *.png (static)                 │    │
│         │         │         │       └── *.html (interactive)           │    │
│         │         │         │                                          │    │
│         │         │         ├── Enrichment analysis (Enrichr API)      │    │
│         │         │         │   └── saves to: enrichment_results/      │    │
│         │         │         │                                          │    │
│         │         │         └── Enrichment plots (matplotlib)          │    │
│         │         │             └── saves to: enrichment_plots/        │    │
│         │         └─────────────────────────────────────────────────────────┤
│         │                                                              │    │
│         └── STEP 5: KEGG Pathway Plotting (R script) ──────────────────┘    │
│                    │                                                        │
│                    └── subprocess: Rscript enrichment/kegg.R                │
│                        └── saves to: plots/                                 │
│                            ├── kegg_dotplot.png                             │
│                            ├── kegg_dotplot.html (interactive)              │
│                            ├── pathway_combined_plot.png                    │
│                            └── pathway_combined_plot.html (interactive)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Output Directory Structure

```
webapp/sessions/<session_id>/
├── differential_expression/           # From omic_analysis.py
│   ├── unpaired_differential_expression_results.csv
│   ├── significant_genes_by_fdr.csv
│   ├── significant_genes_by_fc.csv
│   ├── significant_upregulated_genes.csv
│   └── significant_downregulated_genes.csv
├── volcano_plots/                     # From omic_analysis.py
│   ├── volcano_plot.png
│   ├── volcano_plot.html              # Interactive (plotly)
│   ├── volcano_plot_permissive.png
│   └── volcano_plot_permissive.html   # Interactive (plotly)
├── enrichment_results/                # From omic_analysis.py (Enrichr API)
│   ├── <disease>_all_regulated/
│   ├── <disease>_up_regulated/
│   └── <disease>_down_regulated/
│       └── enrichment_plots/          # Python matplotlib plots
├── plots/                             # From R script (kegg.R)
│   ├── kegg_dotplot.png
│   ├── kegg_dotplot.html
│   ├── pathway_combined_plot.png
│   └── pathway_combined_plot.html
└── top_genes_by_expression.csv        # From workflow
```

### Key Files

| File | Purpose |
|------|---------|
| `agent/simple_magentic_agent.py` | Main agent with FunctionTool wrapper |
| `tools/omic_tools/omic_fetch_analysis_workflow.py` | Orchestrates the full pipeline |
| `tools/omic_tools/omic_analysis.py` | Core DE analysis and enrichment functions |
| `tools/omic_tools/ner_tool.py` | Named entity recognition for queries |
| `tools/omic_tools/subprocess_r.py` | R script execution helper |
| `enrichment/kegg.R` | KEGG pathway visualization (R/plotly) |