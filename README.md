# OmniCellAgent: Towards AI Co-Scientists for Scientific Discovery in Precision Medicine

![](webapp/assets/dash-logo-stripe.svg)

## 🎥 YouTube Video Description (Landing Page)



Meet OmniCellAgent — an AI Co-Scientist for autonomous single-cell omics deep research. This platform combines advanced agentic orchestration systems with bio-focused specialized databases and foundation models to accelerate biomedical discovery. Explore intelligent research automation, transparent step-by-step progress, and rich visual outputs that bring complex analyses to life.

Whether you're exploring disease mechanisms, prioritizing targets, or synthesizing literature and omics data, OmniCellAgent helps you move from questions to insights faster.

Learn more and follow the lab here: https://www.youtube.com/@FuhaiLiAILab

Additional links:
- Lab: https://fuhailiailab.github.io
- GitHub: https://github.com/FuhaiLiAiLab/OmniCellAgent
- Paper: https://www.biorxiv.org/content/10.1101/2025.07.31.667797v1

demo at https://agent.omni-cells.com
 (might not always be up due to maintainence and updates)

## 🤝 Agent-to-Agent (A2A) Protocol Support

OmniCellAgent now supports the **FastA2A protocol**, enabling seamless integration with other AI agents like GitHub Copilot and Claude. This allows external tools to leverage OmniCellAgent's biomedical research capabilities through a standard HTTP API.

**Quick Start:**
```bash
# Start the A2A server (port 8021)
cd fasta2a_service
conda activate a2a-dev
nohup python server.py > server.log 2>&1 &
```

**Key Features:**
- ✅ Full A2A protocol compliance (task submission, polling, artifacts)
- ✅ Async task processing with status tracking
- ✅ Comprehensive biomedical research capabilities (omics, literature, pathways)
- ✅ Compatible with GitHub Copilot, Claude, and other A2A-enabled tools

See [fasta2a_service/README.md](fasta2a_service/README.md) for detailed documentation and API reference.

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



```bash

# Run all default test cases from scratch
source ~/miniconda3/etc/profile.d/conda.sh && conda activate langgraph-dev && python -m agent.langgraph_agent --query "What are the key dysfunctional genes and pathways in pancreatic ductal adenocarcinoma (PDAC)?" --session-id PDAC-test && python -m agent.langgraph_agent --query "What are the key dysfunctional genes and pathways in Alzheimer's Disease?" --session-id AD-test && python -m agent.langgraph_agent --query "What are the key dysfunctional genes and pathways in Lung adenocarcinoma (LUAD)?" --session-id LungCancer-test

```



### 2. Run AI Agent Analysis

**Simple Query (Literature Research):**
```bash
conda run -n langgraph-dev python agent/simple_magentic_agent.py \
  --query "What are the key therapeutic targets for Alzheimer's Disease?"
```

**Full Analysis Pipeline (with Omic Data):**
```bash
conda run -n langgraph-dev python agent/simple_magentic_agent.py \
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
# Via startup script (recommended - starts all services)
bash scripts/startup.sh
# Access locally at http://localhost:8050
# Public access at https://agent.omni-cells.com

# Or standalone
conda run -n langgraph-dev python webapp/index.py
```

The Web UI provides:
- **Responsive Layout**: Auto-adjusts to screen size for optimal viewing
- **Session Management**: Each conversation creates a unique session ID
- **Real-time Progress**: See step-by-step agent reasoning and tool calls
- **Visualization**: Plots and figures from analysis are displayed inline
- **Output Storage**: All session outputs saved in `webapp/sessions/<session_id>/`

### 4. Installation

#### 4.1 Environment & Core Dependencies

```bash
# Create conda environment (Python 3.8+ recommended, 3.10 tested)
conda create -n langgraph-dev python=3.10
conda activate langgraph-dev

# Install graphviz (required for KEGG pathway tools)
conda install anaconda::graphviz

# Install Python dependencies
pip install -r requirements.txt --no-deps
```

**Key Libraries**: The system requires PyTorch and graph-processing libraries compatible with joint GNN and LLM modeling for OmniCellTOSG integration.

#### 4.2 Configuration Files Setup

**Create environment file** (`configs/db.env`):
```bash
# Copy example file
cp configs/db.env.example configs/db.env

# Edit with your credentials
# NEO4J_URI=bolt://localhost:7687
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=your_password
# GOOGLE_API_KEY=your_google_api_key
# OPENAI_API_KEY=your_openai_key
```

**Create paths configuration** (`configs/paths.yaml`):
```bash
# Copy example file
cp configs/paths.yaml.example configs/paths.yaml

# Edit paths to point to your local directories
# Key paths to configure:
# - neo4j_path: Path to Neo4j database directory
# - omnicelltosg_root: Path to OmniCellTOSG dataset
# - sessions_base: Where to store analysis sessions
```

**Example paths.yaml structure**:
```yaml
neo4j:
  database_path: "/path/to/neo4j-community-2025.03.0"
  
omnicelltosg:
  dataset_root: "/path/to/OmniCellTOSG/CellTOSG_dataset_v2"
  checkpoint_dir: "/path/to/checkpoints"

sessions:
  base: "./webapp/sessions"
  
cache:
  author_kb: "./cache/author_kb"
  omic_data: "./cache/omic_data"
```

#### 4.3 Neo4j Database Setup

**Install Neo4j** (version 5.23+ recommended):
```bash
# Follow official instructions for your OS
# https://neo4j.com/docs/operations-manual/current/installation/

# Install required plugins:
# - GenAI plugin: https://neo4j.com/docs/cypher-manual/current/genai-integrations/
# - Graph Data Science library: https://neo4j.com/docs/graph-data-science/current/installation/
```

**Load PrimeKG Dataset**:
- Option 1: Run the Jupyter notebook `data-loading/stark_prime_neo4j_loading.ipynb`
- Option 2: Download database dump from AWS S3: `s3://gds-public-dataset/stark-prime-neo4j523`

**Start Neo4j**:
```bash
# Navigate to Neo4j installation directory
cd /path/to/neo4j-community-2025.03.0

# Start in background
nohup bin/neo4j console > logs/neo4j_log.out 2>&1 &

# Verify it's running
curl http://localhost:7474
```

#### 4.4 OmniCellTOSG Dataset & Model Setup

**Download the dataset**:
```bash
# Option 1: Download from HuggingFace
# Visit: https://huggingface.co/datasets/FuhaiLiAiLab/OmniCellTOSG_Dataset

# Option 2: Use the official repository download script
git clone https://github.com/FuhaiLiAiLab/OmniCellTOSG.git
cd OmniCellTOSG
# Follow download instructions in the repository
```

**Configure dataset path**:
```bash
# Edit configs/paths.yaml and set:
# omnicelltosg:
#   dataset_root: "/path/to/OmniCellTOSG/CellTOSG_dataset_v2"
```

**Download pre-trained model checkpoints**:
```bash
# Create checkpoint directory
mkdir -p checkpoints

# Download OmniCell-v1 weights
# Place in checkpoints/ directory to enable inference
```

**Data Loader Configuration**:
When using OmniCellTOSG in your code:
```python
from tools.omic_tools.data_loader import CellTOSGDataLoader

# Point to your local dataset path
loader = CellTOSGDataLoader(
    root='../OmniCellTOSG/CellTOSG_dataset_v2'
)
```

**Pre-training and Fine-tuning** (optional):
```bash
# Pre-training: Learn topological patterns and interaction mechanisms
python pretrain.py

# Downstream tasks: Disease classification, cell-type identification
python train.py

# Tutorials: Extract cell embeddings
jupyter notebook Tutorial_Cluster_blood.ipynb
```

#### 4.5 Additional Services Setup

**R Environment for KEGG Pathway Analysis**:
```bash
# Install required R packages
cd enrichment
bash install_r_package.sh
```

**Verify all paths are configured**:
```bash
# Check that all required directories exist
python -c "from utils.path_config import get_path; print('Config OK')"
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

# Troubleshooting

## Common Issues

### API Key Issues
Ensure your `.env` file is in the project root with:
```bash
GOOGLE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

Load in Python with:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Neo4j Connection Issues
- Verify Neo4j is running: `curl http://localhost:7474`
- Check credentials in `configs/db.env` match your Neo4j setup
- Ensure ports 7474 and 7687 are not blocked

### OmniCellTOSG Data Loading Issues
- Verify dataset path in `configs/paths.yaml` points to the correct directory
- Ensure you have downloaded the full CellTOSG_dataset_v2
- Check that `df_all` metadata contains required fields: tissue, tissue_general, disease, cell_type

### Graphviz Installation Issues
If KEGG pathway visualization fails, ensure graphviz is installed via conda:
```bash
conda install anaconda::graphviz
```

### Service Connection Issues
Check service logs:
```bash
tail -f logs/service-logs/scientist_tool.log
tail -f logs/service-logs/gretriever_service_output.log
```

## Known Limitations

- **Memory management**: Tool call messages are not preserved in long-running conversations to prevent context overflow (see `autogen_agentchat/teams/_group_chat/_magentic_one/_magentic_one_orchestrator.py` lines 487-488)
- **Tool call summaries**: Summary messages are added to thread instead of full tool responses (line 493)

## Todo
- Add `autogen_ext.memory.canvas` for persistent memory storage
- Implement better context window management for long-running sessions

---

## 🚀 Running the Agent

### Quick Start (Recommended)

**Use the automated startup script**:
```bash
# Start all services (Neo4j, RAG tools, microservices)
bash scripts/startup.sh

# Test all services are running
bash scripts/test_services.sh

# Access Web UI at http://localhost:8050
```

This handles all services automatically. The manual steps below are provided for **troubleshooting** and understanding the system architecture.

### Manual Startup (For Troubleshooting)

Read these steps to understand what `scripts/startup.sh` does internally, or to debug service issues.

#### Step 1: Start Neo4j Database
```bash
# Navigate to Neo4j installation directory
cd /path/to/neo4j-community-2025.03.0

# Start in background
nohup bin/neo4j console > logs/neo4j_log.out 2>&1 &

# Verify it's running
curl http://localhost:7474
```

#### Step 2: Start Scientist RAG Service
```bash
# Option 1: Foreground
python tools/scientist_rag_tools/scientist_tool.py

# Option 2: Background (recommended)
nohup python tools/scientist_rag_tools/scientist_tool.py > logs/scientist_tool_output.log 2>&1 &

# Verify
curl http://localhost:8000/health
```

#### Step 3: Start G-Retriever Service
```bash
# Option 1: Foreground
python tools/gretriever_tools/gretriever_service.py

# Option 2: Background (recommended)  
nohup python tools/gretriever_tools/gretriever_service.py > logs/gretriever_service_output.log 2>&1 &

# Verify
curl http://localhost:8001/health
```

#### Step 4: Run the Agent

**Command Line Interface**:
```bash
# Basic query
python agent/simple_magentic_agent.py \
  --task "What are the key dysfunctional signaling targets in microglia of AD?" \
  --task_id "1" \
  --mode magentic > logs/results.txt

# With LangGraph agent (full pipeline)
python -m agent.langgraph_agent \
  --query "What are the key dysfunctional genes and pathways in pancreatic ductal adenocarcinoma?" \
  --session-id PDAC-test
```

**Web UI**:
```bash
# Start web interface
python webapp/index.py

# Access at http://localhost:8050
# Example query: "What are the key dysfunctional signaling targets in microglia of AD, based on the internal database?"
```

### Stop All Services
```bash
bash scripts/stop_services.sh
```

### Testing Individual Modules

Many modules include testing code in their `__main__` block for easy standalone testing:

```bash
# Test individual tools directly
python tools/scientist_rag_tools/scientist_tool.py    # Starts RAG service
python tools/gretriever_tools/gretriever_service.py   # Starts GRetriever service
python tools/omic_tools/omic_fetch_analysis_workflow.py  # Test omic workflow
python tools/pubmed_tools/query_pubmed_tool.py        # Test PubMed search
python tools/google_search_tools/google_search_w3m.py # Test Google search

# Test utilities
python utils/path_config.py                           # Verify path configuration
python tools/omic_tools/ner_tool.py                   # Test NER extraction
```

This makes it easy to isolate and debug specific components without running the full agent system.

---



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