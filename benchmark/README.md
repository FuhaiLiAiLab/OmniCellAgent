# Benchmark: Automated AI Review of Case Study Reports

Automated evaluation of OmniCellAgent research reports using multiple
peer-review frameworks from the ML/NLP literature.

## Scripts

| Script | Purpose |
|--------|---------|
| `run_peer_review.py` | Original single-framework runner (ai-peer-review only) |
| `run_ai_review.py` | **Unified runner** — supports 6 frameworks via `--reviewer` |

## Supported Review Frameworks

| Key | Framework | Paper | Requirements |
|-----|-----------|-------|-------------|
| `apr` | [AI Peer Review](https://github.com/poldrack/ai-peer-review) | — | API keys (OpenAI, Google) |
| `cycle` | [CycleReviewer](https://arxiv.org/abs/2411.00816) | ICLR 2025 | `pip install ai_researcher` (vLLM + GPU) |
| `openrev` | [OpenReviewer](https://arxiv.org/abs/2412.11948) | NAACL 2025 | OpenRouter API key **or** transformers + GPU |
| `sea` | [SEA Framework](https://arxiv.org/abs/2407.12857) | EMNLP 2024 | transformers + GPU |
| `reviewadvisor` | [ReviewAdvisor](https://arxiv.org/abs/2202.00176) | — | Git clone + trained tagger |
| `litllm` | [LitLLM](https://arxiv.org/abs/2402.01788) | 2024 | Git clone + S2 API key |

## Case Studies

| Key | Disease | Session | Report |
|-----|---------|---------|--------|
| AD | Alzheimer's Disease | `AD-test` | `report_20260311_140104` |
| LungCancer | Lung Adenocarcinoma | `LungCancer-test` | `report_20260311_142036` |
| PDAC | Pancreatic Ductal Adenocarcinoma | `PDAC-test` | `report_20260311_131601` |

## Setup

```bash
conda activate agentbench   # Python >= 3.13

# Core (already installed)
pip install git+https://github.com/poldrack/ai-peer-review.git

# Optional — GPU-based reviewers
pip install ai_researcher           # CycleReviewer (needs vLLM + GPU)
pip install transformers torch      # OpenReviewer & SEA (needs GPU)

# Optional — repo-based reviewers
git clone https://github.com/neulab/ReviewAdvisor      benchmark/repos/ReviewAdvisor
git clone https://github.com/shubhamagarwal92/LitLLM   benchmark/repos/LitLLM

# API keys (set in .env at project root or export)
export OPENAI_API_KEY="..."
export GOOGLE_API_KEY="..."
export OPENROUTER_API_KEY="..."   # optional, for OpenReviewer without GPU
```

## Quick Start

```bash
conda activate agentbench
cd /path/to/OmniCellAgent

# Show which reviewers are available
python benchmark/run_ai_review.py --list

# Run all available reviewers on all 3 cases
python benchmark/run_ai_review.py

# Run specific reviewer(s)
python benchmark/run_ai_review.py --reviewer apr cycle

# Single case study
python benchmark/run_ai_review.py --case AD --overwrite
```

## Output

```
benchmark/ai_review_results/
├── AD/
│   ├── apr/                       # AI Peer Review outputs
│   │   ├── review_gemini-2.5-pro.md
│   │   ├── review_gpt4-o1.md
│   │   ├── review_gpt4-o3-mini.md
│   │   ├── meta_review.md
│   │   └── result.json
│   ├── cycle/                     # CycleReviewer outputs
│   │   ├── review.md
│   │   └── raw.json
│   ├── openrev/                   # OpenReviewer outputs
│   │   └── review.md
│   └── …
├── LungCancer/
├── PDAC/
└── summary.json
```

## API Keys

| Variable | Used by |
|----------|---------|
| `GOOGLE_API_KEY` | Gemini 2.5 Pro (ai-peer-review) |
| `OPENAI_API_KEY` | GPT-4o models (ai-peer-review), LitLLM |
| `OPENROUTER_API_KEY` | OpenReviewer (API mode, no GPU) |
| `ANTHROPIC_API_KEY` | Claude (optional, ai-peer-review) |

## Environment Variables (repo-based reviewers)

| Variable | Default | Purpose |
|----------|---------|---------|
| `REVIEWADVISOR_DIR` | `benchmark/repos/ReviewAdvisor` | Path to cloned ReviewAdvisor repo |
| `LITLLM_DIR` | `benchmark/repos/LitLLM` | Path to cloned LitLLM repo |
| `SEA_DIR` | `benchmark/repos/SEA` | Path to cloned SEA repo |
| `OPENREV_MODEL` | `maxidl/Llama-OpenReviewer-8B` | HuggingFace model ID for OpenReviewer |
