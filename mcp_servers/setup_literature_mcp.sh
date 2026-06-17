#!/usr/bin/env bash
set -euo pipefail

# Setup helper for the Literature MCP server.
#
# It prepares the conda environment, local path config, cache directories, and
# paperscraper metadata dumps needed by the PubMed full-text pipeline.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONDA_ENV="${CONDA_ENV:-langgraph-dev}"
INSTALL_DEPS=1
BUILD_DUMPS=1
ALL_DUMPS=0
DUMP_START="${DUMP_START:-2024-01-01}"
DUMP_END="${DUMP_END:-}"
DUMP_WORKERS="${DUMP_WORKERS:-4}"

usage() {
    cat <<EOF
Usage: bash mcp_servers/setup_literature_mcp.sh [options]

Options:
  --conda-env NAME       Conda env to use (default: langgraph-dev)
  --skip-install         Do not pip install Python prerequisites
  --skip-dumps           Do not build paperscraper biorxiv/medrxiv/chemrxiv dumps
  --all-dumps            Build full historical dumps instead of recent dumps
  --dump-start DATE      Dump start date, YYYY-MM-DD (default: 2024-01-01)
  --dump-end DATE        Dump end date, YYYY-MM-DD (default: today)
  --dump-workers N       Workers for biorxiv/medrxiv dump fetch (default: 4)
  -h, --help             Show this help

Environment overrides:
  CONDA_ENV, DUMP_START, DUMP_END, DUMP_WORKERS

Notes:
  - Full historical dumps can take a long time. Recent dumps are enough for
    "current advances" style literature searches and keep setup manageable.
  - API keys for publisher full-text access still belong in .env, e.g.
    ELSEVIER_TDM_API_KEY and WILEY_TDM_API_TOKEN where applicable.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        --skip-install)
            INSTALL_DEPS=0
            shift
            ;;
        --skip-dumps)
            BUILD_DUMPS=0
            shift
            ;;
        --all-dumps)
            ALL_DUMPS=1
            shift
            ;;
        --dump-start)
            DUMP_START="$2"
            shift 2
            ;;
        --dump-end)
            DUMP_END="$2"
            shift 2
            ;;
        --dump-workers)
            DUMP_WORKERS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

echo "Project root: $PROJECT_ROOT"
echo "Conda env:    $CONDA_ENV"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Could not find conda.sh under ~/miniconda3 or ~/anaconda3" >&2
    exit 1
fi

conda activate "$CONDA_ENV"

cd "$PROJECT_ROOT"

if [[ ! -f configs/paths.yaml ]]; then
    echo "Creating configs/paths.yaml from configs/paths.yaml.example"
    cp configs/paths.yaml.example configs/paths.yaml
else
    echo "configs/paths.yaml already exists"
fi

mkdir -p \
    cache/PudMed_DB/jsonl_cache \
    cache/PudMed_DB/doi_cache \
    mcp_servers/literature_outputs \
    logs/mcp-servers

if [[ "$INSTALL_DEPS" -eq 1 ]]; then
    echo "Installing Literature MCP Python prerequisites..."
    python -m pip install paperscraper pymupdf aiofiles google-genai
else
    echo "Skipping Python prerequisite install"
fi

echo "Verifying Python imports..."
python - <<'PY'
mods = [
    "paperscraper",
    "pymupdf",
    "fitz",
    "aiofiles",
    "google.genai",
    "tools.pubmed_tools.query_pubmed_tool",
]
for mod in mods:
    __import__(mod)
    print(f"  OK {mod}")
PY

if [[ "$BUILD_DUMPS" -eq 1 ]]; then
    echo "Building paperscraper server dumps/index..."
    if [[ "$ALL_DUMPS" -eq 1 ]]; then
        echo "  Scope: full historical dumps"
    else
        if [[ -n "$DUMP_END" ]]; then
            echo "  Scope: $DUMP_START to $DUMP_END"
        else
            echo "  Scope: $DUMP_START to today"
        fi
    fi

    python - <<PY
from paperscraper.get_dumps import biorxiv, medrxiv, chemrxiv

all_dumps = bool(int("$ALL_DUMPS"))
start = None if all_dumps else "$DUMP_START"
end = None if all_dumps or not "$DUMP_END" else "$DUMP_END"
workers = int("$DUMP_WORKERS")

kwargs = {"start_date": start, "end_date": end}

print("  Fetching biorxiv dump...")
biorxiv(**kwargs, max_workers=workers)
print("  Fetching medrxiv dump...")
medrxiv(**kwargs, max_workers=workers)
print("  Fetching chemrxiv dump...")
chemrxiv(**kwargs)
print("  paperscraper dumps complete")
PY
else
    echo "Skipping paperscraper dump build"
fi

echo "Literature MCP setup complete."
echo "Start server: python mcp_servers/literature_server.py --sse"
echo "Test server:  python mcp_servers/test_literature_mcp.py --transport sse --include-full-text"
