#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# run_all_evals.sh — Full AI Review Benchmark (all reviewers, all cases)
# ═══════════════════════════════════════════════════════════════════════
#
# Handles the GPU resource conflict automatically:
#   1. Stops Ollama (frees ~20 GB VRAM)
#   2. Kills any orphaned GPU processes (vLLM engines, etc.)
#   3. Runs each reviewer group sequentially
#   4. Restarts Ollama when done
#
# Usage:
#   bash benchmark/run_all_evals.sh              # run everything
#   bash benchmark/run_all_evals.sh --no-apr      # skip APR (slow, API-heavy)
#   bash benchmark/run_all_evals.sh --only sea    # run just one reviewer
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="agentbench"
RESULTS="$ROOT/benchmark/ai_review_results"
LOG="$RESULTS/run_all.log"
PYTHON="conda run -n $CONDA_ENV --live-stream python"

# ── Parse flags ──────────────────────────────────────────────────────
SKIP_APR=false
ONLY=""
OVERWRITE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-apr)    SKIP_APR=true ;;
    --only)      shift; ONLY="$1" ;;
    --only=*)    ONLY="${1#--only=}" ;;
    --overwrite) OVERWRITE=true ;;
    *)           echo "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done

# Build common extra args
EXTRA_ARGS=()
if $OVERWRITE; then
  EXTRA_ARGS+=(--overwrite)
fi

# ── Helpers ──────────────────────────────────────────────────────────
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

log() {
  local msg="[$(timestamp)] $*"
  echo "$msg"
  echo "$msg" >> "$LOG"
}

gpu_free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | head -1 | tr -d ' '
}

wait_gpu_free() {
  local target=${1:-20000}  # need at least 20 GB free
  local waited=0
  while (( $(gpu_free_mb) < target )); do
    if (( waited > 30 )); then
      log "⚠️  GPU still not free after 30s — continuing anyway"
      return
    fi
    sleep 2
    waited=$((waited + 2))
  done
}

kill_gpu_compute() {
  # Kill any non-display GPU processes (type=C) except our own
  local pids
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  for p in $pids; do
    local name
    name=$(ps -p "$p" -o comm= 2>/dev/null || true)
    log "  ⚡ Killing leftover GPU process $p ($name)"
    kill "$p" 2>/dev/null || sudo kill "$p" 2>/dev/null || true
  done
  sleep 2
}

run_reviewer() {
  local reviewer="$1"
  shift
  log "━━━ Running: $reviewer $*"
  local t0
  t0=$(date +%s)

  if $PYTHON benchmark/run_ai_review.py \
      --reviewer "$reviewer" \
      --overwrite \
      "$@" \
      2>&1 | tee -a "$LOG"; then
    local elapsed=$(( $(date +%s) - t0 ))
    log "✅ $reviewer done (${elapsed}s)"
  else
    log "❌ $reviewer FAILED (exit $?)"
  fi
  echo
}

# ── Ensure output dir + log ──────────────────────────────────────────
mkdir -p "$RESULTS"
: > "$LOG"

log "═══════════════════════════════════════════════════════════"
log "  OmniCellAgent — Full AI Review Benchmark"
log "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
log "  Conda env: $CONDA_ENV"
log "═══════════════════════════════════════════════════════════"
echo

# ══════════════════════════════════════════════════════════════════════
# Phase 1: Stop Ollama and free GPU
# ══════════════════════════════════════════════════════════════════════
log "🔧 Phase 1: Freeing GPU resources"

OLLAMA_WAS_ACTIVE=false
if systemctl is-active --quiet ollama 2>/dev/null; then
  OLLAMA_WAS_ACTIVE=true
  log "  Stopping Ollama service..."
  sudo systemctl stop ollama
  sleep 3
fi

kill_gpu_compute
wait_gpu_free 20000
log "  GPU free: $(gpu_free_mb) MiB"
echo

# ══════════════════════════════════════════════════════════════════════
# Phase 2: Run reviewers
# ══════════════════════════════════════════════════════════════════════
log "🔬 Phase 2: Running reviewers"

T_START=$(date +%s)

if [[ -n "$ONLY" ]]; then
  # ── Single reviewer mode ──
  if [[ "$ONLY" == "apr" ]]; then
    run_reviewer apr --models gpt4-o1,gpt4-o3-mini "${EXTRA_ARGS[@]}"
  else
    run_reviewer "$ONLY" "${EXTRA_ARGS[@]}"
  fi
else
  # ── Group A: API-only reviewers (no GPU needed) ──────────────────
  log "── Group A: API-based reviewers ──"

  if ! $SKIP_APR; then
    run_reviewer apr --models gpt4-o1,gpt4-o3-mini "${EXTRA_ARGS[@]}"
  else
    log "  ⏭️  Skipping APR (--no-apr)"
  fi

  run_reviewer litllm "${EXTRA_ARGS[@]}"

  # ── Group B: Lightweight GPU (ReviewAdvisor BERT tagger) ─────────
  log "── Group B: ReviewAdvisor (BERT tagger, ~500 MiB) ──"
  run_reviewer reviewadvisor "${EXTRA_ARGS[@]}"

  # ── Group C: Heavy GPU models (one at a time) ───────────────────
  log "── Group C: Heavy GPU models (4-bit quantized) ──"

  # SEA — ECNU-SEA/SEA-E (~5 GiB in 4-bit)
  kill_gpu_compute
  wait_gpu_free 18000
  run_reviewer sea "${EXTRA_ARGS[@]}"

  # OpenReviewer — maxidl/Llama-OpenReviewer-8B (~5 GiB in 4-bit)
  kill_gpu_compute
  wait_gpu_free 18000
  run_reviewer openrev "${EXTRA_ARGS[@]}"

  # CycleReviewer — gated model, skipped until HF access is granted
  # kill_gpu_compute
  # wait_gpu_free 18000
  # run_reviewer cycle "${EXTRA_ARGS[@]}"
fi

T_TOTAL=$(( $(date +%s) - T_START ))
echo
log "⏱️  All reviewers finished in ${T_TOTAL}s"

# ══════════════════════════════════════════════════════════════════════
# Phase 3: Restart Ollama
# ══════════════════════════════════════════════════════════════════════
log "🔧 Phase 3: Restoring Ollama"

kill_gpu_compute

if $OLLAMA_WAS_ACTIVE; then
  sudo systemctl start ollama
  log "  ✅ Ollama restarted"
else
  log "  ℹ️  Ollama was not running before — left stopped"
fi

# ══════════════════════════════════════════════════════════════════════
# Phase 4: Summary
# ══════════════════════════════════════════════════════════════════════
echo
log "═══════════════════════════════════════════════════════════"
log "  📊 Results Summary"
log "═══════════════════════════════════════════════════════════"

# Print per-reviewer/case status from result.json files
for case in AD LungCancer PDAC; do
  echo
  echo "  📄 $case"
  for reviewer in apr litllm reviewadvisor sea openrev; do
    rfile="$RESULTS/$case/$reviewer/result.json"
    if [[ -f "$rfile" ]]; then
      ok=$(python3 -c "import json; r=json.load(open('$rfile')); print('✅' if r.get('success') else '❌')" 2>/dev/null || echo "?")
      secs=$(python3 -c "import json; r=json.load(open('$rfile')); print(f\"{r.get('elapsed_seconds',0):.0f}s\")" 2>/dev/null || echo "?")
      score=$(python3 -c "import json; r=json.load(open('$rfile')); s=r.get('score','—'); print(s if s else '—')" 2>/dev/null || echo "—")
      printf "    %-15s %s  %4s  score=%s\n" "$reviewer" "$ok" "$secs" "$score"
    else
      printf "    %-15s ⏭️  not run\n" "$reviewer"
    fi
  done
done

echo
log "  Total time: ${T_TOTAL}s"
log "  Output:     $RESULTS/"
log "  Log:        $LOG"

# Print scores CSV if it exists
if [[ -f "$RESULTS/scores.csv" ]]; then
  echo
  log "  📊 Score Table (scores.csv):"
  column -t -s',' "$RESULTS/scores.csv" 2>/dev/null || cat "$RESULTS/scores.csv"
fi

log "═══════════════════════════════════════════════════════════"
