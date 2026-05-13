#!/usr/bin/env bash
#
# scripts/run_full_pipeline.sh
# =============================
#
# Single entrypoint that runs the entire OmniCellAgent demo pipeline end
# to end without manual hand-holding:
#
#   1. Start microservices (delegates to ``scripts/startup.sh``).
#   2. Wait for ports to come up.
#   3. Run the LangGraph agent on 3 case studies (PDAC, AD, LungCancer)
#      sequentially under unique session IDs.
#   4. Run AI peer reviews (apr + litllm + openrev) on the fresh reports
#      into a fresh ai_review_results dir so v1/v2 outputs are untouched.
#   5. Run improve_and_review.py to revise each report from the aggregated
#      reviewer feedback, then re-review the revised PDFs.
#   6. Compile the original + revised PDFs into a single supplementary
#      bundle under ``logs/appendix/``.
#
# All v-1 / v-2 artefacts stay where they are; this script writes to
# new directories tagged with ``$SUFFIX`` (default ``-test-3``).
#
# Usage
# -----
#   bash scripts/run_full_pipeline.sh                # default suffix -test-3
#   SUFFIX=-test-4 bash scripts/run_full_pipeline.sh # custom version tag
#   SKIP_SERVICES=1 bash scripts/run_full_pipeline.sh  # if services already up
#   SKIP_AGENT=1    bash scripts/run_full_pipeline.sh  # reuse prior agent runs
#
# Re-running is safe: agent session names embed the suffix, report files
# carry timestamps, and the review/compile outputs use suffix-derived names.
#
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
#  Configuration (override via env vars)
# ──────────────────────────────────────────────────────────────────────
SUFFIX="${SUFFIX:--test-3}"             # e.g. -test-3 → AD-test-3, ai_review_results_test-3
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$PROJECT_ROOT/scripts"
LOG_ROOT="$PROJECT_ROOT/logs"
SVC_LOG_DIR="$LOG_ROOT/service-logs"
PIPELINE_LOG_DIR="$LOG_ROOT/pipeline-runs"
mkdir -p "$PIPELINE_LOG_DIR"

# Sanitised suffix for filenames (strip leading dash, replace separators).
SUFFIX_SANITIZED="$(echo "${SUFFIX#-}" | tr ' /' '__')"
RESULTS_DIR_NAME="ai_review_results_${SUFFIX_SANITIZED}"
COMPILED_PDF_NAME="supplementary_reports_${SUFFIX_SANITIZED}.pdf"

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="$PIPELINE_LOG_DIR/run_${SUFFIX_SANITIZED}_${RUN_STAMP}.log"

# Resolve interpreters directly — `conda run` is unreliable in non-interactive shells.
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
LG_PY="$CONDA_BASE/envs/langgraph-dev/bin/python"
AGENTBENCH_PY="$CONDA_BASE/envs/agentbench/bin/python"
for py in "$LG_PY" "$AGENTBENCH_PY"; do
  if [ ! -x "$py" ]; then
    echo "ERROR: python interpreter not found at $py" >&2
    echo "       Set CONDA_BASE or fix env names in this script." >&2
    exit 2
  fi
done

# Cases — keep the queries verbatim from README so this reproduces section 3.
CASE_KEYS=("PDAC" "AD" "LungCancer")
declare -A CASE_QUERY=(
  ["PDAC"]="What are the key dysfunctional genes and pathways in pancreatic ductal adenocarcinoma (PDAC)?"
  ["AD"]="What are the key dysfunctional genes and pathways in Alzheimer's Disease?"
  ["LungCancer"]="What are the key dysfunctional genes and pathways in Lung adenocarcinoma (LUAD)?"
)

# Whether to redo each stage; defaults run everything.
SKIP_SERVICES="${SKIP_SERVICES:-0}"
SKIP_AGENT="${SKIP_AGENT:-0}"
SKIP_REVIEW="${SKIP_REVIEW:-0}"
SKIP_REVISE="${SKIP_REVISE:-0}"
SKIP_COMPILE="${SKIP_COMPILE:-0}"

# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────
log()  { printf '\n[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$RUN_LOG"; }
fail() { printf '\n[%s] ERROR: %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$RUN_LOG" >&2; exit 1; }

banner() {
  local label="$1"
  printf '\n%s\n' "════════════════════════════════════════════════════════════" | tee -a "$RUN_LOG"
  printf '  %s\n' "$label"  | tee -a "$RUN_LOG"
  printf '%s\n'   "════════════════════════════════════════════════════════════" | tee -a "$RUN_LOG"
}

wait_for_port() {
  # wait_for_port <host> <port> <timeout_seconds>
  local host="$1" port="$2" max="$3" elapsed=0
  while (( elapsed < max )); do
    if ss -lnt 2>/dev/null | awk '{print $4}' | grep -qE "(^|:)$port$"; then
      return 0
    fi
    sleep 2; elapsed=$((elapsed + 2))
  done
  return 1
}

# ──────────────────────────────────────────────────────────────────────
#  Phase 1 — services
# ──────────────────────────────────────────────────────────────────────
banner "1/5  Starting microservices"
if [[ "$SKIP_SERVICES" == "1" ]]; then
  log "SKIP_SERVICES=1 — assuming services already up"
else
  bash "$SCRIPT_DIR/startup.sh" | tee -a "$RUN_LOG"
  log "Waiting for ports 7474, 7687, 8000, 8001, 8003, 8050 …"
  for port in 7474 7687 8000 8001 8003 8050; do
    if wait_for_port 127.0.0.1 "$port" 60; then
      log "  ✓ port $port listening"
    else
      log "  ✗ port $port NOT listening after 60s — check $SVC_LOG_DIR"
    fi
  done
fi

# ──────────────────────────────────────────────────────────────────────
#  Phase 2 — agent runs
# ──────────────────────────────────────────────────────────────────────
banner "2/5  Running LangGraph agent on ${#CASE_KEYS[@]} case studies (suffix='$SUFFIX')"
if [[ "$SKIP_AGENT" == "1" ]]; then
  log "SKIP_AGENT=1 — skipping agent runs (will look for existing report files)"
else
  for ck in "${CASE_KEYS[@]}"; do
    session_id="${ck}${SUFFIX}"
    query="${CASE_QUERY[$ck]}"
    log "→ $ck   session='$session_id'   query='$query'"
    "$LG_PY" -m agent.langgraph_agent \
      --query "$query" \
      --session-id "$session_id" 2>&1 | tee -a "$RUN_LOG"
  done
  log "All ${#CASE_KEYS[@]} agent runs complete."
fi

# ──────────────────────────────────────────────────────────────────────
#  Phase 3a — initial AI peer reviews
# ──────────────────────────────────────────────────────────────────────
banner "3/5  AI peer reviews on first-run reports → benchmark/$RESULTS_DIR_NAME"
if [[ "$SKIP_REVIEW" == "1" ]]; then
  log "SKIP_REVIEW=1 — skipping initial AI reviews"
else
  "$AGENTBENCH_PY" "$PROJECT_ROOT/benchmark/run_ai_review.py" \
    "--session-suffix=$SUFFIX" \
    "--output-dir=$PROJECT_ROOT/benchmark/$RESULTS_DIR_NAME" \
    --reviewer apr litllm openrev 2>&1 | tee -a "$RUN_LOG"
  log "Initial AI reviews complete."
fi

# ──────────────────────────────────────────────────────────────────────
#  Phase 3b — revise + re-review
# ──────────────────────────────────────────────────────────────────────
banner "4/5  Revising reports + re-reviewing (improve_and_review.py)"
if [[ "$SKIP_REVISE" == "1" ]]; then
  log "SKIP_REVISE=1 — skipping revision + re-review"
else
  "$LG_PY" "$PROJECT_ROOT/benchmark/improve_and_review.py" \
    "--session-suffix=$SUFFIX" \
    "--results-dir=$RESULTS_DIR_NAME" 2>&1 | tee -a "$RUN_LOG"
  log "Revision + re-review complete."
fi

# ──────────────────────────────────────────────────────────────────────
#  Phase 4 — compile supplementary PDF
# ──────────────────────────────────────────────────────────────────────
banner "5/5  Compiling supplementary PDF → logs/appendix/$COMPILED_PDF_NAME"
if [[ "$SKIP_COMPILE" == "1" ]]; then
  log "SKIP_COMPILE=1 — skipping PDF compile"
else
  "$LG_PY" "$SCRIPT_DIR/combine_supplementary_pdfs.py" \
    "--session-suffix=$SUFFIX" \
    "--output=$COMPILED_PDF_NAME" \
    --skip-regenerate 2>&1 | tee -a "$RUN_LOG"
  log "Compile complete."
fi

# ──────────────────────────────────────────────────────────────────────
#  Summary
# ──────────────────────────────────────────────────────────────────────
banner "DONE"
log "Pipeline run for suffix='$SUFFIX' finished."
log "  Sessions:        webapp/sessions/{PDAC,AD,LungCancer}${SUFFIX}/"
log "  AI reviews:      benchmark/$RESULTS_DIR_NAME/"
log "  Supplementary:   logs/appendix/$COMPILED_PDF_NAME"
log "  Full log:        $RUN_LOG"
