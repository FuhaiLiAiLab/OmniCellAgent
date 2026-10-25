#!/usr/bin/env bash
# Start (or stop) the knowledge-graph service used by BioMarkerKGAgent:
#   1. Neo4j with the PrimeKG graph (bolt://localhost:7687)
#   2. the G-Retriever API on one GPU (http://localhost:8001)
# Both listen on this node only, so run the agent on the same node.
#
# Usage:  scripts/start_kg_service.sh [start|stop|status]      (default: start)
#
# Settings (environment variables; defaults in brackets):
#   NEO4J_HOME            Neo4j install          [external.neo4j_home in configs/paths.yaml]
#   NEO4J_JAVA_HOME       Java 21 for Neo4j      [/home/di.huang/jdk-21.0.2]
#   PYTHON                omnicellagent python   [/storage3/.../miniconda/omnicellagent/bin/python]
#   GPU                   GPU index              [the GPU with the least memory in use]
#   HF_HOME               Hugging Face cache     [/storage3/.../cache/huggingface]
#   GRETRIEVER_BASE_LLM   base model repo        [unsloth/Meta-Llama-3.1-8B-Instruct]
#
# Logs: logs/gretriever_service.log (G-Retriever), $NEO4J_HOME/logs/ (Neo4j)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/storage3/fs1/fuhai.li/Active/di.huang/cache/miniconda/omnicellagent/bin/python}"
NEO4J_HOME="${NEO4J_HOME:-$("$PYTHON" -c "import yaml; print(yaml.safe_load(open('$PROJECT_ROOT/configs/paths.yaml'))['external']['neo4j_home'])")}"
NEO4J_JAVA_HOME="${NEO4J_JAVA_HOME:-/home/di.huang/jdk-21.0.2}"
HF_HOME="${HF_HOME:-/storage3/fs1/fuhai.li/Active/di.huang/cache/huggingface}"
GRETRIEVER_BASE_LLM="${GRETRIEVER_BASE_LLM:-unsloth/Meta-Llama-3.1-8B-Instruct}"
GPU="${GPU:-}"

LOG_DIR="$PROJECT_ROOT/logs"
SERVICE_LOG="$LOG_DIR/gretriever_service.log"
PID_FILE="$LOG_DIR/gretriever_service.pid"
NEO4J_WAIT_SECONDS=180
GRETRIEVER_WAIT_SECONDS=1200   # loading the 17B-parameter checkpoint takes a few minutes

# Runs gretriever_service.py on the GPU without editing it. Three runtime overrides:
#  - device: the service hard-codes 'cpu'
#  - base LLM: the gated meta-llama repo needs a valid HF token; an ungated copy of the same model
#    supplies config and tokenizer (the fine-tuned weights come from the checkpoint)
#  - tokenizer: the service asks for the slow tokenizer, which Llama 3 does not have (transformers
#    then returns a bool and the service fails on pad_token_id)
read -r -d '' LAUNCHER <<'PY' || true
import os, sys
sys.path.insert(0, os.getcwd())
import torch, transformers, uvicorn
import gretriever_service as svc

DEVICE = "cuda"
_from_pretrained = transformers.AutoTokenizer.from_pretrained
def _fast_tokenizer(*args, **kwargs):
    kwargs["use_fast"] = True
    return _from_pretrained(*args, **kwargs)
transformers.AutoTokenizer.from_pretrained = _fast_tokenizer

class GPUWrapper(svc.GRetrieverWrapper):
    def __init__(self, *args, **kwargs):
        kwargs["device"] = DEVICE
        kwargs["llm"] = os.environ["GRETRIEVER_BASE_LLM"]
        super().__init__(*args, **kwargs)

    def load_model(self):
        super().load_model()
        # torch_geometric's LLM records device='cpu' at construction (it judges 8B params too large
        # for one GPU); .to(device) moves the weights but not this attribute, which places the inputs
        self.llm.device = torch.device(DEVICE)

svc.GRetrieverWrapper = GPUWrapper
uvicorn.run(svc.app, host="0.0.0.0", port=8001, reload=False, workers=1)
PY

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
gretriever_pid() { [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null && cat "$PID_FILE"; }
gretriever_healthy() { curl -sf --max-time 30 http://localhost:8001/health | grep -q '"status":"healthy"'; }

start_neo4j() {
    if port_open 7687; then
        echo "Neo4j: already running (bolt :7687)"
        return
    fi
    echo "Neo4j: starting from $NEO4J_HOME"
    JAVA_HOME="$NEO4J_JAVA_HOME" PATH="$NEO4J_JAVA_HOME/bin:$PATH" "$NEO4J_HOME/bin/neo4j" start
    for ((i = 0; i < NEO4J_WAIT_SECONDS; i += 3)); do
        port_open 7687 && { echo "Neo4j: ready (bolt :7687)"; return; }
        sleep 3
    done
    echo "Neo4j: not reachable after ${NEO4J_WAIT_SECONDS}s; see $NEO4J_HOME/logs/" >&2
    exit 1
}

start_gretriever() {
    if gretriever_healthy; then
        echo "G-Retriever: already running and healthy (:8001)"
        return
    fi
    if ! gretriever_pid >/dev/null; then
        if port_open 8001; then
            echo "G-Retriever: port 8001 is taken by another process; stop it first" >&2
            exit 1
        fi
        if [[ -z "$GPU" ]]; then
            GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                  | sort -t, -k2 -n | head -1 | cut -d, -f1)
        fi
        mkdir -p "$LOG_DIR"
        echo "G-Retriever: starting on GPU $GPU (log: $SERVICE_LOG)"
        cd "$PROJECT_ROOT/tools/gretriever_tools"
        # HF_TOKEN is removed: an invalid token makes even public downloads fail with 401
        CUDA_VISIBLE_DEVICES="$GPU" HF_HOME="$HF_HOME" GRETRIEVER_BASE_LLM="$GRETRIEVER_BASE_LLM" \
            nohup env -u HF_TOKEN "$PYTHON" -u -c "$LAUNCHER" > "$SERVICE_LOG" 2>&1 &
        echo $! > "$PID_FILE"
        cd - >/dev/null
    else
        echo "G-Retriever: process $(gretriever_pid) is starting; waiting"
    fi
    for ((i = 0; i < GRETRIEVER_WAIT_SECONDS; i += 10)); do
        if ! gretriever_pid >/dev/null; then
            echo "G-Retriever: process exited; last log lines:" >&2
            tail -20 "$SERVICE_LOG" >&2
            exit 1
        fi
        if gretriever_healthy; then
            # The first query compiles the model (~1 min); run it here so the agent's first call is fast
            echo "G-Retriever: healthy; warm-up query"
            curl -sf --max-time 600 -X POST http://localhost:8001/query -H 'Content-Type: application/json' \
                -d '{"query": "Which genes are associated with microglial activation?", "max_nodes": 5}' >/dev/null \
                || echo "G-Retriever: warm-up query failed; see $SERVICE_LOG" >&2
            echo "G-Retriever: ready (http://localhost:8001)"
            return
        fi
        sleep 10
    done
    echo "G-Retriever: not healthy after ${GRETRIEVER_WAIT_SECONDS}s; see $SERVICE_LOG" >&2
    exit 1
}

stop_all() {
    if pid=$(gretriever_pid); then
        echo "G-Retriever: stopping process $pid"
        kill "$pid"
        for _ in $(seq 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
    else
        echo "G-Retriever: not running (no live process in $PID_FILE)"
    fi
    rm -f "$PID_FILE"
    JAVA_HOME="$NEO4J_JAVA_HOME" PATH="$NEO4J_JAVA_HOME/bin:$PATH" "$NEO4J_HOME/bin/neo4j" stop || true
}

show_status() {
    port_open 7687 && echo "Neo4j: running (bolt :7687)" || echo "Neo4j: not running"
    if gretriever_healthy; then
        echo "G-Retriever: healthy (:8001, pid $(gretriever_pid || echo unknown))"
    elif gretriever_pid >/dev/null; then
        echo "G-Retriever: process $(gretriever_pid) running but not healthy yet; see $SERVICE_LOG"
    else
        echo "G-Retriever: not running"
    fi
}

case "${1:-start}" in
    start)  start_neo4j; start_gretriever
            echo "Knowledge-graph service is up on $(hostname); run the agent on this node." ;;
    stop)   stop_all ;;
    status) show_status ;;
    *)      echo "Usage: $0 [start|stop|status]" >&2; exit 2 ;;
esac
