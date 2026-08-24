#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_ROOT="${LLAMA_ROOT:-/root/autodl-tmp/llama.cpp}"
LLAMA_BUILD_DIR="${LLAMA_BUILD_DIR:-${LLAMA_ROOT}/build-cuda}"
SERVER="${LLAMA_BUILD_DIR}/bin/llama-server"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venvs/bench/bin/python}"
PROMPTS="${PROMPTS:-${PROJECT_ROOT}/datasets/processed/specblock_official/prompts_benchmark_tree15.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/remote_4090}"
MAX_SAMPLES="${MAX_SAMPLES:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
PORT="${PORT:-8080}"

TARGET_REPO="ggml-org/Qwen3.8-27B-GGUF:Q4_K_M"
DRAFT_REPO="incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M"
mkdir -p "${OUTPUT_DIR}" "${PROJECT_ROOT}/logs"
export LLAMA_CACHE="${LLAMA_CACHE:-/root/autodl-tmp/llama_cache}"

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}"
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_for_server() {
  local tries=0
  until curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; do
    tries=$((tries + 1))
    if [[ "${tries}" -ge 360 ]]; then
      echo "llama-server did not become healthy" >&2
      return 1
    fi
    sleep 2
  done
}

run_server_benchmark() {
  local mode="$1"
  shift
  local server_log="${PROJECT_ROOT}/logs/dflash2_${mode}_server.log"
  "${SERVER}" \
    -hf "${TARGET_REPO}" \
    -ngl all \
    -fa on \
    --jinja \
    --ctx-size 4096 \
    --parallel 1 \
    --port "${PORT}" \
    "$@" >"${server_log}" 2>&1 &
  server_pid=$!
  wait_for_server
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/benchmark_llama_server.py" \
    --url "http://127.0.0.1:${PORT}" \
    --prompts "${PROMPTS}" \
    --output "${OUTPUT_DIR}/dflash2_${mode}.jsonl" \
    --implementation "llama.cpp/${mode}" \
    --max-samples "${MAX_SAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}"
  cleanup
  server_pid=""
}

run_server_benchmark baseline
run_server_benchmark speculative \
  -hfd "${DRAFT_REPO}" \
  --spec-type draft-dflash \
  --spec-draft-n-max 7 \
  -ngld all
