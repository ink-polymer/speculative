#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${PROJECT_ROOT}/outputs/remote_4090_download_20260823"
SOCKET="/private/tmp/refined_sd_ssh_13802.sock"
HOST="root@connect.cqa1.seetacloud.com"
PORT="13802"
REMOTE_ROOT="/root/autodl-tmp/refined_sd"

mkdir -p "${DEST}/checkpoints" "${DEST}/inputs"
echo "watch_started=$(date -Iseconds)"

while ssh -S "${SOCKET}" -p "${PORT}" "${HOST}" \
  "tmux has-session -t bench_2k 2>/dev/null"; do
  sleep 60
done

echo "remote_tmux_finished=$(date -Iseconds)"
for attempt in 1 2 3 4 5; do
  if scp -r -o "ControlPath=${SOCKET}" -P "${PORT}" \
      "${HOST}:${REMOTE_ROOT}/outputs/remote_4090/2k_bf16" "${DEST}/"; then
    break
  fi
  echo "result_download_retry=${attempt}"
  sleep 30
done

scp -r -o "ControlPath=${SOCKET}" -P "${PORT}" \
  "${HOST}:${REMOTE_ROOT}/logs" "${DEST}/" || true
scp -o "ControlPath=${SOCKET}" -P "${PORT}" \
  "${HOST}:${REMOTE_ROOT}/checkpoints/rank_head_tree15_retrained_2k.pt" \
  "${DEST}/checkpoints/" || true
scp -o "ControlPath=${SOCKET}" -P "${PORT}" \
  "${HOST}:${REMOTE_ROOT}/datasets/processed/specblock_official/prompts_benchmark.jsonl" \
  "${DEST}/inputs/" || true
scp -o "ControlPath=${SOCKET}" -P "${PORT}" \
  "${HOST}:${REMOTE_ROOT}/datasets/generated/rank_train_tree15_disjoint_2k.jsonl" \
  "${DEST}/inputs/" || true
scp -o "ControlPath=${SOCKET}" -P "${PORT}" \
  "${HOST}:${REMOTE_ROOT}/third_party/OFFICIAL_SOURCES.md" \
  "${DEST}/inputs/" || true

find "${DEST}" -type f -exec shasum -a 256 {} + > "${DEST}/download_sha256.txt"
echo "collection_finished=$(date -Iseconds)" | tee "${DEST}/AUTO_COLLECTION_COMPLETE.txt"
ssh -S "${SOCKET}" -O exit -p "${PORT}" "${HOST}" || true
