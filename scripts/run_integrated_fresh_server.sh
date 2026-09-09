#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"
export PYTHONPATH="$REPOSITORY_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

MODE="${1:-plan}"
PYTHON_BIN="${INTEGRATED_PYTHON:-python}"
PYTHON_RESOLVED="$(command -v "$PYTHON_BIN")"
# A venv's Python can be selected by absolute path while its helper binaries
# (notably Ninja, required by the pinned DDTree C++ extension) remain absent
# from PATH.  Keep the complete selected runtime together.
export PATH="$(dirname "$PYTHON_RESOLVED"):$PATH"
PYTHON_BIN="$PYTHON_RESOLVED"
SUITE="${INTEGRATED_SUITE:-$REPOSITORY_ROOT/configs/adaptive_tree_block_suite.json}"
RUN_DIR="${INTEGRATED_RUN_DIR:-$REPOSITORY_ROOT/outputs/integrated-adaptive-block-fresh}"
ADAPTIVE_DATA_DIR="${ADAPTIVE_DATA_DIR:-$REPOSITORY_ROOT/adaptivetree_paper/datasets/ddtree_official_t0}"
BLOCK_DATA_DIR="${BLOCK_DATA_DIR:-$REPOSITORY_ROOT/datasets/gbv_paper_ddtree_counts}"
DEVICE="${INTEGRATED_DEVICE:-cuda:0}"
CODE_BACKEND="${CODE_BACKEND:-docker}"
LOG_FILE="${RUN_DIR}.log"
PID_FILE="${RUN_DIR}.pid"
STATUS_FILE="${RUN_DIR}.status"

command=(
  "$PYTHON_BIN" -m gbv_experiments run-integrated-suite
  --suite "$SUITE"
  --adaptive-data-dir "$ADAPTIVE_DATA_DIR"
  --block-data-dir "$BLOCK_DATA_DIR"
  --output "$RUN_DIR"
  --device "$DEVICE"
  --code-backend "$CODE_BACKEND"
)

case "$MODE" in
  plan)
    "$PYTHON_BIN" -m gbv_experiments plan-integrated-suite --suite "$SUITE"
    ;;
  audit)
    "$PYTHON_BIN" -m gbv_experiments audit-integrated-suite \
      --suite "$SUITE" --output "${RUN_DIR}.fairness-audit.json"
    ;;
  doctor)
    "$PYTHON_BIN" -m gbv_experiments doctor-integrated-suite \
      --suite "$SUITE" --output "${RUN_DIR}.server-doctor.json" \
      --device "$DEVICE" --code-backend "$CODE_BACKEND"
    ;;
  start)
    if [[ -e "$RUN_DIR" || -e "$LOG_FILE" || -e "$PID_FILE" ]]; then
      echo "Fresh start refused: output/log/PID already exists. Choose a new INTEGRATED_RUN_DIR." >&2
      exit 2
    fi
    mkdir -p "$(dirname "$RUN_DIR")"
    "$PYTHON_BIN" -m gbv_experiments audit-integrated-suite \
      --suite "$SUITE" --output "${RUN_DIR}.fairness-audit.json"
    "$PYTHON_BIN" -m gbv_experiments doctor-integrated-suite \
      --suite "$SUITE" --output "${RUN_DIR}.server-doctor.json" \
      --device "$DEVICE" --code-backend "$CODE_BACKEND"
    nohup "$0" _worker > "$LOG_FILE" 2>&1 < /dev/null &
    worker_pid=$!
    printf '%s\n' "$worker_pid" > "$PID_FILE"
    printf 'started pid=%s run_dir=%s log=%s\n' "$worker_pid" "$RUN_DIR" "$LOG_FILE"
    ;;
  resume)
    if [[ ! -d "$RUN_DIR" ]]; then
      echo "Resume refused: run directory does not exist: $RUN_DIR" >&2
      exit 2
    fi
    if [[ -f "$RUN_DIR/completed.json" ]]; then
      echo "Run is already complete: $RUN_DIR/completed.json"
      exit 0
    fi
    if [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null; then
      echo "Run is already active with PID $(<"$PID_FILE")" >&2
      exit 2
    fi
    nohup "$0" _worker >> "$LOG_FILE" 2>&1 < /dev/null &
    worker_pid=$!
    printf '%s\n' "$worker_pid" > "$PID_FILE"
    printf 'resumed pid=%s run_dir=%s log=%s\n' "$worker_pid" "$RUN_DIR" "$LOG_FILE"
    ;;
  status)
    if [[ -f "$RUN_DIR/completed.json" ]]; then
      echo "complete: $RUN_DIR/completed.json"
    elif [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null; then
      echo "running: PID $(<"$PID_FILE")"
    elif [[ -f "$STATUS_FILE" ]]; then
      echo "stopped: $(<"$STATUS_FILE")"
    else
      echo "not running"
    fi
    if [[ -f "$LOG_FILE" ]]; then
      tail -n 20 "$LOG_FILE"
    fi
    ;;
  _worker)
    set +e
    "${command[@]}"
    exit_code=$?
    printf '%s exit_code=%s run_dir=%s\n' "$(date -Is)" "$exit_code" "$RUN_DIR" > "$STATUS_FILE"
    rm -f "$PID_FILE"
    exit "$exit_code"
    ;;
  *)
    echo "Usage: $0 {plan|audit|doctor|start|resume|status}" >&2
    exit 2
    ;;
esac
