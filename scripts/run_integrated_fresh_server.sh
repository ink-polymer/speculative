#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$REPOSITORY_ROOT/scripts/run_integrated_fresh_server.sh"
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
RUN_DIR="${INTEGRATED_RUN_DIR:-$REPOSITORY_ROOT/outputs/integrated-adaptive-ddtree-dflash-t0-t1}"
ADAPTIVE_DATA_DIR="${ADAPTIVE_DATA_DIR:-$REPOSITORY_ROOT/adaptivetree_paper/datasets/ddtree_official_t0}"
# BLOCK_DATA_DIR remains a compatibility alias for older launch commands. This
# phase contains T=1 Target/DFlash/DDTree plus the registered same-tree verifier.
SAMPLING_DATA_DIR="${SAMPLING_DATA_DIR:-${BLOCK_DATA_DIR:-$REPOSITORY_ROOT/datasets/gbv_paper_ddtree_counts}}"
DEVICE="${INTEGRATED_DEVICE:-cuda:0}"
CODE_BACKEND="${CODE_BACKEND:-docker}"
LOG_FILE="${RUN_DIR}.log"
PID_FILE="${RUN_DIR}.pid"
STATUS_FILE="${RUN_DIR}.status"
LOCK_FILE="${RUN_DIR}.lifecycle.lock"

command=(
  "$PYTHON_BIN" -m gbv_experiments run-integrated-suite
  --suite "$SUITE"
  --adaptive-data-dir "$ADAPTIVE_DATA_DIR"
  --sampling-data-dir "$SAMPLING_DATA_DIR"
  --output "$RUN_DIR"
  --device "$DEVICE"
  --code-backend "$CODE_BACKEND"
)

acquire_lifecycle_lock_and_reexec() {
  local locked_mode="$1"
  mkdir -p "$(dirname "$RUN_DIR")"
  # Python supplies the same kernel flock primitive on Linux and macOS.  The
  # descriptor is made inheritable, then retained by the re-execed launcher
  # and its background worker for the worker's entire lifetime.  Kernel close
  # semantics also make a killed worker's lock recoverable without stale-lock
  # deletion races.
  exec "$PYTHON_BIN" -c '
import errno
import fcntl
import os
import sys

lock_path, script_path, locked_mode = sys.argv[1:]
fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError as error:
    if error.errno not in (errno.EACCES, errno.EAGAIN):
        raise
    print(f"Run lifecycle is already locked: {lock_path}", file=sys.stderr)
    raise SystemExit(2)
os.set_inheritable(fd, True)
environment = os.environ.copy()
environment["INTEGRATED_LIFECYCLE_LOCK_FD"] = str(fd)
os.execve(script_path, [script_path, locked_mode], environment)
' "$LOCK_FILE" "$SCRIPT_PATH" "$locked_mode"
}

verify_inherited_lifecycle_lock() {
  local lock_fd="${INTEGRATED_LIFECYCLE_LOCK_FD:-}"
  if [[ ! "$lock_fd" =~ ^[0-9]+$ ]]; then
    echo "Internal locked mode requires an inherited lifecycle lock." >&2
    return 2
  fi
  "$PYTHON_BIN" -c '
import fcntl
import os
import sys

fd = int(sys.argv[1])
os.fstat(fd)
fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
' "$lock_fd" || {
    echo "Inherited lifecycle lock is invalid." >&2
    return 2
  }
}

write_atomic_line() {
  local destination="$1"
  local value="$2"
  local temporary="${destination}.tmp.$$"
  (umask 077; printf '%s\n' "$value" > "$temporary")
  mv -f "$temporary" "$destination"
}

launch_worker() {
  local action="$1"
  if [[ "$action" == "started" ]]; then
    nohup "$SCRIPT_PATH" _worker > "$LOG_FILE" 2>&1 < /dev/null &
  else
    nohup "$SCRIPT_PATH" _worker >> "$LOG_FILE" 2>&1 < /dev/null &
  fi
  local worker_pid=$!
  # The lifecycle lock serializes writers; rename prevents status readers from
  # ever observing a partial PID.
  write_atomic_line "$PID_FILE" "$worker_pid"
  printf '%s pid=%s run_dir=%s log=%s\n' \
    "$action" "$worker_pid" "$RUN_DIR" "$LOG_FILE"
}

wait_for_pid_ownership() {
  local attempt current_pid
  for ((attempt=0; attempt<500; attempt++)); do
    current_pid=""
    if [[ -f "$PID_FILE" ]]; then
      IFS= read -r current_pid < "$PID_FILE" || true
    fi
    if [[ "$current_pid" == "$$" ]]; then
      return 0
    fi
    sleep 0.01
  done
  echo "Worker did not receive PID-file ownership: expected $$" >&2
  return 2
}

remove_pid_if_owned() {
  local current_pid=""
  if [[ -f "$PID_FILE" ]]; then
    IFS= read -r current_pid < "$PID_FILE" || true
  fi
  if [[ "$current_pid" == "$$" ]]; then
    rm -f -- "$PID_FILE"
  fi
}

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
    acquire_lifecycle_lock_and_reexec _start_locked
    ;;
  _start_locked)
    verify_inherited_lifecycle_lock
    if [[ -e "$RUN_DIR" || -e "$LOG_FILE" || -e "$PID_FILE" ]]; then
      echo "Fresh start refused: output/log/PID already exists. Choose a new INTEGRATED_RUN_DIR." >&2
      exit 2
    fi
    mkdir -p "$(dirname "$RUN_DIR")"
    "$PYTHON_BIN" "$REPOSITORY_ROOT/scripts/audit_formal_experiment_matrix.py" \
      --matrix "$REPOSITORY_ROOT/configs/formal_experiment_matrix.json" \
      --output "${RUN_DIR}.matrix-audit.json"
    "$PYTHON_BIN" -m gbv_experiments audit-integrated-suite \
      --suite "$SUITE" --output "${RUN_DIR}.fairness-audit.json"
    "$PYTHON_BIN" -m gbv_experiments doctor-integrated-suite \
      --suite "$SUITE" --output "${RUN_DIR}.server-doctor.json" \
      --device "$DEVICE" --code-backend "$CODE_BACKEND"
    # Establish the resumable run identity before the background worker starts.
    # If it dies during its own repeated doctor/preflight, resume must still have
    # an explicit run directory to target.
    mkdir "$RUN_DIR"
    launch_worker started
    ;;
  resume)
    acquire_lifecycle_lock_and_reexec _resume_locked
    ;;
  _resume_locked)
    verify_inherited_lifecycle_lock
    if [[ ! -d "$RUN_DIR" ]]; then
      echo "Resume refused: run directory does not exist: $RUN_DIR" >&2
      exit 2
    fi
    if [[ -f "$RUN_DIR/completed.json" ]]; then
      echo "Run is already complete: $RUN_DIR/completed.json"
      exit 0
    fi
    # Acquiring the kernel lock proves that no prior worker is active.  A PID
    # left by SIGKILL is stale and is atomically replaced by launch_worker.
    launch_worker resumed
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
    verify_inherited_lifecycle_lock
    wait_for_pid_ownership
    trap remove_pid_if_owned EXIT
    set +e
    "${command[@]}"
    exit_code=$?
    write_atomic_line "$STATUS_FILE" \
      "$(date -Is) exit_code=$exit_code run_dir=$RUN_DIR"
    exit "$exit_code"
    ;;
  *)
    echo "Usage: $0 {plan|audit|doctor|start|resume|status}" >&2
    exit 2
    ;;
esac
