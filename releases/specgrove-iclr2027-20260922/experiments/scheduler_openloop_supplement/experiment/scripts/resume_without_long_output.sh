#!/usr/bin/env bash
set -euo pipefail
model=${1:?model}
parent=${2:?parent}
parent_start=${3:?parent_start}
worker=${4:?worker}
worker_start=${5:?worker_start}
case "$model" in qwen3_4b|qwen3_8b) ;; *) exit 2;; esac
proc_matches() {
  [[ -r "/proc/$1/stat" ]] || return 1
  local line rest
  IFS= read -r line < "/proc/$1/stat" || return 1
  rest=${line##*) }
  read -r -a proc_fields <<< "$rest"
  [[ "${proc_fields[19]}" == "$2" ]]
}
while proc_matches "$worker" "$worker_start"; do
  [[ "${proc_fields[0]}" == Z || "${proc_fields[0]}" == X ]] && break
  sleep 5
done
# The old shell is stopped; its existing worker has now exited.
# Check process identity before retiring that shell. No live GPU worker is killed.
if proc_matches "$parent" "$parent_start"; then
  [[ "${proc_fields[0]}" == T || "${proc_fields[0]}" == t ]] || exit 3
  kill -KILL "$parent"
fi
exec bash /root/dp-paper-natural-once-20260917/scripts/run_natural_paper_suite.sh "$model"
