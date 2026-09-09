#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"
export PYTHONPATH="$REPOSITORY_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$#" -eq 0 ]]; then
  set -- plan-integrated-suite --suite configs/adaptive_tree_block_suite.json
fi

"${INTEGRATED_PYTHON:-python}" -m gbv_experiments "$@"
