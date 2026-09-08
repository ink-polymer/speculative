#!/usr/bin/env bash
# Compatibility entry point for the complete two-model study.
set -euo pipefail
exec bash "$(dirname "$0")/run_gbv_paper.sh" complete "$@"
