#!/usr/bin/env bash
# Compatibility entry point; defaults to the current DDTree sample-count protocol.
set -euo pipefail
exec bash "$(dirname "$0")/run_gbv_paper.sh" "$@"
