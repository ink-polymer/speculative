#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_ROOT="${VENV_ROOT:-${PROJECT_ROOT}/.venvs}"

DFLASH_ENV="${VENV_ROOT}/dflash-official"
DDTREE_ENV="${VENV_ROOT}/ddtree-official"

"${PYTHON_BIN}" -m venv "${DFLASH_ENV}"
"${DFLASH_ENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${DFLASH_ENV}/bin/python" -m pip install -e "${PROJECT_ROOT}/third_party/dflash_official[local]"

"${PYTHON_BIN}" -m venv "${DDTREE_ENV}"
"${DDTREE_ENV}/bin/python" -m pip install --upgrade pip setuptools wheel
# FlashAttention needs torch and ninja to be importable while its extension is built.
# Install the official requirement set in that dependency-safe order.
"${DDTREE_ENV}/bin/python" -m pip install \
  torch transformers datasets numpy loguru tqdm matplotlib typing_extensions ninja
"${DDTREE_ENV}/bin/python" -m pip install flash-attn --no-build-isolation

"${DFLASH_ENV}/bin/python" "${PROJECT_ROOT}/scripts/check_official_reference_env.py" --implementation dflash
"${DDTREE_ENV}/bin/python" "${PROJECT_ROOT}/scripts/check_official_reference_env.py" --implementation ddtree
