#!/usr/bin/env bash
# shellcheck shell=bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
REPO_NAME="$(basename "$REPO_ROOT")"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.env"
  set +a
fi

if command -v module >/dev/null 2>&1; then
  module load cuda/12.8.1 >/dev/null 2>&1 || true
fi

WORK_ROOT="${WORK:?WORK must be set}"
SCRATCH_ROOT="${SCRATCH:-$WORK_ROOT}"
VAULT_ROOT="${VAULT:-/home/vault/$(id -gn)/$USER}"
DEFAULT_WORK_ROOT="$WORK_ROOT/$REPO_NAME"
if [[ -d "$VAULT_ROOT" && -w "$VAULT_ROOT" ]]; then
  DEFAULT_WORK_ROOT="$VAULT_ROOT/$REPO_NAME"
fi

export PUZZLE_JEPA_REPO_ROOT="$REPO_ROOT"
export PUZZLE_JEPA_WORK_ROOT="${PUZZLE_JEPA_WORK_ROOT:-$DEFAULT_WORK_ROOT}"
export PUZZLE_JEPA_SCRATCH_ROOT="${PUZZLE_JEPA_SCRATCH_ROOT:-$SCRATCH_ROOT/$REPO_NAME}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PUZZLE_JEPA_WORK_ROOT/.cache}"

export http_proxy="${http_proxy:-http://proxy:80}"
export https_proxy="${https_proxy:-http://proxy:80}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"

export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$XDG_CACHE_HOME/uv}"

export TMPDIR="${TMPDIR:-$PUZZLE_JEPA_SCRATCH_ROOT/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  unset ROCR_VISIBLE_DEVICES || true
fi

mkdir -p \
  "$PUZZLE_JEPA_WORK_ROOT" \
  "$PUZZLE_JEPA_SCRATCH_ROOT" \
  "$XDG_CACHE_HOME" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$TORCH_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$UV_CACHE_DIR" \
  "$TMPDIR"
