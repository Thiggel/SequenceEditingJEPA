#!/usr/bin/env bash
# shellcheck shell=bash

set -euo pipefail

# Resolve repository root from this script location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
REPO_NAME="$(basename "$REPO_ROOT")"

# Optional dotenv support for local overrides.
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.env"
  set +a
fi

# Match cluster defaults used in sibling RLVL projects.
if command -v module >/dev/null 2>&1; then
  module load cuda/12.8.1 >/dev/null 2>&1 || true
fi

WORK_ROOT="${WORK:?WORK must be set}"
SCRATCH_ROOT="${SCRATCH:-$WORK_ROOT}"

export SEQ_EDIT_JEPA_REPO_ROOT="$REPO_ROOT"
export SEQ_EDIT_JEPA_REPO_NAME="$REPO_NAME"
export SEQ_EDIT_JEPA_WORK_ROOT="${SEQ_EDIT_JEPA_WORK_ROOT:-$WORK_ROOT/$REPO_NAME}"
export SEQ_EDIT_JEPA_SCRATCH_ROOT="${SEQ_EDIT_JEPA_SCRATCH_ROOT:-$SCRATCH_ROOT/$REPO_NAME}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORK_ROOT/.cache}"

export http_proxy="${http_proxy:-http://proxy:80}"
export https_proxy="${https_proxy:-http://proxy:80}"
export no_proxy="${no_proxy:-localhost,127.0.0.1}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"

export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$XDG_CACHE_HOME/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export TRITON_HOME="${TRITON_HOME:-$XDG_CACHE_HOME/.triton}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$XDG_CACHE_HOME/uv}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-$XDG_CACHE_HOME/tvm-ffi}"
# Optional torch-c-dlpack JIT build in tvm_ffi is slow and repeatedly fails on our torch stack.
# Disable it to avoid startup overhead and spurious writes to $HOME/.cache.
export TVM_FFI_DISABLE_TORCH_C_DLPACK="${TVM_FFI_DISABLE_TORCH_C_DLPACK:-1}"

export WANDB_DIR="${WANDB_DIR:-$SEQ_EDIT_JEPA_WORK_ROOT/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$XDG_CACHE_HOME/wandb}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-$SEQ_EDIT_JEPA_WORK_ROOT/wandb_artifacts}"

export TMPDIR="${TMPDIR:-$SEQ_EDIT_JEPA_SCRATCH_ROOT/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  unset ROCR_VISIBLE_DEVICES || true
fi

mkdir -p \
  "$SEQ_EDIT_JEPA_WORK_ROOT" \
  "$SEQ_EDIT_JEPA_SCRATCH_ROOT" \
  "$XDG_CACHE_HOME" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$HF_MODULES_CACHE" \
  "$VLLM_CACHE_ROOT" \
  "$TORCH_HOME" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$TRITON_HOME" \
  "$UV_CACHE_DIR" \
  "$TVM_FFI_CACHE_DIR" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$WANDB_ARTIFACT_DIR" \
  "$TMPDIR"
