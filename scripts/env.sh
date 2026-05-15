#!/usr/bin/env bash
# shellcheck shell=bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

export VIRTUAL_ENV="${VIRTUAL_ENV:-$WORK/.venv}"
export PATH="$VIRTUAL_ENV/bin:$PATH"
case ":${PYTHONPATH:-}:" in
  *":$SEQ_EDIT_JEPA_REPO_ROOT:"*) ;;
  *) export PYTHONPATH="$SEQ_EDIT_JEPA_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
