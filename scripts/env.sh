#!/usr/bin/env bash
# shellcheck shell=bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../env.sh"

if [[ -x "$PUZZLE_JEPA_WORK_ROOT/.venv/bin/python" ]]; then
  export VIRTUAL_ENV="$PUZZLE_JEPA_WORK_ROOT/.venv"
else
  export VIRTUAL_ENV="${VIRTUAL_ENV:-$WORK/.venv}"
fi
export PATH="$VIRTUAL_ENV/bin:$PATH"
case ":${PYTHONPATH:-}:" in
  *":$PUZZLE_JEPA_REPO_ROOT:"*) ;;
  *) export PYTHONPATH="$PUZZLE_JEPA_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
