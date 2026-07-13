#!/usr/bin/env bash
set -euo pipefail

printf 'Retired: controlled experiments are single-CLS hierarchy/rollout only; no full-grid or Delta gate may be submitted.\n' >&2
exit 2
