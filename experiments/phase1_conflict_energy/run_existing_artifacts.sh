#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/coder/lhc/CEPO}
cd "$ROOT"

OUT_DIR=${OUT_DIR:-experiments/phase1_conflict_energy/outputs/existing_artifacts}
PYTHON=${PYTHON:-python}

"$PYTHON" experiments/phase1_conflict_energy/scripts/analyze_existing_artifacts.py \
  --repo-root "$ROOT" \
  --output-dir "$OUT_DIR" \
  "$@"
