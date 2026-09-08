#!/usr/bin/env bash
# 5-seed ensemble evaluation on the CodeEpi PyG release test split.
#
# This is the supported public evaluation entry point. It:
#   1. runs each of the 5 released student_seedN.pt on the val + test splits,
#   2. averages the val probabilities across the 5 seeds,
#   3. searches a single macro threshold ``thr_val_search_macro`` on that
#      averaged val,
#   4. applies that threshold to (a) the ensemble-averaged test probability
#      for the headline metric, and (b) each of the 5 per-seed test
#      probabilities for the mean±std block.
#
# Checkpoints: pass 5 student_seedN.pt paths as arguments to evaluate any
# set of seeds (e.g. freshly retrained ones under outputs/reproduce/); with
# no arguments it falls back to the released bundle under
# checkpoints_CodeEpi_v1.0/seeds/. Either layout is accepted by
# codeepi.evaluate.student_ensemble.
#
# Output: written to a fresh, timestamped file under results/reproduce/ on
# every run, so nothing is ever overwritten. Override the exact path with
# the ENS_OUT environment variable if you need a fixed name.
#
# Usage:
#   bash scripts/evaluate/evaluate_student_5seed.sh                # released seeds
#   bash scripts/evaluate/evaluate_student_5seed.sh CKPT1 ... CKPT5 # custom seeds
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PATHS="$REPO_ROOT/configs/paths.yaml"
if [[ ! -f "$PATHS" ]]; then
    PATHS="$REPO_ROOT/configs/paths_template.yaml"
fi

if [[ "$#" -gt 0 ]]; then
    CKPTS=("$@")
else
    CKPTS=(
        "$REPO_ROOT/checkpoints_CodeEpi_v1.0/seeds/student_seed1.pt"
        "$REPO_ROOT/checkpoints_CodeEpi_v1.0/seeds/student_seed2.pt"
        "$REPO_ROOT/checkpoints_CodeEpi_v1.0/seeds/student_seed3.pt"
        "$REPO_ROOT/checkpoints_CodeEpi_v1.0/seeds/student_seed4.pt"
        "$REPO_ROOT/checkpoints_CodeEpi_v1.0/seeds/student_seed5.pt"
    )
fi

# Fresh timestamped output every run; never clobber a previous file. The
# committed reference lives in results/reproduce/student_summary.json.
ENS_OUT="${ENS_OUT:-$REPO_ROOT/results/reproduce/student_5seed_ensemble_metrics_$(date -u +%Y%m%dT%H%M%SZ).json}"
mkdir -p "$(dirname "$ENS_OUT")"

python -m codeepi.evaluate.student_ensemble \
    --checkpoints "${CKPTS[@]}" \
    --paths  "$PATHS" \
    --output "$ENS_OUT"

echo "[evaluate_student_5seed] ensemble metrics -> $ENS_OUT"
