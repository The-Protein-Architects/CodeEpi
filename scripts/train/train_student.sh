#!/usr/bin/env bash
# Train a fresh full student model (single seed).
#
# Usage:
#   bash scripts/train/train_student.sh <seed_index>   # 1..5
#
# Optional env var:
#   CODEBOOK  path to the codebook checkpoint. If set, it is passed as
#             --codebook and overrides codebook_checkpoint in paths.yaml
#             (used by train_all_pipeline.sh to consume a freshly built
#             codebook without editing paths.yaml). Unset -> paths.yaml.
set -euo pipefail

SEED_INDEX="${1:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PATHS="$REPO_ROOT/configs/paths.yaml"
if [[ ! -f "$PATHS" ]]; then
    PATHS="$REPO_ROOT/configs/paths_template.yaml"
fi

extra_args=()
if [[ -n "${CODEBOOK:-}" ]]; then
    extra_args+=(--codebook "$CODEBOOK")
fi

python -m codeepi.student.train_student \
    --config "$REPO_ROOT/configs/train/student.yaml" \
    --paths  "$PATHS" \
    --seed_index "$SEED_INDEX" \
    --tag student \
    "${extra_args[@]}"
