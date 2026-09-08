#!/usr/bin/env bash
# Train the CodeEpi teacher on Complex1723.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PATHS="$REPO_ROOT/configs/paths.yaml"
if [[ ! -f "$PATHS" ]]; then
    PATHS="$REPO_ROOT/configs/paths_template.yaml"
fi

# Both knobs are optional.
TEACHER_TAG="${TEACHER_TAG:-teacher}"
EXTRA_ARGS=()
if [[ -n "${TEACHER_SEED:-}" ]]; then
    EXTRA_ARGS+=(--seed "$TEACHER_SEED")
fi

python -m codeepi.teacher.train_teacher \
    --config "$REPO_ROOT/configs/train/teacher.yaml" \
    --paths  "$PATHS" \
    --tag "$TEACHER_TAG" \
    "${EXTRA_ARGS[@]}"
