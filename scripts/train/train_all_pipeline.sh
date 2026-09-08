#!/usr/bin/env bash
# End-to-end CodeEpi training pipeline:
#
#   1. train teacher (auto-exports outputs/<tag>_seed<seed>/teacher_residue_table.pt)
#   2. build codebook   (consumes the residue table from step 1)
#   3. train full student for each of the 5 seeds against that codebook
#
# All artifacts stay under outputs/. The codebook is built to
# CODEBOOK_OUT (default outputs/codebook_4bin.pt) and passed straight to
# the student via the CODEBOOK env var, so no paths.yaml edit is required.
#
# This is a convenience wrapper. If you want fine-grained control over
# teacher seed / output directory, run the three sub-scripts by hand as
# documented in docs/training.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export TEACHER_TAG="${TEACHER_TAG:-teacher}"
export TEACHER_SEED="${TEACHER_SEED:-7876}"
# Build the codebook under outputs/ by default. Step 3
# below is pointed straight at this path, so no paths.yaml edit is needed.
CODEBOOK_OUT="${CODEBOOK_OUT:-$REPO_ROOT/outputs/codebook_4bin.pt}"

echo "[pipeline] 1) train teacher (tag=${TEACHER_TAG} seed=${TEACHER_SEED})"
bash "$REPO_ROOT/scripts/train/train_teacher.sh"

RESIDUE_TABLE="$REPO_ROOT/outputs/${TEACHER_TAG}_seed${TEACHER_SEED}/teacher_residue_table.pt"
if [[ ! -f "$RESIDUE_TABLE" ]]; then
    echo "[pipeline] ERROR: expected teacher_residue_table.pt at" >&2
    echo "           $RESIDUE_TABLE" >&2
    echo "           but the file does not exist. Either the teacher trainer" >&2
    echo "           was run with --no_export_residue_table, or the tag/seed" >&2
    echo "           differ from TEACHER_TAG=${TEACHER_TAG} TEACHER_SEED=${TEACHER_SEED}." >&2
    echo "           Rerun this script after setting the correct env vars, or" >&2
    echo "           invoke steps 2 and 3 manually per docs/training.md." >&2
    exit 2
fi

echo "[pipeline] 2) build 4-bin codebook from $RESIDUE_TABLE"
bash "$REPO_ROOT/scripts/train/build_codebook.sh" "$RESIDUE_TABLE" "$CODEBOOK_OUT"

echo "[pipeline] 3) train full student (5 seeds) against $CODEBOOK_OUT"
for i in 1 2 3 4 5; do
    CODEBOOK="$CODEBOOK_OUT" bash "$REPO_ROOT/scripts/train/train_student.sh" "$i"
done

echo "[pipeline] done."
