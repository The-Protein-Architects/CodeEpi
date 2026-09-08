#!/usr/bin/env bash
# Reproduce all five seeds of the CodeEpi full student sequentially.
#
# Usage:
#   bash scripts/reproduce/reproduce_student_5seed.sh
#
# Every seed writes to outputs/reproduce/seed_<i>/; the official reference
# summary tracked in results/reproduce/ is left untouched. All 5 seeds
# together define the release ensemble.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PATHS="$REPO_ROOT/configs/paths.yaml"
if [[ ! -f "$PATHS" ]]; then
    PATHS="$REPO_ROOT/configs/paths_template.yaml"
fi

OUT_ROOT="$REPO_ROOT/outputs/reproduce"
mkdir -p "$OUT_ROOT"

for i in 1 2 3 4 5; do
    echo
    echo "==============================================================="
    echo "[reproduce_student_5seed] seed_index=$i"
    echo "==============================================================="
    python -m codeepi.student.train_student \
        --config "$REPO_ROOT/configs/train/student.yaml" \
        --paths  "$PATHS" \
        --seed_index "$i" \
        --output_dir "$OUT_ROOT" \
        --tag student
done

echo
echo "[reproduce_student_5seed] all five seeds finished."
echo "[reproduce_student_5seed] per-seed outputs:"
for i in 1 2 3 4 5; do
    echo "    outputs/reproduce/seed_${i}/student_seed${i}.json"
done
echo "[reproduce_student_5seed] official reference summary lives in:"
echo "    results/reproduce/student_summary.json"
echo "[reproduce_student_5seed] next step: ensemble the 5 retrained ckpts in place:"
echo "    bash scripts/evaluate/evaluate_student_5seed.sh \\"
for i in 1 2 3 4 5; do
    sep=" \\"; [ "$i" -eq 5 ] && sep=""
    echo "        outputs/reproduce/seed_${i}/student_seed${i}.pt${sep}"
done
