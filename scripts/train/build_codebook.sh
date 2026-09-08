#!/usr/bin/env bash
# Build the 4-bin teacher-derived codebook.
#
# Usage
# -----
#     bash scripts/train/build_codebook.sh <teacher_residue_table.pt> [out.pt]
#
# The residue table is the intermediate artifact automatically written by
# codeepi.teacher.train_teacher at the end of a training run
# (outputs/<tag>_seed<seed>/teacher_residue_table.pt); it contains
# per-training-residue teacher_prob and 256-d post-QKV embeddings.
#
# The env-var form is also supported for backward compatibility:
#     RESIDUE_TABLE=<...>.pt OUT_PATH=<...>.pt bash scripts/train/build_codebook.sh
#
# Bin edges default to CodeEpi's canonical [0.06, 0.50, 0.87]; override
# via env vars E1/E2/E3.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RESIDUE_TABLE="${1:-${RESIDUE_TABLE:-}}"
OUT_PATH="${2:-${OUT_PATH:-checkpoints_CodeEpi_v1.0/codebook_4bin.pt}}"
E1="${E1:-0.06}"
E2="${E2:-0.50}"
E3="${E3:-0.87}"

if [[ -z "$RESIDUE_TABLE" ]]; then
    echo "Usage: $0 <teacher_residue_table.pt> [out.pt]" >&2
    echo "Alternatively: RESIDUE_TABLE=<...>.pt OUT_PATH=<...>.pt $0" >&2
    echo "Typical: bash scripts/train/build_codebook.sh \\" >&2
    echo "             outputs/teacher_seed<seed>/teacher_residue_table.pt \\" >&2
    echo "             checkpoints_CodeEpi_v1.0/codebook_4bin.pt" >&2
    exit 2
fi

python -m codeepi.codebook.build_codebook \
    --residue-table "$RESIDUE_TABLE" \
    --out "$OUT_PATH" \
    --e1 "$E1" \
    --e2 "$E2" \
    --e3 "$E3"
