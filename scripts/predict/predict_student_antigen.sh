#!/usr/bin/env bash
# CodeEpi student antigen-only 5-seed ensemble prediction.
#
# Two usages (both supported):
#
#   Single antigen :
#     bash scripts/predict/predict_student_antigen.sh <pdb> <ag_fasta> <antigen_chain> [output_dir]
#     Defaults to outputs/predict/student_antigen/ if <output_dir> is omitted.
#
#   Batch (many antigens from a CSV manifest):
#     bash scripts/predict/predict_student_antigen.sh --manifest <manifest.csv> [output_dir]
#     Defaults to outputs/predict/student_antigen_batch/ if <output_dir> is
#     omitted. Each row writes the same file layout under <output_dir>/<name>/,
#     plus a top-level batch_summary.json. One failing row does not abort the
#     batch.
#
# Loads every seed under checkpoints_CodeEpi_v1.0/seeds/student_seed{1..5}.pt
# and averages per-residue probabilities.
#
# <ag_fasta> is the antigen FASTA (one record per antigen chain). It is the
# SEQRES authority ESM-C is computed on; the structure surface is mapped
# back to SEQRES (anchor) and written to seqmap.json.
#
# ESM-C is PRECOMPUTED ONLY. Encode the full-length antigen SEQRES first in the separate `codeepi-esmc` environment:
#
#   conda activate codeepi-esmc
#   export ESMC_API_KEY="your_key_here"
#   python -m codeepi.features.encode_esmc \
#       --ag_fasta my_antigen.fasta --antigen_chain A --out my_antigen_esmc.pt
#
# then pass the printed .pt path back to prediction:
#   ESMC_AG_PT    REQUIRED (single-antigen mode). Precomputed ESM-C tensor
#                 (L, 2560) for the full antigen SEQRES. In batch mode use the
#                 manifest's esmc_ag_pt column instead.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Epitope probability threshold (default 0.45)
EPITOPE_THRESHOLD="${EPITOPE_THRESHOLD:-0.45}"

CKPT_ARGS=(
    --checkpoints
        "$REPO_ROOT/checkpoints_CodeEpi_v1.0/seeds/student_seed1.pt"
        "$REPO_ROOT/checkpoints_CodeEpi_v1.0/seeds/student_seed2.pt"
        "$REPO_ROOT/checkpoints_CodeEpi_v1.0/seeds/student_seed3.pt"
        "$REPO_ROOT/checkpoints_CodeEpi_v1.0/seeds/student_seed4.pt"
        "$REPO_ROOT/checkpoints_CodeEpi_v1.0/seeds/student_seed5.pt"
    --codebook "$REPO_ROOT/checkpoints_CodeEpi_v1.0/codebook_4bin.pt"
)

if [[ "${1:-}" == "--manifest" ]]; then
    # ----- batch mode -----
    MANIFEST="${2:?manifest CSV path required after --manifest}"
    OUT="${3:-$REPO_ROOT/outputs/predict/student_antigen_batch}"
    mkdir -p "$OUT"
    python -m codeepi.predict.student_antigen_ensemble \
        --manifest "$MANIFEST" \
        "${CKPT_ARGS[@]}" \
        --output "$OUT" \
        --threshold "$EPITOPE_THRESHOLD"
else
    # ----- single-antigen mode -----
    PDB="${1:?PDB path required (or use --manifest <csv> for batch)}"
    AG_FASTA="${2:?antigen FASTA path required}"
    A="${3:?antigen chain required}"
    OUT="${4:-$REPO_ROOT/outputs/predict/student_antigen}"
    mkdir -p "$OUT"

    if [[ -z "${ESMC_AG_PT:-}" ]]; then
        echo "ERROR: ESMC_AG_PT is required (precomputed ESM-C .pt for the" >&2
        echo "full antigen SEQRES). Encode it first in the codeepi-esmc env:" >&2
        echo "  export ESMC_API_KEY=\"your_key_here\"" >&2
        echo "  python -m codeepi.features.encode_esmc --ag_fasta $AG_FASTA \\" >&2
        echo "      --antigen_chain $A --out my_antigen_esmc.pt" >&2
        echo "then re-run with ESMC_AG_PT=my_antigen_esmc.pt" >&2
        exit 1
    fi

    python -m codeepi.predict.student_antigen_ensemble \
        --pdb "$PDB" \
        --ag_fasta "$AG_FASTA" \
        --antigen_chain "$A" \
        "${CKPT_ARGS[@]}" \
        --output "$OUT" \
        --esmc_ag_pt "$ESMC_AG_PT" \
        --threshold "$EPITOPE_THRESHOLD"
fi
