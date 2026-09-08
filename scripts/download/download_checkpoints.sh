#!/usr/bin/env bash
# Download and unpack the single CodeEpi ensemble checkpoint bundle v1.0.
#
# The archive extracts to:
#
#   checkpoints_CodeEpi_v1.0/
#   ├── teacher.pt
#   ├── codebook_4bin.pt
#   └── seeds/
#       ├── student_seed1.pt
#       ├── student_seed2.pt
#       ├── student_seed3.pt
#       ├── student_seed4.pt
#       └── student_seed5.pt
#
# The 5-seed ensemble is the supported inference/evaluation path.

set -euo pipefail


URL="https://zenodo.org/records/22299505/files/checkpoints_CodeEpi_v1.0.tar.gz"

OUT="${CODEEPI_CHECKPOINTS_ARCHIVE:-checkpoints_CodeEpi_v1.0.tar.gz}"

SHA256="6a18a17c176c8cc02b6bc3483566343dc95a76f2e4f735554a9b2923c7c0a2c0"


HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"


echo "[download_checkpoints] fetching $URL -> $OUT"

wget \
    --show-progress \
    -O "$OUT" \
    "$URL"


echo "[download_checkpoints] checking SHA256"

echo "$SHA256  $OUT" | sha256sum -c -


echo "[download_checkpoints] extracting"

tar -xzf "$OUT"


echo "[download_checkpoints] done."

echo "[download_checkpoints] expect:"
echo "    checkpoints_CodeEpi_v1.0/{teacher.pt, codebook_4bin.pt}"
echo "    checkpoints_CodeEpi_v1.0/seeds/student_seed{1..5}.pt"