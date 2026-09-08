#!/usr/bin/env bash
# Download and unpack the CodeEpi PyG Release v1.0 dataset.
#
# Resulting local layout:
#
#     data/codeepi_pyg_release_v1.0/
#         Complex1723/
#         Antigen1321/
#         dataset/
#         full_README.md
#
# Set ``dataset_root: data/codeepi_pyg_release_v1.0`` in
# ``configs/paths.yaml`` after this script finishes.

set -euo pipefail


URL="https://zenodo.org/records/22299505/files/CodeEpi_pyg_release_v1.0.tar.gz"

OUT="${CODEEPI_DATASET_ARCHIVE:-codeepi_pyg_release_v1.0.tar.gz}"

SHA256="9987b2f774f07d69ee98ef9128ff39e5750d673ab"


HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"


DEST_DIR="${CODEEPI_DATASET_DIR:-data/codeepi_pyg_release_v1.0}"


mkdir -p "$DEST_DIR"


echo "[download_dataset] fetching $URL -> $OUT"

wget \
    --show-progress \
    -O "$OUT" \
    "$URL"


echo "[download_dataset] checking SHA256"

echo "$SHA256  $OUT" | sha256sum -c -


echo "[download_dataset] extracting into $DEST_DIR/"

tar -xzf "$OUT" -C "$DEST_DIR"


echo "[download_dataset] done."
echo "[download_dataset] dataset is at $DEST_DIR/"
echo "[download_dataset] expected:"
echo "    $DEST_DIR/{Complex1723,Antigen1321,dataset,full_README.md}"
echo "[download_dataset] set 'dataset_root: $DEST_DIR' in configs/paths.yaml"