# Dataset release

CodeEpi training, reproduction, and evaluation consume the external
**CodeEpi PyG Release v1.0** dataset. The release is *not* stored in this
GitHub repository; it is distributed as compressed archives from
Zenodo (see `scripts/download/download_dataset.sh`).

## Contents

The release provides two preprocessed PyTorch Geometric datasets:

| dataset | samples | split (train / val / test) | purpose |
|---|---:|---|---|
| Complex1723 | 1,723 antibody–antigen complex graphs | 1425 / 145 / 153 | teacher training and teacher-complex prediction examples |
| Antigen1321 | 1,321 representative antigen-only graphs | 1057 / 132 / 132 | antigen-only student training and reproduction |

Each graph is self-contained. Standard CodeEpi training and reproduction
do **not** need rebuild ESM-IF, ESM-C, coordinates, RSA, or graph edges.

## Layout

Archive: `codeepi_pyg_release_v1.0.tar.gz`

`download_dataset.sh` unpacks them into `data/codeepi_pyg_release_v1.0/`:

```
data/codeepi_pyg_release_v1.0/
├── Antigen1321/
│   ├── graphs/{rep_abdbid}.pt
│   ├── structures_antigen_single_pdb/{rep_abdbid}.pdb
│   ├── metadata/
│   │   ├── antigen_manifest.csv
│   │   ├── rep_to_members.csv
│   │   ├── rep_epitope_fused.csv
│   │   ├── positions_v4.csv
│   │   ├── positions_esmif_v4.csv
│   │   ├── clip_positions_v4.json
│   ├── sequences/
│   │   ├── representative_antigen_sequences.fasta
│   ├── for-Antigen1321_metadata_README.md
│   └── for-graphs_README.md
│
├── Complex1723/
│   ├── graphs/{abdbid}.pt
│   ├── structures_complex_pdb/{abdbid}.pdb
│   ├── metadata/
│   │   ├── complex_manifest.csv
│   │   ├── complex_to_rep.csv
│   │   ├── split_dict.pt
│   │   ├── positions_original.csv
│   ├── sequences/
│   │   ├── antibody_sequences.fasta
│   │   ├── antigen_sequences.fasta
│   ├── for-Complex1723_metadata_README.md
│   └──  for-graphs_README.md
│
├── dataset/
│   ├── __init__.py
│   ├── _graph_utils.py
│   ├── antigen1321.py
│   ├── complex1723.py
│   └── for-dataset_README.md
└── full_README.md
```

Total on-disk size after extraction is approximately 13 GB.

## Recommended local layout

We recommend unpacking the archive under this repository
into `data/codeepi_pyg_release_v1.0/`:

```bash
mkdir -p data/codeepi_pyg_release_v1.0
tar -xzf codeepi_pyg_release_v1.0.tar.gz -C data/codeepi_pyg_release_v1.0
```

`scripts/download/download_dataset.sh` performs exactly this. Then set
(or leave as-is, since it is the shipped default):

```yaml
dataset_root: data/codeepi_pyg_release_v1.0
```

The dataset can also be stored outside this repository. In that case,
set `dataset_root` to the absolute path of the unpacked directory, for
example:

```yaml
dataset_root: /scratch/user/datasets/codeepi_pyg_release_v1.0
```

All training, reproduction, and prediction scripts read `dataset_root`
from a paths YAML (`--paths configs/paths.yaml`).

## Using the datasets directly

```python
import sys
from pathlib import Path
from torch_geometric.loader import DataLoader

DR = "data/codeepi_pyg_release_v1.0"
sys.path.insert(0, str(Path(DR).resolve()))
from dataset import Complex1723Dataset, Antigen1321Dataset

train_c = Complex1723Dataset(root=f"{DR}/Complex1723", split="train")
train_a = Antigen1321Dataset(root=f"{DR}/Antigen1321", split="train")

for batch in DataLoader(train_c, batch_size=8, shuffle=True):
    pass

for batch in DataLoader(train_a, batch_size=8, shuffle=True):
    pass
```

Inside CodeEpi, `codeepi.data.release` wraps these two dataset classes
and exposes `AntigenReleaseDataset` / `ComplexReleaseDataset` with the
field names the CodeEpi trainers expect.

## Split inheritance

For `Complex1723`, non-representative complex members inherit the split
of their representative antigen. Complexes from the same antigen group
therefore always remain in the same split, and `Complex1723` and
`Antigen1321` share the same split assignment.




