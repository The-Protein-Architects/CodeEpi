# Data

This directory is intentionally empty in the GitHub repository,CodeEpi uses the external
CodeEpi PyG Release v1.0 dataset.

Download and unpack it with:

```bash
bash scripts/download/download_dataset.sh
```

The default layout is:

```
data/codeepi_pyg_release_v1.0/
├── Complex1723/
├── Antigen1321/
├── dataset/
└── full_README.md
```

The default path is already configured in `configs/paths_template.yaml`:

```yaml
dataset_root: data/codeepi_pyg_release_v1.0
```

To store the dataset elsewhere, set `dataset_root` to its absolute path in
`configs/paths.yaml`.

See [`docs/dataset_release.md`](../docs/dataset_release.md) for the complete
dataset description.

