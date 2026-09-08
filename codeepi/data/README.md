# `codeepi.data`

Thin adapter over the **CodeEpi PyG Release v1.0** dataset (unpacks to
`codeepi_pyg_release_v1.0/`). CodeEpi does **not** re-run ESM-IF / ESM-C
encoding, surface / RSA extraction, or graph construction for
reproduction / training. All of that is already baked into
`codeepi_pyg_release_v1.0/Complex1723/graphs/*.pt` and
`codeepi_pyg_release_v1.0/Antigen1321/graphs/*.pt`.

## Requirements on the user

Prepare a local copy of the release (see `docs/dataset_release.md`) and
point `configs/paths.yaml::dataset_root` at it (copy from
`configs/paths_template.yaml` if you have not yet). Expected layout under
`dataset_root`:

```
codeepi_pyg_release_v1.0/
├── Antigen1321/                    # 1,321 antigen-level graphs (splits: 1057/132/132)
├── Complex1723/                    # 1,723 complex-level graphs (splits: 1425/145/153)
├── dataset/                        # release-shipped PyG dataset package
└── full_README.md
```

The complete on-disk tree (including per-dataset `graphs/`,
`sequences/`, `structures_*_pdb/`, `metadata/` subtrees) is documented
in `codeepi_pyg_release_v1.0/full_README.md`.

## Public API

Only the antigen path is wrapped here:

```python
from codeepi.data.release import AntigenReleaseDataset

train = AntigenReleaseDataset(root="codeepi_pyg_release_v1.0", split="train")
val   = AntigenReleaseDataset(root="codeepi_pyg_release_v1.0", split="val")
test  = AntigenReleaseDataset(root="codeepi_pyg_release_v1.0", split="test")
```

The complex path is consumed directly through the release's own
`Complex1723Dataset` inside `codeepi.teacher.group_dataset`; no adapter
is exposed here.

## Fields returned by `AntigenReleaseDataset.get()`

Each `AntigenReleaseDataset[i]` yields a `torch_geometric.data.Data`
whose fields are all in one of two coordinate systems: **surface**
(length `N` = number of surface residues, the model's input space) and
**resolved** (length `n_resolved` = number of residues with PDB-ATOM
coords, the primary reporting space).

| Field | dtype | shape | coord sys | meaning |
|---|---|---|---|---|
| `x_esmif` | float32 | `[N, 512]` | surface | ESM-IF single-sequence structural embedding |
| `x_esmc` | float32 | `[N, 2560]` | surface | ESM-C per-residue sequence embedding |
| `pos` | float32 | `[N, 3, 3]` | surface | backbone N / Cα / C coordinates |
| `rsa` | float32 | `[N]` | surface | relative solvent-accessible surface area |
| `edge_index` | int64 | `[2, E]` | surface | 8 Å radius graph on surface Cα |
| `edge_attr` | float32 | `[E, 32]` | surface | 16-D RBF distance + 16-D sinusoidal positional encoding |
| `y_node` | float32 | `[N]` | surface | fused epitope label on surface (primary training target) |
| `surf_idx_in_resolved` | int64 | `[N]` | resolved | surface node → PDB-ATOM slot (range `[0, n_resolved)`) |
| `y_resolved` | float32 | `[n_resolved]` | resolved | fused epitope label on PDB-ATOM residues (**primary metric space**) |
| `n_resolved` | int64 | `[1]` | — | number of PDB-ATOM residues in this graph |
| `num_nodes` | int | scalar | — | number of surface nodes (`== N`) |
| `abdbid` | str | — | — | representative complex ID (release field: `rep_abdbid`; renamed by the adapter) |

## Design notes

* The PDB-ATOM eval space is derived from
  `Antigen1321/graphs/{rep}.pt::ag_seqres_resolved` (`bool` mask over
  SEQRES, `True` = residue has backbone ATOM records in the antigen-only
  PDB). Full per-graph schema lives in
  `codeepi_pyg_release_v1.0/Antigen1321/for-graphs_README.md`.
* `codeepi_pyg_release_v1.0/dataset/` is registered under the top-level
  module name `dataset` (that is the module path baked into the pickled
  graph files — the `.pt` classes are `dataset._graph_utils.ComplexData`
  and `dataset._graph_utils.AntigenData`). The same package object is
  also exposed under the stable alias `codeepi_release_dataset` for our
  own imports.
* Because `dataset` is a common top-level name, the loader raises loudly
  if the running process already has an unrelated `dataset` package
  bound, or if a different `release_root` was loaded earlier in the same
  process. Unload the conflicting package before instantiating a release
  dataset.
* Non-existent splits and a missing top-level
  layout are caught eagerly by `detect_release_root`.

## Windows / macOS multiprocessing note

All bundled configs default to `num_workers=0`, which is the intended
setting for reproduction. If you raise it and your Python runs
DataLoader workers under the `spawn` start method (Windows default,
optional on macOS), each worker must be able to `import dataset` on its
own — the parent process's dynamic `sys.modules` registration does not
propagate. Either keep `num_workers=0`, or add
`codeepi_pyg_release_v1.0/` to `PYTHONPATH` before launching Python so
worker processes can resolve the package by name. Linux `fork` workers
inherit `sys.modules` and are unaffected.
