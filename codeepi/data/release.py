"""Thin adapters over the **CodeEpi PyG Release v1.0** dataset (extracted
directory: ``codeepi_pyg_release_v1.0/``).

Users point ``configs/paths_template.yaml::dataset_root`` at their local copy
of ``codeepi_pyg_release_v1.0/`` and both trainers read from that path.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from torch_geometric.data import Data, Dataset


# ---------------------------------------------------------------------------
# release import helpers
# ---------------------------------------------------------------------------

def detect_release_root(dataset_root: str | Path) -> Path:
    """Validate that ``dataset_root`` points at a CodeEpi PyG Release v1.0 layout.

    Expects the standard layout::

        <dataset_root>/
            Antigen1321/
                graphs/
                metadata/antigen_manifest.csv
            Complex1723/
                graphs/
                metadata/complex_manifest.csv
            dataset/
                __init__.py
                complex1723.py
                antigen1321.py
    """
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"dataset_root not found: {root}")
    required = [
        root / "Antigen1321" / "graphs",
        root / "Antigen1321" / "metadata" / "antigen_manifest.csv",
        root / "Complex1723" / "graphs",
        root / "Complex1723" / "metadata" / "complex_manifest.csv",
        root / "dataset" / "__init__.py",
        root / "dataset" / "complex1723.py",
        root / "dataset" / "antigen1321.py",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "release layout mismatch; missing:\n  "
            + "\n  ".join(str(p) for p in missing)
            + f"\nchecked under: {root}"
        )
    return root


def _module_origin(module) -> Optional[Path]:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return None
    return Path(module_file).resolve()


def _import_release_module(release_root: Path):
    """Import ``<release_root>/dataset/`` as the top-level ``dataset`` package.

    The release's per-graph ``.pt`` files were pickled with class references
    such as ``dataset._graph_utils.ComplexData`` and
    ``dataset._graph_utils.AntigenData`` (the submodules under
    ``dataset/`` only re-export those classes; ``__module__`` is
    ``dataset._graph_utils``). So at unpickle time the top-level module name
    ``dataset`` and its ``_graph_utils`` submodule must resolve to this
    release. We register the package under the pickle name ``dataset`` and
    also expose the same object under the stable alias
    ``codeepi_release_dataset`` for our own imports.

    Because the pickle format forces us to occupy the top-level name
    ``dataset``, this call fails loudly if:
      * ``sys.modules["dataset"]`` is already bound to a package loaded
        from a different location (user's own ``dataset`` package or a
        previously-loaded release_root), or
      * the alias ``codeepi_release_dataset`` was cached from a different
        release_root in this same process.
    """
    alias = "codeepi_release_dataset"
    pkg_name = "dataset"

    pkg_dir = release_root / "dataset"
    init_path = (pkg_dir / "__init__.py").resolve()

    existing = sys.modules.get(pkg_name)
    if existing is not None:
        existing_origin = _module_origin(existing)
        if existing_origin != init_path:
            raise ImportError(
                f"top-level module name 'dataset' is already bound to "
                f"{existing_origin}, but CodeEpi requires it to point at "
                f"{init_path} (the release's dataset package, whose class "
                f"paths are baked into the pickled graph files). Unload the "
                f"conflicting package before instantiating a release dataset."
            )

    cached = sys.modules.get(alias)
    if cached is not None:
        cached_origin = _module_origin(cached)
        if cached_origin != init_path:
            raise ImportError(
                f"codeepi_release_dataset was already loaded from "
                f"{cached_origin}, but the requested release is {init_path}. "
                f"Two release roots cannot coexist in one process."
            )
        return cached

    spec = importlib.util.spec_from_file_location(
        pkg_name,
        init_path,
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load release dataset package from {pkg_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(pkg_name, None)
        raise
    sys.modules[alias] = module
    return module


# ---------------------------------------------------------------------------
# antigen-only dataset (used by the student trainer)
# ---------------------------------------------------------------------------

class AntigenReleaseDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform=None,
        pre_transform=None,
    ):
        release_root = detect_release_root(root)
        release = _import_release_module(release_root)
        self._release_ds = release.Antigen1321Dataset(
            root=str(release_root / "Antigen1321"),
            split=split,
        )
        self._split = split
        self._release_root = release_root

        super().__init__(root=str(release_root), transform=transform,
                         pre_transform=pre_transform)

    # -- required PyG Dataset API --
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return list(self._release_ds.processed_file_names)

    def download(self):
        return

    def process(self):
        return

    def len(self):
        return len(self._release_ds)

    def get(self, idx: int) -> Data:
        src = self._release_ds.get(idx)
        # The release stores backbone coords as (N, 3, 3) under
        # ``pos_ag_backbone``. Some release builds may also carry an alias
        # ``pos``; either way we normalize to the trainer-facing ``pos``.
        pos = getattr(src, "pos_ag_backbone", None)
        if pos is None:
            pos = getattr(src, "pos", None)
        if pos is None:
            raise KeyError(
                f"AntigenReleaseDataset: no backbone coords in graph for "
                f"{getattr(src, 'rep_abdbid', 'unknown')}"
            )

        x_esmif = getattr(src, "x_ag_esmif_single", None)
        x_esmc = getattr(src, "x_ag_esmc_raw", None)
        if x_esmif is None or x_esmc is None:
            raise KeyError(
                "AntigenReleaseDataset: missing x_ag_esmif_single / x_ag_esmc_raw "
                f"in graph for {getattr(src, 'rep_abdbid', 'unknown')}"
            )

        rsa = getattr(src, "rsa")
        edge_index = getattr(src, "edge_index_egnn")
        edge_attr = getattr(src, "edge_attr_egnn")
        y_surf = getattr(src, "y_epitope_fused_surf")
        y_node = y_surf.float() if torch.is_tensor(y_surf) else torch.tensor(y_surf).float()
        abdbid = str(getattr(src, "rep_abdbid"))

        # Build the PDB-ATOM (resolved) coordinate mapping. 
        surf_idx_seqres_src = getattr(src, "ag_node_to_seqres_index", None)
        y_full_src = getattr(src, "y_epitope_fused_seqres", None)
        resolved_mask = getattr(src, "ag_seqres_resolved", None)
        if surf_idx_seqres_src is None or y_full_src is None or resolved_mask is None:
            raise KeyError(
                f"AntigenReleaseDataset[{abdbid}]: release graph missing one of "
                "{ag_node_to_seqres_index, y_epitope_fused_seqres, ag_seqres_resolved}; "
                "rebuild the release with the v1.0 build."
            )
        surf_idx_seqres = surf_idx_seqres_src.long()
        y_full = (y_full_src.float() if torch.is_tensor(y_full_src)
                  else torch.tensor(y_full_src).float())
        n_full = int(y_full.shape[0])
        if int(surf_idx_seqres.numel()) != int(x_esmif.shape[0]):
            raise RuntimeError(
                f"AntigenReleaseDataset[{abdbid}]: surface node count "
                f"{int(surf_idx_seqres.numel())} != x_esmif rows "
                f"{int(x_esmif.shape[0])}"
            )
        if int(surf_idx_seqres.max().item()) >= n_full or int(surf_idx_seqres.min().item()) < 0:
            raise RuntimeError(
                f"AntigenReleaseDataset[{abdbid}]: surface SEQRES index out of "
                f"range [0, {n_full}); got "
                f"[{int(surf_idx_seqres.min())}, {int(surf_idx_seqres.max())}]"
            )

        # ag_seqres_resolved: bool mask over SEQRES marking positions with
        # ATOM coords in the antigen-only PDB. True positions are the
        # PDB-ATOM eval slots (the primary reporting coordinate).
        resolved_idx = torch.where(resolved_mask)[0].long()
        assert resolved_idx.numel() > 0, (
            f"AntigenReleaseDataset[{abdbid}]: ag_seqres_resolved has no True positions"
        )
        assert int(resolved_idx.max()) < n_full, (
            f"AntigenReleaseDataset[{abdbid}]: resolved_idx max {int(resolved_idx.max())} "
            f">= n_full {n_full}"
        )
        assert torch.isin(surf_idx_seqres, resolved_idx).all(), (
            f"AntigenReleaseDataset[{abdbid}]: surface nodes not ⊆ resolved_idx "
            f"(release inconsistency)"
        )
        n_resolved = int(resolved_idx.numel())
        y_resolved = y_full[resolved_idx].float()

        # Map each surface SEQRES position to its slot in the resolved coord.
        seqres_to_resolved = torch.full((n_full,), -1, dtype=torch.long)
        seqres_to_resolved[resolved_idx] = torch.arange(n_resolved, dtype=torch.long)
        surf_idx_in_resolved = seqres_to_resolved[surf_idx_seqres]

        # ``surf_idx_in_resolved`` is length-``num_nodes`` and its entries are
        # slots in the PDB-ATOM (resolved) coordinate system. 
        return Data(
            x_esmif=x_esmif.float(),
            x_esmc=x_esmc.float(),
            pos=pos.float(),
            rsa=rsa.float(),
            edge_index=edge_index.long(),
            edge_attr=edge_attr.float(),
            y_node=y_node,
            surf_idx_in_resolved=surf_idx_in_resolved,
            y_resolved=y_resolved,
            n_resolved=torch.tensor([n_resolved], dtype=torch.long),
            abdbid=abdbid,
            num_nodes=int(x_esmif.shape[0]),
        )

    # -- convenience --
    @property
    def manifest(self):
        return self._release_ds.manifest

    @property
    def abdbids(self):
        return tuple(self._release_ds.rep_abdbids)

    def __repr__(self) -> str:
        return (f"AntigenReleaseDataset(split={self._split!r}, "
                f"num_graphs={len(self)})")
