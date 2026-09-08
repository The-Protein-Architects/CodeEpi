"""Group-aware teacher dataset over ``codeepi_pyg_release_v1.0``.

Each training sample is a *rep antigen group*: one representative complex
plus all sibling complexes that share its antigen SEQRES. Within a group,
antibody CDR residues from every member are aggregated onto the rep-fused
antigen surface coordinate system so the teacher's cross-attention runs in
a single, canonical coordinate frame (see paper section 2.3).

The group_index is built once at construction time from the release
metadata alone (no external cache): ``Antigen1321`` supplies the fused
surface and epitope in seqres coordinates, ``Complex1723`` supplies the
per-member ESM-IF/ESM-C features, internal graphs, and 4.5 A CDR-surface
contacts. Member surface residues are mapped back into rep-fused-surface
index space through the shared antigen seqres coordinate.

``collate_groups`` fans out K members across a mini-batch of B groups,
adds per-member node offsets for the two GCNs, and produces the flat
tensors the teacher (``codeepi.teacher.model.CodeEpiTeacher``)
consumes.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from ..data.release import _import_release_module, detect_release_root


# ---------------------------------------------------------------------------
# group_index construction (release-only, no external data_v2 cache)
# ---------------------------------------------------------------------------

def _load_positions_original(complex_meta_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load per-complex antigen surface seqres positions from the release."""
    out: Dict[str, Dict[str, Any]] = {}
    path = complex_meta_dir / "positions_original.csv"
    with open(path) as fh:
        for row in csv.DictReader(fh):
            ab = row["abdbid"]
            surf = row.get("ag_surf_seqres_positions", "")
            out[ab] = {
                "ag_seqres": row["ag_seqres"],
                "ag_surf_pos": [int(x) for x in surf.split(",")] if surf else [],
            }
    return out


def _load_complex_manifest(complex_meta_dir: Path
                           ) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (abdbid -> split, abdbid -> rep_abdbid)."""
    split_of: Dict[str, str] = {}
    rep_of: Dict[str, str] = {}
    with open(complex_meta_dir / "complex_manifest.csv") as fh:
        for row in csv.DictReader(fh):
            split_of[row["abdbid"]] = row["split"]
            rep_of[row["abdbid"]] = row["rep_abdbid"]
    return split_of, rep_of


def build_group_index(dataset_root: str | Path) -> Dict[str, Any]:
    """Construct the rep-fused-surface group index from release metadata.

    Uses only files inside ``codeepi_pyg_release_v1.0``:
      * ``Complex1723/metadata/complex_manifest.csv`` -- splits + rep-of map
      * ``Complex1723/metadata/positions_original.csv`` -- per-member surf
      * ``Antigen1321/graphs/{rep}.pt`` -- fused surface + fused epitope
      * ``Complex1723/graphs/{member}.pt`` -- ``edge_index_contact``

    Returns a dict with the same shape the training loop expects::

        {
          'reps'       : list[str] rep_abdbids in release rep order
          'split_of'   : dict rep_abdbid -> {"train","val","test"}
          'meta'       : dict rep_abdbid -> per-rep meta (see module docstring)
        }
    """
    root = detect_release_root(dataset_root)
    release = _import_release_module(root)

    complex_meta = root / "Complex1723" / "metadata"
    pos_orig = _load_positions_original(complex_meta)
    _, rep_of_complex = _load_complex_manifest(complex_meta)

    # rep_abdbid -> list of member abdbids (rep first)
    members_of: Dict[str, List[str]] = {}
    for ab, rep in rep_of_complex.items():
        members_of.setdefault(rep, []).append(ab)
    for rep, mem in members_of.items():
        # keep rep as first member so is_rep flags map cleanly.
        mem.sort(key=lambda a: (a != rep, a))

    # Iterate the Antigen1321 splits to get rep list + split assignment +
    # fused surface coord (num_nodes) + fused epitope. The release-shipped
    # Antigen1321 graph already encodes fused surface & epi.
    reps_all: List[str] = []
    split_of_rep: Dict[str, str] = {}
    meta: Dict[str, Any] = {}

    ag_split_ds = {
        s: release.Antigen1321Dataset(
            root=str(root / "Antigen1321"), split=s,
        )
        for s in ("train", "val", "test")
    }
    complex_split_ds = {
        s: release.Complex1723Dataset(
            root=str(root / "Complex1723"), split=s,
        )
        for s in ("train", "val", "test")
    }
    # index member -> (split, position in that split) so we can pull the
    # release ComplexData for edge_index_contact without a filesystem walk
    complex_index: Dict[str, Tuple[str, int]] = {}
    for s, ds in complex_split_ds.items():
        for i, ab in enumerate(ds.abdbids):
            complex_index[ab] = (s, i)

    for s, ag_ds in ag_split_ds.items():
        for i in range(len(ag_ds)):
            data = ag_ds.get(i)
            rep = str(data.rep_abdbid)
            reps_all.append(rep)
            split_of_rep[rep] = s

            node2seq = data.ag_node_to_seqres_index
            if torch.is_tensor(node2seq):
                rep_surf_seqres = [int(x) for x in node2seq.tolist()]
            else:
                rep_surf_seqres = [int(x) for x in list(node2seq)]
            rep_pos2idx = {p: k for k, p in enumerate(rep_surf_seqres)}

            epi_mask = data.y_epitope_fused_surf
            if torch.is_tensor(epi_mask):
                epi_in_surf = torch.where(epi_mask.bool())[0].tolist()
            else:
                epi_in_surf = [k for k, v in enumerate(list(epi_mask)) if bool(v)]
            epi_in_surf = [int(x) for x in epi_in_surf]

            # Internal SEQRES-space intermediates used to build the
            # PDB-ATOM (resolved) side-channel. 
            y_full_src = getattr(data, "y_epitope_fused_seqres", None)
            resolved_mask_src = getattr(data, "ag_seqres_resolved", None)
            if y_full_src is None or resolved_mask_src is None:
                raise KeyError(
                    f"Antigen1321 graph for {rep} missing y_epitope_fused_seqres "
                    "or ag_seqres_resolved; regenerate the release with the v1.0 build."
                )
            y_full = (y_full_src.float() if torch.is_tensor(y_full_src)
                      else torch.tensor(y_full_src).float())
            n_full = int(y_full.shape[0])

            # PDB-ATOM (resolved) mask: ag_seqres_resolved is a bool tensor
            # over SEQRES marking positions that have ATOM-record coordinates.
            # Surface nodes are guaranteed ⊆ resolved ⊆ SEQRES, so scattering
            # surface probs into the resolved slot never drops an epitope.
            rm = (resolved_mask_src.bool() if torch.is_tensor(resolved_mask_src)
                  else torch.tensor(resolved_mask_src).bool())
            resolved_idx = [int(x) for x in torch.where(rm)[0].tolist()]
            n_resolved = len(resolved_idx)
            # y_resolved: label vector aligned to the PDB-ATOM coordinate
            seqres_to_resolved = {p: k for k, p in enumerate(resolved_idx)}
            y_resolved = torch.zeros(n_resolved, dtype=torch.float32)
            for k, p in enumerate(resolved_idx):
                y_resolved[k] = y_full[p]

            # Precompute surface_node -> resolved_slot mapping so the
            # training loop never needs the SEQRES intermediate again.
            missing = [p for p in rep_surf_seqres if p not in seqres_to_resolved]
            if missing:
                raise RuntimeError(
                    f"Antigen1321 graph for {rep}: {len(missing)} surface "
                    f"SEQRES positions are not in the resolved set (release "
                    f"inconsistency, e.g. pos {missing[:3]})."
                )
            surf_idx_in_resolved = torch.tensor(
                [seqres_to_resolved[p] for p in rep_surf_seqres],
                dtype=torch.long,
            )

            group_members: List[str] = members_of.get(rep, [rep])
            ag_align: Dict[str, List[Tuple[int, int]]] = {}
            ab_buckets: Dict[str, List[Tuple[int, int]]] = {}

            for m in group_members:
                m_surf = pos_orig.get(m, {}).get("ag_surf_pos", [])
                ag_pairs: List[Tuple[int, int]] = []
                m_surf_to_rep: Dict[int, int] = {}
                for j, p in enumerate(m_surf):
                    if p in rep_pos2idx:
                        ri = rep_pos2idx[p]
                        ag_pairs.append((ri, j))
                        m_surf_to_rep[j] = ri
                ag_align[m] = ag_pairs

                # pull member's edge_index_contact from Complex1723
                if m not in complex_index:
                    ab_buckets[m] = []
                    continue
                m_split, m_pos = complex_index[m]
                cd = complex_split_ds[m_split].get(m_pos)
                ec = getattr(cd, "edge_index_contact", None)
                if ec is None or not torch.is_tensor(ec) or ec.numel() == 0:
                    ab_buckets[m] = []
                    continue
                ab_pairs: List[Tuple[int, int]] = []
                # row0 = CDR node index (member cdr space)
                # row1 = surface node index (member surf space)
                for k in range(ec.shape[1]):
                    cdr_j = int(ec[0, k].item())
                    surf_k = int(ec[1, k].item())
                    if surf_k in m_surf_to_rep:
                        ab_pairs.append((m_surf_to_rep[surf_k], cdr_j))
                ab_buckets[m] = ab_pairs

            meta[rep] = {
                "members": [m for m in group_members if m in ag_align],
                "rep_surf_seqres": rep_surf_seqres,
                "ag_align": ag_align,
                "ab_buckets": ab_buckets,
                "epi_in_surf": epi_in_surf,
                # PDB-ATOM (resolved) side-channel — the only reporting space.
                "n_resolved": n_resolved,
                "y_resolved": y_resolved,          # (n_resolved,) PDB-ATOM-space
                "surf_idx_in_resolved": surf_idx_in_resolved,  # (n_surf,) long
            }

    return {
        "reps": reps_all,
        "split_of": split_of_rep,
        "meta": meta,
        "_meta_": {
            "source": "codeepi_pyg_release_v1.0",
            "n_reps": len(reps_all),
            "n_total_members": sum(len(v["members"]) for v in meta.values()),
        },
    }


# ---------------------------------------------------------------------------
# per-group dataset
# ---------------------------------------------------------------------------

class GroupTeacherDataset(Dataset):
    """Rep-antigen-group teacher dataset.

    ``rep_list`` is a subset of ``group_index['reps']`` (typically the reps
    whose ``split == split_name``). ``member_graphs`` is a mapping
    ``member_abdbid -> torch_geometric.data.Data`` covering every abdbid
    referenced by ``rep_list``. In practice the caller passes a lazy
    accessor (``_MemberGraphLoader``) that fetches per-member release
    graphs on demand.
    """

    def __init__(
        self,
        rep_list: Sequence[str],
        group_index: Dict[str, Any],
        member_graphs,  # callable: abdbid -> Data
    ):
        super().__init__()
        self.reps: List[str] = list(rep_list)
        self.gi = group_index
        self._member_graphs = member_graphs

    def __len__(self) -> int:
        return len(self.reps)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rep = self.reps[idx]
        meta = self.gi["meta"][rep]
        rep_surf_len = len(meta["rep_surf_seqres"])

        member_data: List[Dict[str, Any]] = []
        for m in meta["members"]:
            cd = self._member_graphs(m)
            x_g = cd.x_ag_esmif_raw.float()
            x_b = cd.x_ab_esmif_raw.float()
            x_g_esmc = cd.x_ag_esmc_raw.float()
            x_b_esmc = cd.x_ab_esmc_raw.float()
            e_ag = cd.edge_index_ag.long()
            e_ab = cd.edge_index_ab.long()

            ag_pairs = meta["ag_align"].get(m, [])
            ab_pairs = meta["ab_buckets"].get(m, [])
            ag_pair_t = (torch.tensor(ag_pairs, dtype=torch.long).reshape(-1, 2)
                         if ag_pairs else torch.zeros((0, 2), dtype=torch.long))
            ab_pair_t = (torch.tensor(ab_pairs, dtype=torch.long).reshape(-1, 2)
                         if ab_pairs else torch.zeros((0, 2), dtype=torch.long))

            member_data.append({
                "abdbid": m,
                "x_g": x_g, "x_b": x_b,
                "x_g_esmc": x_g_esmc, "x_b_esmc": x_b_esmc,
                "edge_index_g": e_ag, "edge_index_b": e_ab,
                "ag_pairs": ag_pair_t,
                "ab_pairs": ab_pair_t,
                "is_rep": (m == rep),
            })

        y = torch.zeros(rep_surf_len, dtype=torch.float32)
        for k in meta["epi_in_surf"]:
            if 0 <= int(k) < rep_surf_len:
                y[int(k)] = 1.0

        return {
            "rep_abdbid": rep,
            "members": [d["abdbid"] for d in member_data],
            "member_data": member_data,
            "rep_surf_len": rep_surf_len,
            "y_g": y,
            "n_resolved": int(meta["n_resolved"]),
            "y_resolved": meta["y_resolved"].clone().float(),
            "surf_idx_in_resolved": meta["surf_idx_in_resolved"].clone().long(),
        }


# ---------------------------------------------------------------------------
# collate: flatten K members across B groups + build all index tensors
# ---------------------------------------------------------------------------

def collate_groups(batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    B = len(batch)
    x_g_list, x_b_list = [], []
    x_g_esmc_list, x_b_esmc_list = [], []
    e_ag_list, e_ab_list = [], []
    ag_pairs_list, ab_pairs_list = [], []
    ab_pairs_self_list, ab_pairs_other_list = [], []
    x_b_batch_list = []
    ag_node_offset = 0
    ab_node_offset = 0
    rep_surf_lens: List[int] = []
    y_list: List[torch.Tensor] = []

    for gi, item in enumerate(batch):
        rep_surf_lens.append(int(item["rep_surf_len"]))
        y_list.append(item["y_g"])
        for md in item["member_data"]:
            x_g_list.append(md["x_g"])
            x_b_list.append(md["x_b"])
            x_g_esmc_list.append(md["x_g_esmc"])
            x_b_esmc_list.append(md["x_b_esmc"])
            e_ag_list.append(md["edge_index_g"] + ag_node_offset)
            e_ab_list.append(md["edge_index_b"] + ab_node_offset)
            n_g = int(md["x_g"].shape[0])
            n_b = int(md["x_b"].shape[0])
            x_b_batch_list.append(torch.full((n_b,), gi, dtype=torch.long))

            if md["ag_pairs"].numel() > 0:
                gp = md["ag_pairs"]
                global_idx_g = gp[:, 1] + ag_node_offset
                ag_pairs_list.append(torch.stack([
                    torch.full_like(gp[:, 0], gi), gp[:, 0], global_idx_g,
                ], dim=1))
            if md["ab_pairs"].numel() > 0:
                bp = md["ab_pairs"]
                global_idx_b = bp[:, 1] + ab_node_offset
                stacked = torch.stack([
                    torch.full_like(bp[:, 0], gi), bp[:, 0], global_idx_b,
                ], dim=1)
                ab_pairs_list.append(stacked)
                if md["is_rep"]:
                    ab_pairs_self_list.append(stacked)
                else:
                    ab_pairs_other_list.append(stacked)

            ag_node_offset += n_g
            ab_node_offset += n_b

    zeros3 = torch.zeros((0, 3), dtype=torch.long)
    ag_pairs_all = torch.cat(ag_pairs_list, dim=0) if ag_pairs_list else zeros3
    ab_pairs_all = torch.cat(ab_pairs_list, dim=0) if ab_pairs_list else zeros3
    ab_pairs_self_all = (torch.cat(ab_pairs_self_list, dim=0)
                         if ab_pairs_self_list else zeros3)
    ab_pairs_other_all = (torch.cat(ab_pairs_other_list, dim=0)
                          if ab_pairs_other_list else zeros3)

    x_g_all = torch.cat(x_g_list, dim=0) if x_g_list else torch.zeros(0, 512)
    x_b_all = torch.cat(x_b_list, dim=0) if x_b_list else torch.zeros(0, 512)
    x_g_esmc_all = (torch.cat(x_g_esmc_list, dim=0)
                    if x_g_esmc_list else torch.zeros(0, 2560))
    x_b_esmc_all = (torch.cat(x_b_esmc_list, dim=0)
                    if x_b_esmc_list else torch.zeros(0, 2560))
    e_ag_all = (torch.cat(e_ag_list, dim=1)
                if e_ag_list else torch.zeros(2, 0, dtype=torch.long))
    e_ab_all = (torch.cat(e_ab_list, dim=1)
                if e_ab_list else torch.zeros(2, 0, dtype=torch.long))
    x_b_batch_all = (torch.cat(x_b_batch_list, dim=0)
                     if x_b_batch_list else torch.zeros(0, dtype=torch.long))

    max_rep = max(rep_surf_lens) if rep_surf_lens else 0
    y_pad = torch.zeros(B, max_rep, dtype=torch.float32)
    y_mask = torch.zeros(B, max_rep, dtype=torch.bool)
    for gi in range(B):
        L = rep_surf_lens[gi]
        y_pad[gi, :L] = y_list[gi]
        y_mask[gi, :L] = True

    # PDB-ATOM (resolved) side-channel: scatter surface probs directly to
    # ATOM-coord space via a precomputed per-node index, matching the
    # student's "resolved" eval protocol.
    surf_idx_in_resolved_per_rep = [it["surf_idx_in_resolved"] for it in batch]
    n_resolved_per_rep = [int(it["n_resolved"]) for it in batch]
    y_resolved_per_rep = [it["y_resolved"] for it in batch]

    return {
        "x_g": x_g_all,
        "x_b": x_b_all,
        "x_g_esmc": x_g_esmc_all,
        "x_b_esmc": x_b_esmc_all,
        "edge_index_g": e_ag_all,
        "edge_index_b": e_ab_all,
        "x_b_batch": x_b_batch_all,
        "ag_pairs": ag_pairs_all,
        "ab_pairs": ab_pairs_all,
        "ab_pairs_self": ab_pairs_self_all,
        "ab_pairs_other": ab_pairs_other_all,
        "rep_surf_lens": torch.tensor(rep_surf_lens, dtype=torch.long),
        "y_g_pad": y_pad,
        "y_mask": y_mask,
        "rep_abdbids": [it["rep_abdbid"] for it in batch],
        # PDB-ATOM (resolved) side-channel — the only reporting space.
        "surf_idx_in_resolved_per_rep": surf_idx_in_resolved_per_rep,
        "n_resolved_per_rep": n_resolved_per_rep,
        "y_resolved_per_rep": y_resolved_per_rep,
    }


# ---------------------------------------------------------------------------
# lazy member-graph loader (release-backed)
# ---------------------------------------------------------------------------

class ReleaseMemberGraphLoader:
    """Callable ``abdbid -> release ComplexData`` backed by Complex1723.

    Builds a per-abdbid index once, then serves individual graphs on
    demand. Loading happens inside DataLoader worker processes.
    """

    def __init__(self, dataset_root: str | Path):
        root = detect_release_root(dataset_root)
        release = _import_release_module(root)
        self._datasets = {
            s: release.Complex1723Dataset(
                root=str(root / "Complex1723"), split=s,
            )
            for s in ("train", "val", "test")
        }
        self._index: Dict[str, Tuple[str, int]] = {}
        for s, ds in self._datasets.items():
            for i, ab in enumerate(ds.abdbids):
                self._index[ab] = (s, i)

    def __call__(self, abdbid: str):
        s, i = self._index[abdbid]
        return self._datasets[s].get(i)
