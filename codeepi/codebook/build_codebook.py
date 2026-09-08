"""Build the CodeEpi teacher-derived 4-bin codebook.

Given a **teacher residue table** (a ``.pt`` file with the trained teacher's
per-training-residue probabilities and 256-d post-QKV antibody-context antigen embeddings), the builder:

  1. Digitizes each residue's teacher probability into ``K=4`` bins whose cuts
     are provided on the CLI (defaults ``0.06 / 0.50 / 0.87``).
  2. Computes prototypes per bin over the row-normalized embeddings:
     - ``C_weighted_mean`` : teacher-probability-weighted mean (used to
       initialize the learnable ``W`` and the anchor by the student head)
  3. Emits ``bin_score`` = geometric mean of each bin's cut edges
     (``bin_0`` floored at ``1e-7``), which the student uses as
     ``p = sum_k a_k * bin_score_k``.

The residue table itself is produced by a one-off teacher-forward pass over
the training complexes (see the internal notes; the release ships the
resulting codebook so usually do not need to regenerate it).

CLI
---
    python -m codeepi.codebook.build_codebook \
        --residue-table teacher_residue_table.pt \
        --out checkpoints_CodeEpi_v1.0/codebook_4bin.pt \
        --e1 0.06 --e2 0.50 --e3 0.87
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .utils import DEFAULT_BIN_EDGES, validate_edges


def bin_score_from_cut_edges(edges: Sequence[float]) -> torch.Tensor:
    """Per-bin score = geometric mean of the bin's (left, right) probability
    edges. Uses the closed [0, 1] range; ``bin_0`` is floored at ``1e-7``
    so an accidentally-tiny ``e1`` cannot collapse it to zero.
    """
    e1, e2, e3 = validate_edges(edges)
    full = np.array([0.0, e1, e2, e3, 1.0], dtype=np.float64)
    full = np.clip(full, 1e-9, 1.0)
    centers = np.sqrt(full[:-1] * full[1:])
    # Floor bin-0 score to 1e-7. With canonical e1=0.06 the floor is never
    # triggered (centers[0] ≈ 7.75e-6), but guards against extremely small e1.
    centers[0] = max(centers[0], 1e-7)
    return torch.from_numpy(np.clip(centers, 0.0, 1.0).astype(np.float32))


def build_codebook_from_residue_table(
    residue_table_path: str | Path,
    edges: Iterable[float],
    d_proto: int = 256,
) -> Dict[str, torch.Tensor | int | float | list]:
    """Compute a CodeEpi codebook from a teacher residue table.

    The residue table is expected to be a ``.pt`` dict with:
        ``split``           : list[str]   ("train"/"val"/"test")
        ``teacher_prob``    : Tensor (M,) teacher-forwarded probability
        ``after_qkv_emb_{d_proto}`` : Tensor (M, d_proto)  post-QKV embeddings

    Only ``split == "train"`` residues are used.
    """
    e1, e2, e3 = validate_edges(edges)
    K = 4
    cuts = np.array([e1, e2, e3], dtype=np.float64)
    full_edges = np.array([0.0, e1, e2, e3, 1.0 + 1e-6], dtype=np.float64)
    bin_score = bin_score_from_cut_edges(cuts)

    rt = torch.load(str(residue_table_path), map_location="cpu", weights_only=False)
    split = np.array(rt["split"])
    train_mask = split == "train"
    prob_train = rt["teacher_prob"][train_mask].numpy()
    emb_key = f"after_qkv_emb_{d_proto}"
    if emb_key not in rt:
        raise KeyError(
            f"residue table missing '{emb_key}'. available keys: {list(rt.keys())}"
        )
    emb_train = rt[emb_key][train_mask].float()

    bin_id = np.digitize(prob_train, cuts, right=False).astype(np.int64)
    bin_id = np.clip(bin_id, 0, K - 1)

    emb_n = F.normalize(emb_train, dim=-1)
    C_mean = torch.zeros(K, d_proto)
    C_wmean = torch.zeros(K, d_proto)
    per_bin_n = []
    w_all = torch.from_numpy(prob_train).float()
    for k in range(K):
        m = torch.from_numpy((bin_id == k).astype(bool))
        per_bin_n.append(int(m.sum().item()))
        if not m.any():
            C_mean[k] = torch.randn(d_proto) / np.sqrt(d_proto)
            C_wmean[k] = torch.randn(d_proto) / np.sqrt(d_proto)
            continue
        e_k = emb_n[m]
        w_k = w_all[m].clamp(min=1e-6)
        C_mean[k] = e_k.mean(0)
        C_wmean[k] = (e_k * w_k.unsqueeze(-1)).sum(0) / w_k.sum()
    C_mean = F.normalize(C_mean, dim=-1)
    C_wmean = F.normalize(C_wmean, dim=-1)

    return {
        "K": K,
        "d_proto": int(d_proto),
        "C_mean": C_mean,
        "C_weighted_mean": C_wmean,
        "bin_score": bin_score,
        "bin_edges": torch.tensor(cuts, dtype=torch.float32),
        "full_edges": torch.tensor(full_edges, dtype=torch.float32),
        "per_bin_n": per_bin_n,
        "note": "CodeEpi 4-bin codebook (geometric-mean bin_score, bin0 floor 1e-7)",
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="codeepi.codebook.build_codebook",
        description="Build the CodeEpi teacher-derived 4-bin codebook.",
    )
    p.add_argument("--residue-table", type=str, required=True,
                   help="Path to teacher_residue_table.pt.")
    p.add_argument("--out", type=str, required=True,
                   help="Output .pt path for the codebook (default suffix .pt).")
    p.add_argument("--e1", type=float, default=DEFAULT_BIN_EDGES[0])
    p.add_argument("--e2", type=float, default=DEFAULT_BIN_EDGES[1])
    p.add_argument("--e3", type=float, default=DEFAULT_BIN_EDGES[2])
    p.add_argument("--d-proto", type=int, default=256,
                   help="Prototype dimensionality (matches the teacher's post-QKV).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output.")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    out_pt = Path(args.out)
    if out_pt.exists() and not args.force:
        raise SystemExit(f"[abort] {out_pt} exists; use --force to overwrite.")
    out_pt.parent.mkdir(parents=True, exist_ok=True)

    payload = build_codebook_from_residue_table(
        residue_table_path=args.residue_table,
        edges=(args.e1, args.e2, args.e3),
        d_proto=args.d_proto,
    )
    torch.save(payload, str(out_pt))

    # sidecar JSON summary
    js = {
        "edges": [args.e1, args.e2, args.e3],
        "bin_score": payload["bin_score"].tolist(),
        "per_bin_n": payload["per_bin_n"],
        "K": payload["K"],
        "d_proto": payload["d_proto"],
    }
    with open(out_pt.with_suffix(".json"), "w") as fh:
        json.dump(js, fh, indent=2)
    print(f"[codeepi.codebook] wrote {out_pt}")
    print(f"[codeepi.codebook] wrote {out_pt.with_suffix('.json')}")
    print(f"[codeepi.codebook] per_bin_n={payload['per_bin_n']} "
          f"bin_score={payload['bin_score'].tolist()}")


if __name__ == "__main__":
    main()
