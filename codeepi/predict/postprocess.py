"""Post-processing for prediction outputs.

Consolidates writing:

* ``residue_predictions.csv``  (residue-level table)
* ``residue_predictions.json`` (same content, JSON form)
* ``antigen_colored_by_probability.pdb`` (input PDB with epitope probability
  written into the antigen residues' b-factor column, capped to two decimals)
* ``config_used.yaml``  (record of every runtime parameter)
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


from ..codebook.utils import DEFAULT_BIN_EDGES


def rank_and_bin(
    probability: np.ndarray,
    edges: Sequence[float] = DEFAULT_BIN_EDGES,
) -> Dict[str, np.ndarray]:
    """Add ``rank`` (1-based, descending prob) and ``probability_bin`` columns.

    ``edges`` are the K-1=3 cut points that partition ``[0, 1]`` into 4 bins.
    Callers should read them from the codebook file
    (``codebook_4bin.pt["bin_edges"]``) so a rebuilt codebook stays
    consistent with the reported ``probability_bin`` column.
    """
    prob = np.asarray(probability, dtype=np.float64)
    order = np.argsort(-prob, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(prob) + 1)

    e = tuple(float(x) for x in edges)
    if len(e) != 3 or not (0.0 < e[0] < e[1] < e[2] <= 1.0):
        raise ValueError(
            f"rank_and_bin: expected 3 strictly-increasing edges in (0, 1], got {edges}"
        )
    bins = np.zeros(len(prob), dtype=np.int64)
    bins[(prob >= 0.0) & (prob < e[0])] = 0
    bins[(prob >= e[0]) & (prob < e[1])] = 1
    bins[(prob >= e[1]) & (prob < e[2])] = 2
    bins[(prob >= e[2])] = 3
    return {"rank": ranks, "probability_bin": bins}


def load_codebook_bin_edges(codebook_path: str | Path) -> Tuple[float, float, float]:
    """Load ``bin_edges`` from a codebook ``.pt`` written by ``build_codebook``.

    Falls back to :data:`DEFAULT_BIN_EDGES` when the file is missing the
    field (e.g. a hand-crafted stub codebook).
    """
    import torch

    d = torch.load(str(codebook_path), map_location="cpu", weights_only=False)
    be = d.get("bin_edges")
    if be is None:
        return DEFAULT_BIN_EDGES
    if torch.is_tensor(be):
        be = be.tolist()
    return tuple(float(x) for x in be)  # type: ignore[return-value]


def write_residue_predictions_csv(
    out_path: Path,
    *,
    residue_index: Sequence[int],
    pdb_residue_number: Sequence[str],
    chain_id: Sequence[str],
    aa: Sequence[str],
    probability: Sequence[float],
    is_surface: Sequence[bool],
    rank: Optional[Sequence[int]] = None,
    probability_bin: Optional[Sequence[int]] = None,
    is_epitope: Optional[Sequence[bool]] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["residue_index", "pdb_residue_number", "chain_id", "aa",
              "probability", "is_surface"]
    if rank is not None:
        header.append("rank")
    if probability_bin is not None:
        header.append("probability_bin")
    if is_epitope is not None:
        header.append("is_epitope")

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i in range(len(residue_index)):
            row = [
                int(residue_index[i]),
                str(pdb_residue_number[i]),
                str(chain_id[i]),
                str(aa[i]),
                float(probability[i]),
                bool(is_surface[i]),
            ]
            if rank is not None:
                row.append(int(rank[i]))
            if probability_bin is not None:
                row.append(int(probability_bin[i]))
            if is_epitope is not None:
                row.append(bool(is_epitope[i]))
            w.writerow(row)


def write_residue_predictions_json(out_path: Path, records: List[Dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({"residues": records}, fh, indent=2)


def write_pdb_with_probability(
    in_pdb: Path,
    out_pdb: Path,
    prob_of: Dict[tuple, float],
    scale: float = 100.0,
    restrict_chains: Optional[Sequence[str]] = None,
) -> None:
    allowed = set(restrict_chains) if restrict_chains is not None else None
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    with in_pdb.open("r", encoding="utf-8") as fin, \
            out_pdb.open("w", encoding="utf-8") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                chain = line[21].strip()
                if allowed is None or chain in allowed:
                    resnum = line[22:27].strip()
                    key = (chain, resnum)
                    prob = prob_of.get(key, 0.0)
                    bf = f"{min(999.99, max(-999.99, prob * scale)):6.2f}"
                    line = line[:60] + bf + line[66:]
            fout.write(line)


def dump_config_used(out_path: Path, config: Dict) -> None:
    """Write a YAML record of the runtime parameters used by the prediction."""
    import yaml

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=True)
