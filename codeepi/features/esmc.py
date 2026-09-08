"""ESM-C access for prediction: **precomputed**.

The prediction pipeline (``codeepi.predict.student_antigen_ensemble``) 
loads a user-supplied ``.pt`` embedding of the full-length antigen SEQRES. 
This keeps the prediction environment
(``codeepi``, which carries ``fair-esm`` for ESM-IF) free of the
EvolutionaryScale ``esm`` client, which shares the ``esm`` import namespace
with ``fair-esm`` and cannot be co-installed.

To produce the ``.pt`` from an antigen FASTA, run the standalone encoder
``codeepi.features.encode_esmc`` inside the separate ``codeepi-esmc``
environment (see docs/prediction.md), then feed its output path back here
via ``--esmc_ag_pt``.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch


ESMC_DIM = 2560
ESMC_MODEL_DEFAULT = "esmc-6b-2024-12"


def load_precomputed(pt_path: str | Path, expect_n: Optional[int] = None) -> torch.Tensor:
    """Load a precomputed ESM-C tensor from ``pt_path``.

    Accepts either a raw ``Tensor`` of shape ``(N, 2560)`` or a dict with
    key ``embedding`` of the same shape.
    """
    p = Path(pt_path)
    if not p.exists():
        raise FileNotFoundError(f"precomputed ESM-C not found: {p}")
    data = torch.load(str(p), map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        emb = data.get("embedding")
    else:
        emb = data
    if not torch.is_tensor(emb):
        raise ValueError(f"expected torch.Tensor for ESM-C, got {type(emb)}")
    emb = emb.float()
    if emb.ndim != 2 or emb.size(-1) != ESMC_DIM:
        raise ValueError(
            f"ESM-C tensor must have shape (N, {ESMC_DIM}); got {tuple(emb.shape)}"
        )
    if expect_n is not None and emb.size(0) != expect_n:
        raise ValueError(
            f"ESM-C row count {emb.size(0)} != expected residue count {expect_n}"
        )
    return emb
