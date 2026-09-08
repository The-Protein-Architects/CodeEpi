"""Codebook helper utilities."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


# Reference CodeEpi codebook cut edges (K=4).
# Corresponds to the winning bin edges from the internal binedge sweep.
# Single source of truth — imported by build_codebook.py CLI defaults and
# by predict.postprocess.load_codebook_bin_edges fallback so any change
# is picked up in one place.
DEFAULT_BIN_EDGES: Tuple[float, float, float] = (0.06, 0.50, 0.87)


def default_bin_edges() -> Tuple[float, float, float]:
    """Reference CodeEpi codebook cut edges (K=4)."""
    return DEFAULT_BIN_EDGES


def load_codebook(path: str | Path) -> Dict:
    """Load a codebook ``.pt`` written by ``build_codebook.py``.

    Returns the full dict (``K``, ``d_proto``, ``bin_score``, ``bin_edges``,
    ``full_edges``, ``C_mean``, ``C_weighted_mean``, ``per_bin_n``).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"codebook file not found: {p}")
    return torch.load(str(p), map_location="cpu", weights_only=False)


def validate_edges(edges: Iterable[float]) -> Tuple[float, float, float]:
    """Sanity-check the 3 cut edges: strictly increasing in (0, 1].

    ``e3 == 1.0`` is allowed (bin 3 becomes the virtual right edge; the
    student never emits ``p >= 1`` due to sigmoid clamping, so bin 3 is
    empty in that case — degenerate but not an error). Matches
    ``codeepi.predict.postprocess.rank_and_bin``.
    """
    e = tuple(float(x) for x in edges)
    if len(e) != 3:
        raise ValueError(f"expected 3 cut edges, got {len(e)}")
    if not (0.0 < e[0] < e[1] < e[2] <= 1.0):
        raise ValueError(f"edges must satisfy 0 < e1 < e2 < e3 <= 1, got {e}")
    return e  # type: ignore[return-value]
