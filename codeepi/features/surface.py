"""Surface-residue identification from RSA.

Uses a simple RSA threshold on the standardized RSA vector to declare surface
residues.
"""
from __future__ import annotations
from typing import Sequence

import numpy as np


# Training recipe: surface = rsa > 0.
# Any solvent-exposed residue is surface. 
DEFAULT_RSA_THRESHOLD = 0.0


def compute_surface(rsa: np.ndarray, threshold: float = DEFAULT_RSA_THRESHOLD) -> np.ndarray:
    """Return a boolean ``(N_res,)`` mask over antigen residues.
    """
    rsa = np.asarray(rsa)
    t = float(threshold)
    if t == 0.0:
        return (rsa > 0.0).astype(bool)          # training recipe: rsa > 0
    return (rsa >= t).astype(bool)
