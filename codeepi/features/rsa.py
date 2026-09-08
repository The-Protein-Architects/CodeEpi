"""Relative solvent-accessible area (RSA) via DSSP.

Wraps ``Bio.PDB.DSSP`` and returns a per-residue RSA vector, aligned to the
residue order produced by ``pdb_parser.parse_pdb``.

DSSP must be installed on the system (``mkdssp`` binary). If unavailable, the
function raises a clear error.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


# Sander & Rost 1994 reference max ASA (A^2) for standard amino acids.
_MAX_ASA = {
    "A": 115.0, "R": 265.0, "N": 150.0, "D": 160.0, "C": 135.0,
    "E": 190.0, "Q": 180.0, "G": 75.0,  "H": 195.0, "I": 175.0,
    "L": 170.0, "K": 200.0, "M": 185.0, "F": 210.0, "P": 145.0,
    "S": 115.0, "T": 140.0, "W": 255.0, "Y": 230.0, "V": 155.0,
}


def compute_rsa(
    pdb_path: str | Path,
    chain_of: Sequence[str],
    resnum: Sequence[str],
    seq: Sequence[str],
    dssp_binary: str = "mkdssp",
) -> np.ndarray:
    """Return a ``(N_res,)`` RSA vector aligned to ``(chain_of, resnum)``.

    Notes
    -----
    * Uses ``Bio.PDB.DSSP``. Requires ``mkdssp`` on ``PATH`` — the released
      teacher and student were trained with RSA features, and silently
      substituting zeros here degrades prediction quality without warning.
      Callers get a hard :class:`RuntimeError` if DSSP is missing or fails.
    """
    import shutil

    try:
        from Bio.PDB import DSSP, PDBParser
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "codeepi.features.rsa requires biopython. Install with "
            "`pip install biopython` (and make sure `mkdssp` is on PATH)."
        ) from e

    if shutil.which(dssp_binary) is None:
        raise RuntimeError(
            f"DSSP binary {dssp_binary!r} not found on PATH. Install DSSP "
            "(``conda install -c salilab dssp`` or the OS package "
            "``mkdssp``) so RSA can be computed; the released checkpoints "
            "were trained with RSA and cannot run without it."
        )

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    model = next(structure.get_models())

    try:
        dssp = DSSP(model, str(pdb_path), dssp=dssp_binary)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            f"DSSP failed on {pdb_path}: {e!r}. Ensure the PDB is standard "
            "(no altlocs/hydrogens/multi-model issues) and that "
            f"{dssp_binary!r} is a working DSSP install."
        ) from e

    # DSSP results are keyed by (chain_id, (' ', seqid, icode)). Bio.PDB.DSSP
    # returns a 14-tuple per residue:
    #   (dssp_idx, aa, ss, rel_ASA, phi, psi, NH_O_1_relidx, NH_O_1_energy, ...)
    # index 3 is the RELATIVE ASA (already normalised by Bio.PDB's max-ASA
    # scale, i.e. in [0,1]); index 4 is the phi dihedral. Use index 3 directly
    # as the RSA — do NOT re-divide by _MAX_ASA (that table applies to an
    # ABSOLUTE ASA, which this tuple does not carry). DSSP marks residues it
    # cannot compute with 'NA'.
    rsa = np.zeros(len(seq), dtype=np.float32)
    for i, (c, rn, aa) in enumerate(zip(chain_of, resnum, seq)):
        # split resnum back into (seqid, icode)
        icode = " "
        if rn and rn[-1].isalpha():
            icode = rn[-1]
            seqid = int(rn[:-1])
        else:
            seqid = int(rn)
        key = (c, (" ", seqid, icode))
        try:
            rec = dssp[key]
        except KeyError:
            continue
        rel_asa = rec[3]
        try:
            rel_asa = float(rel_asa)
        except (TypeError, ValueError):
            continue  # 'NA' -> leave as 0
        rsa[i] = float(np.clip(rel_asa, 0.0, 1.0))
    return rsa
