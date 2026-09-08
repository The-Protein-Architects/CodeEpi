"""Minimal PDB parser used by the prediction CLIs.

Uses ``Bio.PDB`` to iterate over residues in the requested chains, returning:

  * per-residue backbone atoms  (N, CA, C)  as a ``(N_res, 3, 3)`` array
  * per-residue single-letter amino-acid sequence
  * per-residue PDB residue numbers (auth_seq_id + insertion code)
  * per-residue chain IDs (in the same order)

Only standard 20 amino acids are kept (residues without N/CA/C are dropped).

``parse_pdb_heavy_atoms`` returns the full non-hydrogen atom coordinate list
per residue in the exact same residue order as ``parse_pdb`` (aligned by
``(chain_id, resnum)``). Used by the teacher-complex predictor to compute
real heavy-atom 4.5 Å CDR↔surface contact edges (matching the training
release ``edge_index_contact``).
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _fmt_resnum(auth_seq_id: int, insertion: str) -> str:
    return f"{auth_seq_id}{insertion.strip()}" if insertion.strip() else str(auth_seq_id)


def parse_pdb(pdb_path: str | Path, chain_ids: Sequence[str]) -> Dict[str, np.ndarray | List]:
    """Parse ``pdb_path`` and return the residues of ``chain_ids`` in order.

    Returns a dict with:
        ``seq``           : list[str]  single-letter sequence (concatenated across chains)
        ``chain_of``      : list[str]  chain ID of each residue
        ``resnum``        : list[str]  PDB residue number (with insertion code)
        ``backbone``      : (N_res, 3, 3) float32 array with N, CA, C
        ``residues_by_chain`` : {chain_id: [(resnum, aa), ...]}
    """
    try:
        from Bio.PDB import PDBParser
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "codeepi.features.pdb_parser requires biopython. "
            "Install with `pip install biopython`."
        ) from e

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    model = next(structure.get_models())

    seq: List[str] = []
    chain_of: List[str] = []
    resnum: List[str] = []
    backbone: List[np.ndarray] = []
    residues_by_chain: Dict[str, List[Tuple[str, str]]] = {c: [] for c in chain_ids}

    for cid in chain_ids:
        if cid not in [c.id for c in model.get_chains()]:
            raise ValueError(f"chain {cid!r} not found in {pdb_path}")
        chain = model[cid]
        for residue in chain:
            hetflag, seqid, icode = residue.id
            if hetflag.strip() != "":
                continue
            resname = residue.get_resname().strip().upper()
            aa = _THREE_TO_ONE.get(resname)
            if aa is None:
                continue
            try:
                n = residue["N"].coord
                ca = residue["CA"].coord
                c = residue["C"].coord
            except KeyError:
                continue
            bb = np.stack([n, ca, c], axis=0).astype(np.float32)
            rn = _fmt_resnum(seqid, icode)
            seq.append(aa)
            chain_of.append(cid)
            resnum.append(rn)
            backbone.append(bb)
            residues_by_chain[cid].append((rn, aa))

    if not backbone:
        raise ValueError(f"no standard amino-acid residues found in chains {chain_ids} of {pdb_path}")

    return {
        "seq": seq,
        "chain_of": chain_of,
        "resnum": resnum,
        "backbone": np.stack(backbone, axis=0),
        "residues_by_chain": residues_by_chain,
    }


def parse_pdb_heavy_atoms(
    pdb_path: str | Path,
    chain_of: Sequence[str],
    resnum: Sequence[str],
) -> List[np.ndarray]:
    """Return the heavy-atom coords (excluding hydrogens) for each residue in
    ``(chain_of[i], resnum[i])`` order. Element ``i`` is an ``(A_i, 3)`` float32
    array. Residues not found in the PDB (should not happen for outputs of
    :func:`parse_pdb`) yield an empty ``(0, 3)`` array.
    """
    try:
        from Bio.PDB import PDBParser
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "codeepi.features.pdb_parser requires biopython. "
            "Install with `pip install biopython`."
        ) from e

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    model = next(structure.get_models())

    lookup: Dict[Tuple[str, str], np.ndarray] = {}
    for chain in model.get_chains():
        cid = chain.id
        for residue in chain:
            hetflag, seqid, icode = residue.id
            if hetflag.strip() != "":
                continue
            rn = _fmt_resnum(seqid, icode)
            coords = [
                atom.coord for atom in residue.get_atoms()
                if atom.element.strip().upper() != "H"
            ]
            if coords:
                lookup[(cid, rn)] = np.asarray(coords, dtype=np.float32)

    out: List[np.ndarray] = []
    for c, r in zip(chain_of, resnum):
        out.append(lookup.get((c, r), np.zeros((0, 3), dtype=np.float32)))
    return out
