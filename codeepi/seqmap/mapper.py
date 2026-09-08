"""Top-level API: turn seqres positions into concrete PDB residues.

Pipeline
--------
1. Extract `atmseq` and per-residue metadata from the PDB chain.
2. NW-align seqres <-> atmseq (BLOSUM62, affine gap).
3. Greedy same-character walk on seqres <-> atmseq.
4. Build anchor set (positions the two alignments agree on).
5. For each seqres position of interest, walk the anchor prediction;
   fall back to greedy only when the anchor cannot decide.
6. Emit the PDB residue label (resseq + icode) for that seqres index.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from .pdb_io import PDBResidue, extract_atmseq
from .nw import nw_aligned_map, greedy_same_char_map
from .anchor import build_anchor_set


def _seqres_idx_to_atm_idx(seqres_idx: int,
                           cw: dict, gr: dict) -> Tuple[Optional[int], str]:
    """Direct forward map from a seqres index to an atmseq index,
    preferring NW; fall back to greedy if NW put this position in a gap.
    """
    if seqres_idx in cw:
        return cw[seqres_idx], "nw"
    if seqres_idx in gr:
        return gr[seqres_idx], "greedy"
    return None, "unmapped"


def seqres_to_pdb(seqres_positions: Iterable[int],
                  seqres: str,
                  pdb_path: str,
                  chain_id: str,
                  *,
                  open_gap: float = -10.0,
                  extend_gap: float = -1.0
                  ) -> List[Optional[PDBResidue]]:
    """Map a list of 0-based seqres indices to PDB residues on `chain_id`.

    Returns a parallel list; entries are `None` where the position falls
    into a gap in both NW and greedy alignments (i.e. the seqres residue
    is not observed in the structure).
    """
    atmseq, residues = extract_atmseq(pdb_path, chain_id)
    if not residues:
        raise ValueError(f"chain {chain_id!r} not found in {pdb_path}")
    cw = nw_aligned_map(seqres, atmseq, open_gap=open_gap, extend_gap=extend_gap)
    gr = greedy_same_char_map(seqres, atmseq)
    out: List[Optional[PDBResidue]] = []
    for s in seqres_positions:
        a, _ = _seqres_idx_to_atm_idx(int(s), cw, gr)
        out.append(residues[a] if a is not None and 0 <= a < len(residues) else None)
    return out


def seqres_positions_to_pdb(seqres_positions: Iterable[int],
                            atm_positions: Optional[Iterable[int]],
                            seqres: str,
                            pdb_path: str,
                            chain_id: str,
                            *,
                            correct_with_anchor: bool = True,
                            allow_terminal_single_side: bool = True,
                            open_gap: float = -10.0,
                            extend_gap: float = -1.0
                            ) -> Tuple[List[Optional[PDBResidue]], List[dict]]:
    """Batch mapping with optional anchor-based correction.

    If `atm_positions` is provided and `correct_with_anchor` is True,
    each (stored_seqres, atm) pair is passed through `anchor_correct`
    (anchor -> greedy fallback -> stored fallback) before the final
    seqres->PDB lookup.
    """
    atmseq, residues = extract_atmseq(pdb_path, chain_id)
    if not residues:
        raise ValueError(f"chain {chain_id!r} not found in {pdb_path}")

    seqres_positions = [int(s) for s in seqres_positions]

    if correct_with_anchor and atm_positions is not None:
        from .anchor import anchor_correct
        atm_positions = [int(a) for a in atm_positions]
        corrected, report = anchor_correct(
            seqres_positions, atm_positions, seqres, atmseq, residues,
            allow_terminal_single_side=allow_terminal_single_side,
            open_gap=open_gap, extend_gap=extend_gap)
    else:
        corrected = list(seqres_positions)
        report = [{"k": k, "stored": s, "pred": s,
                   "mode": "no_correction", "status": "unchanged"}
                  for k, s in enumerate(seqres_positions)]

    cw = nw_aligned_map(seqres, atmseq, open_gap=open_gap, extend_gap=extend_gap)
    gr = greedy_same_char_map(seqres, atmseq)
    residues_out: List[Optional[PDBResidue]] = []
    for s in corrected:
        a, _ = _seqres_idx_to_atm_idx(s, cw, gr)
        residues_out.append(residues[a] if a is not None and 0 <= a < len(residues) else None)
    return residues_out, report
