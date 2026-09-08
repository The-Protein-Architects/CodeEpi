"""correction of seqres <-> atmseq alignment.
    anchor = { A : cw_inv[A] == gr_inv[A] }
    pred_L = anchor[L] + (resseq[T] - resseq[L])
    pred_R = anchor[R] - (resseq[R] - resseq[T])
"""
from __future__ import annotations

import bisect
from typing import Dict, List, Optional, Tuple

from .nw import nw_aligned_map, greedy_same_char_map
from .pdb_io import PDBResidue


AnchorPrediction = Tuple[Optional[int], str, Optional[int], Optional[int]]
# (predicted_seqres_index, mode, left_anchor_atm, right_anchor_atm)


# ------------------------------ core anchor set ---------------------------

def build_anchor_set(seqres: str, atmseq: str,
                     open_gap: float = -10.0,
                     extend_gap: float = -1.0
                     ) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    - cw_map : seqres_idx -> atmseq_idx from Needleman-Wunsch.
    - gr_map : seqres_idx -> atmseq_idx from greedy same-char walk.
    - anchor : atmseq_idx -> seqres_idx for atmseq indices where both
               maps agree (i.e. cw_inv[A] == gr_inv[A]).
    """
    cw = nw_aligned_map(seqres, atmseq, open_gap=open_gap, extend_gap=extend_gap)
    gr = greedy_same_char_map(seqres, atmseq)
    cw_inv = {a: s for s, a in cw.items()}
    gr_inv = {a: s for s, a in gr.items()}
    anchor = {A: cw_inv[A] for A in cw_inv if A in gr_inv and cw_inv[A] == gr_inv[A]}
    return cw, gr, anchor


# ------------------------------ anchor prediction -------------------------

def anchor_predict(atm: int,
                   residues: List[PDBResidue],
                   anchor: Dict[int, int],
                   atms_sorted: Optional[List[int]] = None,
                   *,
                   allow_terminal_single_side: bool = True
                   ) -> AnchorPrediction:
    """Predict the seqres index for atmseq index `atm`.

    Modes returned in the tuple[1]:
        "self"                      atm itself is an anchor
        "neighbour_agree"           pred_L == pred_R
        "neighbour_disagree"        pred_L != pred_R (no prediction)
        "terminal_left_only"        atm precedes all anchors — right-only
                                    prediction accepted (if enabled)
        "terminal_right_only"       atm follows all anchors — left-only
                                    prediction accepted (if enabled)
        "no_anchor_one_side"        interior one-side (rejected)
        "no_anchor_any_side"        empty anchor set
    """
    if atms_sorted is None:
        atms_sorted = sorted(anchor)

    if atm in anchor:
        return anchor[atm], "self", None, None

    if not atms_sorted:
        return None, "no_anchor_any_side", None, None

    lo = bisect.bisect_left(atms_sorted, atm)
    left = atms_sorted[lo - 1] if lo - 1 >= 0 else None
    ri = bisect.bisect_right(atms_sorted, atm)
    right = atms_sorted[ri] if ri < len(atms_sorted) else None

    R_T = residues[atm].resseq

    if left is None and right is None:
        return None, "no_anchor_any_side", None, None

    if left is None:
        # atm precedes the first anchor — sequence-head extremity.
        if not allow_terminal_single_side:
            return None, "no_anchor_one_side", left, right
        pR = anchor[right] - (residues[right].resseq - R_T)
        return pR, "terminal_left_only", left, right

    if right is None:
        # atm follows the last anchor — sequence-tail extremity.
        if not allow_terminal_single_side:
            return None, "no_anchor_one_side", left, right
        pL = anchor[left] + (R_T - residues[left].resseq)
        return pL, "terminal_right_only", left, right

    pL = anchor[left] + (R_T - residues[left].resseq)
    pR = anchor[right] - (residues[right].resseq - R_T)
    if pL != pR:
        return None, "neighbour_disagree", left, right
    return pL, "neighbour_agree", left, right


# ------------------------------ full correction path ---------------------

CORRECTION_STATUS = {
    "unchanged",         # anchor confirms the stored value
    "corrected_anchor",  # anchor overrides the stored value
    "corrected_greedy",  # anchor abstained, greedy overrides
    "kept_no_pred",      # anchor abstained, greedy absent, stored kept
}


def anchor_correct(stored_seqres_positions: List[int],
                   atm_positions: List[int],
                   seqres: str,
                   atmseq: str,
                   residues: List[PDBResidue],
                   *,
                   allow_terminal_single_side: bool = True,
                   greedy_fallback: bool = True,
                   open_gap: float = -10.0,
                   extend_gap: float = -1.0
                   ) -> Tuple[List[int], List[dict]]:
    """Take a stored (seqres_pos, atm_pos) pair-list and produce a
    corrected seqres_pos list, plus a per-position audit trail.
    """
    cw, gr, anchor = build_anchor_set(seqres, atmseq,
                                      open_gap=open_gap,
                                      extend_gap=extend_gap)
    atms_sorted = sorted(anchor)
    gr_inv = {a: s for s, a in gr.items()}

    out_positions: List[int] = []
    report: List[dict] = []

    for k, (stored, atm) in enumerate(zip(stored_seqres_positions, atm_positions)):
        entry = {"k": k, "atm": atm, "stored": stored,
                 "pred": None, "mode": None, "status": None,
                 "left": None, "right": None}
        if not (0 <= atm < len(residues)):
            entry["mode"] = "atm_out_of_range"
            entry["status"] = "kept_no_pred"
            out_positions.append(stored)
            report.append(entry)
            continue
        pred, mode, left, right = anchor_predict(
            atm, residues, anchor, atms_sorted,
            allow_terminal_single_side=allow_terminal_single_side)
        entry["left"] = left; entry["right"] = right; entry["mode"] = mode

        if pred is not None:
            entry["pred"] = pred
            if pred == stored:
                entry["status"] = "unchanged"
            else:
                entry["status"] = "corrected_anchor"
            out_positions.append(pred)
            report.append(entry)
            continue

        # anchor abstained -> try greedy fallback
        if greedy_fallback and atm in gr_inv:
            g = gr_inv[atm]
            entry["pred"] = g
            entry["mode"] = f"{mode}+greedy"
            entry["status"] = "unchanged" if g == stored else "corrected_greedy"
            out_positions.append(g)
            report.append(entry)
            continue

        entry["status"] = "kept_no_pred"
        out_positions.append(stored)
        report.append(entry)

    return out_positions, report
