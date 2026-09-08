"""Anchor alignment between structure (atmseq) and SEQRES (FASTA) spaces.

Prediction contract
-------------------
* ESM-C is computed on the full-length **SEQRES** (from the FASTA).
* ESM-IF / DSSP / heavy-atom contacts live in the **structure** (atmseq)
  space produced from the PDB ATOM records.
* The teacher/student consume the intersection for training: structure residues
  (CDR / surface), with ESM-C cropped to the matching atmseq positions.

This module reconciles the two spaces with the vendored ``codeepi.seqmap``
anchor algorithm. It is the single place that:

1. builds a per-chain anchor set (SEQRES vs atmseq),
2. maps a structure-space residue selection (e.g. CDR or surface atmseq
   indices) back to concatenated-SEQRES positions, honouring the
   per-chain SEQRES offset (heavy then light; antigen chains in order),
3. enforces the three-parser AA-identity guard (``load_coords`` seq vs
   ``extract_atmseq`` atmseq vs ``parse_pdb`` sequence).

Every failure is a hard error — no silent fallback (mirrors ``rsa.py``
raising when DSSP is missing).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..seqmap import build_anchor_set, extract_atmseq
from ..seqmap.anchor import anchor_predict
from ..seqmap.pdb_io import PDBResidue


@dataclass
class ChainAlign:
    """Per-chain alignment state between its SEQRES and its atmseq."""
    chain_id: str
    seqres: str
    atmseq: str
    residues: List[PDBResidue]
    anchor: Dict[int, int]        # atmseq_idx -> seqres_idx (agreement set)
    atms_sorted: List[int]

    @property
    def n_atm(self) -> int:
        return len(self.residues)

    @property
    def n_seqres(self) -> int:
        return len(self.seqres)


def build_chain_align(pdb_path: str, chain_id: str, seqres: str) -> ChainAlign:
    """Build the anchor alignment for one chain against its FASTA SEQRES."""
    atmseq, residues = extract_atmseq(pdb_path, chain_id)
    if not residues:
        raise RuntimeError(
            f"chain {chain_id!r} has no standard-AA ATOM residues in {pdb_path}"
        )
    if not seqres:
        raise RuntimeError(f"empty SEQRES supplied for chain {chain_id!r}")
    _, _, anchor = build_anchor_set(seqres, atmseq)
    return ChainAlign(
        chain_id=chain_id,
        seqres=seqres,
        atmseq=atmseq,
        residues=residues,
        anchor=anchor,
        atms_sorted=sorted(anchor),
    )


def atm_local_to_seqres_local(align: ChainAlign, atm_idx: int) -> int:
    """Map a single chain-local atmseq index to its chain-local SEQRES index.

    Uses the anchor prediction (self / neighbour-agree / terminal single
    side). Raises on ``no_pred`` — never keeps a stale/guessed value.
    """
    pred, mode, _, _ = anchor_predict(
        atm_idx, align.residues, align.anchor, align.atms_sorted
    )
    if pred is None:
        raise RuntimeError(
            f"anchor produced no prediction for chain {align.chain_id!r} "
            f"atmseq index {atm_idx} (mode={mode!r}); refusing to guess the "
            f"SEQRES position. Inspect the PDB/FASTA alignment for this chain."
        )
    if not (0 <= pred < align.n_seqres):
        raise RuntimeError(
            f"anchor predicted SEQRES index {pred} out of range "
            f"[0,{align.n_seqres}) for chain {align.chain_id!r} atm {atm_idx}."
        )
    return pred


def map_structure_to_seqres(
    aligns: Sequence[ChainAlign],
    per_chain_atm_local: Sequence[Sequence[int]],
) -> List[int]:
    """Map selected structure residues to concatenated-SEQRES positions.

    ``aligns`` is the ordered per-chain alignment list (heavy, light for
    the antibody; antigen chains in ``--antigen_chain`` order). Element
    ``per_chain_atm_local[c]`` is the list of chain-local atmseq indices
    selected on chain ``c`` (already in the desired output order). The
    returned positions are 0-based indices into the concatenation
    ``seqres[0] ++ seqres[1] ++ ...`` — exactly the space the release
    ``*_seqres_positions`` columns (and the ESM-C tensors) live in.
    """
    if len(aligns) != len(per_chain_atm_local):
        raise ValueError(
            f"aligns ({len(aligns)}) and per_chain_atm_local "
            f"({len(per_chain_atm_local)}) length mismatch"
        )
    out: List[int] = []
    seqres_off = 0
    for align, atm_locals in zip(aligns, per_chain_atm_local):
        for a in atm_locals:
            if not (0 <= a < align.n_atm):
                raise RuntimeError(
                    f"chain {align.chain_id!r} atm index {a} out of range "
                    f"[0,{align.n_atm})"
                )
            out.append(seqres_off + atm_local_to_seqres_local(align, a))
        seqres_off += align.n_seqres
    return out


def assert_aa_identity(
    chain_id: str,
    atmseq: str,
    parse_pdb_seq: str,
    load_coords_seq: Optional[str] = None,
) -> None:
    """Hard-assert the three PDB parsers see the same residues, in order.

    ``atmseq``          from ``codeepi.seqmap.extract_atmseq``
    ``parse_pdb_seq``   from ``codeepi.features.pdb_parser.parse_pdb``
    ``load_coords_seq`` from ``esm.inverse_folding.util.load_coords`` (the
                        ESM-IF input); ``None`` skips that arm (e.g. when
                        the caller has no ESM-IF available).

    ``load_coords`` emits ``'X'`` for residues it cannot map to a standard
    amino acid; those positions are compared leniently (any vs ``X``) so a
    benign coordinate-only ambiguity does not trip the guard, but a real
    identity swap does. Any genuine mismatch raises ``RuntimeError`` with a
    per-position diff, mirroring ``rsa.py``'s no-silent-fallback style.
    """
    def _diff(a: str, b: str, name_a: str, name_b: str) -> None:
        if len(a) != len(b):
            raise RuntimeError(
                f"chain {chain_id!r}: {name_a} length {len(a)} != {name_b} "
                f"length {len(b)}; the PDB parsers disagree on residue count. "
                f"{name_a}[:40]={a[:40]!r} {name_b}[:40]={b[:40]!r}"
            )
        for i, (ca, cb) in enumerate(zip(a, b)):
            if ca == cb:
                continue
            if "X" in (ca, cb):  # coordinate-only ambiguity, tolerate
                continue
            raise RuntimeError(
                f"chain {chain_id!r}: {name_a} vs {name_b} residue identity "
                f"mismatch at position {i}: {ca!r} != {cb!r}. The PDB parsers "
                f"disagree — refusing to proceed (would misalign ESM-C/ESM-IF)."
            )

    _diff(atmseq, parse_pdb_seq, "extract_atmseq", "parse_pdb")
    if load_coords_seq is not None:
        _diff(atmseq, load_coords_seq, "extract_atmseq", "load_coords")


def check_chain_alignment(
    align: ChainAlign,
    atm_locals: Sequence[int],
    seqres_locals: Sequence[Optional[int]],
) -> Tuple[List[Dict], bool]:
    """
    Four checks per residue (the three-parser structure-sequence identity,
    is a chain-level guard handled upstream by ``assert_aa_identity`` and
    is deliberately *not* re-done here):

       in_range     0 <= seqres_local < n_seqres
       aa_match     seqres[seqres_local] == structure residue AA
       injective    no two structure residues map to the same SEQRES pos
       monotonic    SEQRES positions strictly increase in atm order

    A residue ``validated`` only if all four hold. Returns the per-residue
    check dicts and an overall-ok bool.
    """
    if len(atm_locals) != len(seqres_locals):
        raise ValueError(
            f"chain {align.chain_id!r}: atm_locals ({len(atm_locals)}) and "
            f"seqres_locals ({len(seqres_locals)}) length mismatch"
        )
    out: List[Dict] = []
    seen_seqres: Dict[int, int] = {}   # seqres_local -> first atm_local
    last_accepted: Optional[int] = None
    ok_all = True
    for a, s in zip(atm_locals, seqres_locals):
        res = align.residues[a] if 0 <= a < align.n_atm else None
        in_range = s is not None and 0 <= s < align.n_seqres
        aa_match = bool(in_range and res is not None
                        and align.seqres[s] == res.aa)
        injective = bool(in_range and s not in seen_seqres)
        monotonic = bool(in_range and (last_accepted is None or s > last_accepted))
        validated = bool(in_range and aa_match and injective and monotonic)

        detail = ""
        if not in_range:
            detail = f"seqres position {s} out of range [0,{align.n_seqres})"
        elif not aa_match:
            detail = (f"AA mismatch: seqres[{s}]="
                      f"{align.seqres[s]!r} != structure "
                      f"{res.aa!r} (pdb {res.label()})")
        elif not injective:
            detail = (f"SEQRES position {s} already used by atm "
                      f"{seen_seqres[s]} (non-injective mapping)")
        elif not monotonic:
            detail = (f"SEQRES position {s} not increasing after "
                      f"{last_accepted} (order/crossing violation)")

        out.append({
            "aa_match": aa_match,
            "in_range": in_range,
            "injective": injective,
            "monotonic": monotonic,
            "validated": validated,
            "check_detail": detail,
        })
        if in_range:
            seen_seqres.setdefault(s, a)
            if monotonic:
                last_accepted = s
        ok_all = ok_all and validated
    return out, ok_all


def build_seqmap_report(
    aligns: Sequence[ChainAlign],
    per_chain_atm_local: Sequence[Sequence[int]],
    kind: str,
    seqres_locals_override: Optional[Sequence[Sequence[int]]] = None,
) -> Dict:
    """Produce an auditable seqmap record for ``outputs/seqmap.json``.

    ``kind`` is a free-form label (``"ab_cdr"`` / ``"ag_surf"``). The
    report lists, for every selected structure residue, the chain, the
    PDB residue label, the chain-local atm index, the concatenated-SEQRES
    position, the anchor mode used, and whether it validated.

    By default the chain-local SEQRES index for each residue comes from the
    anchor predictor. When ``seqres_locals_override`` is given (one list of
    chain-local SEQRES indices per chain, parallel to
    ``per_chain_atm_local``), those positions are used instead — this is the
    manual-override path. Either way the residue is ``validated`` only if it
    passes the four checks in ``check_chain_alignment``; the ``anchor_mode``
    field records ``"manual_override"`` for override entries.
    """
    if seqres_locals_override is not None and \
            len(seqres_locals_override) != len(aligns):
        raise ValueError(
            f"seqres_locals_override has {len(seqres_locals_override)} chains "
            f"but there are {len(aligns)} aligns"
        )
    entries: List[Dict] = []
    seqres_off = 0
    ok = True
    for ci, (align, atm_locals) in enumerate(zip(aligns, per_chain_atm_local)):
        # chain-local SEQRES index + anchor mode for each residue
        preds: List[Optional[int]] = []
        modes: List[str] = []
        if seqres_locals_override is not None:
            ovr = seqres_locals_override[ci]
            if len(ovr) != len(atm_locals):
                raise ValueError(
                    f"chain {align.chain_id!r}: override has {len(ovr)} "
                    f"positions but {len(atm_locals)} residues were selected"
                )
            for s in ovr:
                preds.append(int(s) if s is not None else None)
                modes.append("manual_override")
        else:
            for a in atm_locals:
                pred, mode, _, _ = anchor_predict(
                    a, align.residues, align.anchor, align.atms_sorted
                )
                preds.append(pred)
                modes.append(mode)

        checks, chain_ok = check_chain_alignment(align, atm_locals, preds)
        ok = ok and chain_ok
        for a, pred, mode, chk in zip(atm_locals, preds, modes, checks):
            res = align.residues[a]
            entries.append({
                "chain_id": align.chain_id,
                "pdb_residue_label": res.label(),
                "aa": res.aa,
                "atm_local_index": int(a),
                "seqres_concat_position": (
                    int(seqres_off + pred) if pred is not None else None
                ),
                "anchor_mode": mode,
                "aa_match": chk["aa_match"],
                "in_range": chk["in_range"],
                "injective": chk["injective"],
                "monotonic": chk["monotonic"],
                "validated": chk["validated"],
                "check_detail": chk["check_detail"],
            })
        seqres_off += align.n_seqres
    failed = [e for e in entries if not e["validated"]]
    return {
        "kind": kind,
        "chains": [
            {"chain_id": al.chain_id, "n_atm": al.n_atm,
             "n_seqres": al.n_seqres, "n_anchor": len(al.anchor)}
            for al in aligns
        ],
        "n_selected": len(entries),
        "n_failed": len(failed),
        "all_validated": bool(ok),
        "residues": entries,
        "failed_residues": failed,
    }
