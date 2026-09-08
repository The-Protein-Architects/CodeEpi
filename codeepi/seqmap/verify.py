from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .pdb_io import extract_atmseq
from .anchor import build_anchor_set, anchor_predict


@dataclass
class VerifyEntry:
    k: int
    kind: str               
    atm_pos: int
    stored_seqres_pos: int
    seqres_AA_at_stored: str
    pdb_resseq: Optional[int]
    pdb_icode: str
    atm_AA: str
    anchor_pred: Optional[int]
    anchor_mode: str
    aa_identity: bool        
    status: str              
    detail: str


@dataclass
class VerifyRowResult:
    abdbid: str
    chain: str
    n_total: int
    n_ok: int
    n_fail: int
    entries: List[VerifyEntry]

    def summary(self) -> Dict[str, int]:
        return {"n_total": self.n_total,
                "n_ok": self.n_ok,
                "n_fail": self.n_fail}


def _one_kind(kind: str,
              stored_pos: List[int],
              atm_pos: List[int],
              seqres: str,
              atmseq: str,
              residues,
              anchor: Dict[int, int],
              atms_sorted: List[int],
              allow_terminal_single_side: bool) -> List[VerifyEntry]:
    entries: List[VerifyEntry] = []
    for k, (s, a) in enumerate(zip(stored_pos, atm_pos)):
        if not (0 <= a < len(residues)):
            entries.append(VerifyEntry(
                k=k, kind=kind, atm_pos=a, stored_seqres_pos=s,
                seqres_AA_at_stored="",
                pdb_resseq=None, pdb_icode="", atm_AA="",
                anchor_pred=None, anchor_mode="atm_out_of_range",
                aa_identity=False,
                status="fail_out_of_range",
                detail=f"atm_pos={a} n_residues={len(residues)}"))
            continue
        R = residues[a]
        pred, mode, _, _ = anchor_predict(
            a, residues, anchor, atms_sorted,
            allow_terminal_single_side=allow_terminal_single_side)
        seq_aa = seqres[s] if 0 <= s < len(seqres) else ""
        ok = (seq_aa == R.aa) and seq_aa != ""
        entries.append(VerifyEntry(
            k=k, kind=kind, atm_pos=a, stored_seqres_pos=s,
            seqres_AA_at_stored=seq_aa,
            pdb_resseq=R.resseq, pdb_icode=R.icode, atm_AA=R.aa,
            anchor_pred=pred, anchor_mode=mode,
            aa_identity=ok,
            status="ok" if ok else "fail_aa_mismatch",
            detail=""
                   if ok
                   else f"seqres[{s}]={seq_aa!r} atm[{a}]={R.aa!r} "
                        f"pdb={R.resseq}{R.icode or ''}"))
    return entries


def verify_row(*,
               abdbid: str,
               chain: str,
               seqres: str,
               pdb_path: str,
               surf_seqres: List[int],
               surf_atm: List[int],
               epi_seqres: List[int],
               epi_atm: List[int],
               allow_terminal_single_side: bool = True
               ) -> VerifyRowResult:
    atmseq, residues = extract_atmseq(pdb_path, chain)
    _, _, anchor = build_anchor_set(seqres, atmseq)
    atms_sorted = sorted(anchor)
    entries: List[VerifyEntry] = []
    entries.extend(_one_kind("surf", surf_seqres, surf_atm, seqres, atmseq,
                             residues, anchor, atms_sorted,
                             allow_terminal_single_side))
    entries.extend(_one_kind("epi", epi_seqres, epi_atm, seqres, atmseq,
                             residues, anchor, atms_sorted,
                             allow_terminal_single_side))
    n_ok = sum(1 for e in entries if e.status == "ok")
    return VerifyRowResult(abdbid=abdbid, chain=chain,
                           n_total=len(entries), n_ok=n_ok,
                           n_fail=len(entries) - n_ok,
                           entries=entries)


def _parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    return [int(x) for x in s.split(",")] if s else []


def verify_csv(csv_path: str,
               structures_dir: str,
               *,
               abdbid_col: str = "abdbid",
               chain_col: str = "ag_chain_id",
               seqres_col: str = "ag_seqres",
               surf_seqres_col: str = "ag_surf_seqres_positions",
               surf_atm_col: str = "ag_surf_atm_positions",
               epi_seqres_col: str = "ag_epi_seqres_positions",
               epi_atm_col: str = "ag_epi_atm_positions",
               allow_terminal_single_side: bool = True,
               progress: bool = False,
               ) -> Tuple[Dict[str, int], List[dict]]:
    import csv
    from pathlib import Path
    struct_dir = Path(structures_dir)
    totals = {"rows": 0, "row_load_error": 0,
              "surf_total": 0, "surf_ok": 0, "surf_fail": 0,
              "epi_total": 0,  "epi_ok": 0,  "epi_fail": 0}
    failed: List[dict] = []
    with open(csv_path, "r", newline="") as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
    it = rows
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(rows)
        except ImportError:
            pass
    for row in it:
        totals["rows"] += 1
        abdbid = row[abdbid_col]; chain = row[chain_col]
        pdb_path = struct_dir / f"{abdbid}.pdb"
        try:
            res = verify_row(
                abdbid=abdbid, chain=chain,
                seqres=row[seqres_col], pdb_path=str(pdb_path),
                surf_seqres=_parse_int_list(row[surf_seqres_col]),
                surf_atm=_parse_int_list(row[surf_atm_col]),
                epi_seqres=_parse_int_list(row[epi_seqres_col]),
                epi_atm=_parse_int_list(row[epi_atm_col]),
                allow_terminal_single_side=allow_terminal_single_side)
        except Exception as e:
            totals["row_load_error"] += 1
            failed.append({"abdbid": abdbid, "chain": chain,
                           "status": "load_error",
                           "detail": f"{type(e).__name__}: {e}"})
            continue
        for e in res.entries:
            if e.kind == "surf":
                totals["surf_total"] += 1
                if e.aa_identity: totals["surf_ok"] += 1
                else:
                    totals["surf_fail"] += 1
                    failed.append({"abdbid": abdbid, "chain": chain, **asdict(e)})
            else:
                totals["epi_total"] += 1
                if e.aa_identity: totals["epi_ok"] += 1
                else:
                    totals["epi_fail"] += 1
                    failed.append({"abdbid": abdbid, "chain": chain, **asdict(e)})
    return totals, failed
