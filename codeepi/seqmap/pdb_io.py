from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional


_STANDARD_AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass
class PDBResidue:
    resseq: int
    icode: str        
    aa: str           
    resname: str       

    def label(self) -> str:
        return f"{self.resseq}{self.icode}" if self.icode else str(self.resseq)


def _seqres_from_lines(lines: List[str], chain_id: str) -> str:
    out: List[str] = []
    for ln in lines:
        if not ln.startswith("SEQRES"):
            continue
        cid = ln[11:12].strip()
        if cid != chain_id:
            continue
        parts = ln[19:].split()
        for r in parts:
            out.append(_STANDARD_AA_3TO1.get(r.upper(), "X"))
    return "".join(out)


def _first_model_atom_lines(lines: List[str]) -> List[str]:
    inside = False
    seen_model = False
    keep: List[str] = []
    for ln in lines:
        if ln.startswith("MODEL"):
            if seen_model:
                break
            seen_model = True
            inside = True
            continue
        if ln.startswith("ENDMDL") and inside:
            break
        if seen_model and not inside:
            continue
        keep.append(ln)
    return keep if seen_model else lines


def _atmseq_from_lines(lines: List[str], chain_id: str
                       ) -> Tuple[str, List[PDBResidue]]:
    atmseq: List[str] = []
    residues: List[PDBResidue] = []
    seen_key: Optional[Tuple[str, int, str]] = None
    for ln in lines:
        if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
            continue
        if len(ln) < 27:
            continue
        alt = ln[16]
        if alt not in (" ", "A"):
            continue
        resname = ln[17:20].strip().upper()
        cid = ln[21:22].strip()
        if cid != chain_id:
            continue
        try:
            resseq = int(ln[22:26])
        except ValueError:
            continue
        icode = ln[26:27].strip()
        aa1 = _STANDARD_AA_3TO1.get(resname)
        if aa1 is None:
            continue
        key = (cid, resseq, icode)
        if key == seen_key:
            continue
        seen_key = key
        atmseq.append(aa1)
        residues.append(PDBResidue(resseq=resseq, icode=icode,
                                   aa=aa1, resname=resname))
    return "".join(atmseq), residues


def extract_atmseq(pdb_path: str, chain_id: str
                   ) -> Tuple[str, List[PDBResidue]]:
    with open(pdb_path, "r") as fh:
        lines = fh.readlines()
    lines = _first_model_atom_lines(lines)
    return _atmseq_from_lines(lines, chain_id)


def extract_seqres(pdb_path: str, chain_id: str) -> str:
    with open(pdb_path, "r") as fh:
        lines = fh.readlines()
    return _seqres_from_lines(lines, chain_id)
