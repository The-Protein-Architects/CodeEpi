"""Minimal FASTA reader for the prediction CLIs.

The prediction contract makes the FASTA the *authority* for the SEQRES
(full-length) sequence that ESM-C is computed on, while the PDB provides
the structure (ESM-IF, DSSP surface, heavy-atom contacts). The anchor
alignment (``codeepi.seqmap``) reconciles the two spaces.

Antibody FASTA is expected to carry the heavy and light chains as two
separate records (``H_and_L_encoded_separately_then_concatenated_HL``
ESM-C contract). Antigen FASTA carries one record per antigen chain.

Standard library only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path: str | Path) -> "OrderedRecords":
    """Parse ``path`` and return records in file order.

    Returns a list of ``(header_id, sequence)`` tuples where ``header_id``
    is the first whitespace-delimited token after ``>`` and ``sequence`` is
    the uppercased residue string (no gaps, no whitespace).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"FASTA not found: {p}")
    records: List[Tuple[str, str]] = []
    header: str | None = None
    buf: List[str] = []
    with p.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(buf)))
                header = line[1:].strip().split()[0] if line[1:].strip() else ""
                buf = []
            else:
                buf.append(line.strip().upper())
        if header is not None:
            records.append((header, "".join(buf)))
    if not records:
        raise ValueError(f"no FASTA records found in {p}")
    for hid, seq in records:
        if not seq:
            raise ValueError(f"FASTA record {hid!r} in {p} has empty sequence")
        bad = set(seq) - _AA
        if bad:
            raise ValueError(
                f"FASTA record {hid!r} in {p} contains non-standard residue "
                f"symbols {sorted(bad)}; only the 20 standard amino acids are "
                f"allowed (the SEQRES fed to ESM-C must be canonical)."
            )
    return records


OrderedRecords = List[Tuple[str, str]]


def map_records_to_chains(
    records: OrderedRecords,
    chain_ids: Sequence[str],
) -> Dict[str, str]:
    """Assign one FASTA record to each chain ID, in order.

    Matching rule (deterministic, fail-loud):

    1. If every chain ID appears (exactly once) as a record header token,
       match by header (order-independent, robust to record reordering).
    2. Otherwise, if the record count equals the chain count, match
       positionally in the given ``chain_ids`` order.
    3. Otherwise raise — we never guess.

    Returns ``{chain_id: sequence}``.
    """
    chain_ids = list(chain_ids)
    headers = [h for h, _ in records]

    # rule 1: header match
    header_set = set(headers)
    if len(header_set) == len(headers) and set(chain_ids).issubset(header_set):
        by_header = {h: s for h, s in records}
        return {cid: by_header[cid] for cid in chain_ids}

    # rule 2: positional match on equal counts
    if len(records) == len(chain_ids):
        return {cid: records[i][1] for i, cid in enumerate(chain_ids)}

    raise ValueError(
        f"cannot map {len(records)} FASTA record(s) {headers} to chains "
        f"{chain_ids}: headers do not cover all chain IDs and record count "
        f"!= chain count. Rename FASTA headers to the chain IDs, or provide "
        f"exactly one record per chain in chain order."
    )
