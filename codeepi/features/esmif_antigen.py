"""ESM-IF encoding for antigen-only PDBs (student).

Runs ESM-IF-1 on a single-chain (or multi-chain, same antigen) PDB and
returns ``(N_res, 512)`` per-residue representations.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Sequence, Tuple

import torch


ESMIF_DIM = 512

# ESM-IF-1 weights are NOT downloaded automatically. The user must place the
# checkpoint into the torch hub cache beforehand; if it is absent we abort with
# the download command instead of letting fair-esm reach out to the network.
ESMIF_WEIGHT_NAME = "esm_if1_gvp4_t16_142M_UR50.pt"
ESMIF_WEIGHT_URL = (
    "https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt"
)


def _esmif_weight_path() -> "Path":
    """torch hub checkpoint location for the ESM-IF-1 weights.

    Honors ``TORCH_HOME``; otherwise ``~/.cache/torch``. Mirrors
    ``torch.hub.get_dir()`` without importing torch here.
    """
    import os

    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        hub_dir = Path(torch_home).expanduser() / "hub"
    else:
        xdg = os.environ.get("XDG_CACHE_HOME", "~/.cache")
        hub_dir = Path(xdg).expanduser() / "torch" / "hub"
    return hub_dir / "checkpoints" / ESMIF_WEIGHT_NAME


def _require_esmif_weights_local() -> None:
    """Fail loudly if the ESM-IF-1 checkpoint is not already cached.

    We never trigger an automatic download. The user downloads the weights
    once, by hand, into the torch hub cache.
    """
    p = _esmif_weight_path()
    if p.exists():
        return
    raise FileNotFoundError(
        "ESM-IF-1 weights are required but not found in the torch hub cache:\n"
        f"    {p}\n"
        "They are NOT downloaded automatically. Fetch them once by hand:\n"
        f"    mkdir -p {p.parent}\n"
        f"    wget -O {p} \\\n"
        f"        {ESMIF_WEIGHT_URL}\n"
        "(~1.7 GB; no contact-regression file is needed for ESM-IF.)\n"
        "To use a different cache location, set TORCH_HOME before running."
    )


def encode_antigen(
    pdb_path: str | Path,
    ag_chain_ids: Sequence[str],
    device: str = "cpu",
) -> Tuple[torch.Tensor, Dict[str, str]]:
    """Encode a single antigen PDB with ESM-IF-1.

    Returns ``(x, seqs)`` where ``x`` is ``(N_res, 512)`` (chains
    concatenated in ``ag_chain_ids`` order) and ``seqs`` maps each chain ID
    to the per-residue sequence ``load_coords`` returned for it (the exact
    ESM-IF input, used by the AA-identity consistency guard).
    """
    try:
        from esm.inverse_folding.util import load_coords
        from esm.pretrained import esm_if1_gvp4_t16_142M_UR50
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "codeepi.features.esmif_antigen requires fair-esm with ESM-IF weights."
        ) from e

    _require_esmif_weights_local()
    model, _ = esm_if1_gvp4_t16_142M_UR50()
    model = model.to(device).eval()

    parts = []
    seqs: Dict[str, str] = {}
    for cid in ag_chain_ids:
        coords, seq = load_coords(str(pdb_path), cid)
        seqs[cid] = seq
        coords_t = torch.from_numpy(coords).unsqueeze(0).to(device)
        n_res = coords_t.shape[1]
        padding_mask = torch.zeros(1, n_res, dtype=torch.bool, device=device)
        confidence = torch.ones(1, n_res, device=device)
        with torch.no_grad():
            enc = model.encoder(
                coords_t,
                padding_mask,
                confidence,
                return_all_hiddens=False,
            )
            rep = enc["encoder_out"][0].transpose(0, 1).squeeze(0)
        parts.append(rep.cpu())
    x = torch.cat(parts, dim=0) if parts else torch.zeros(0, ESMIF_DIM)
    return x, seqs
