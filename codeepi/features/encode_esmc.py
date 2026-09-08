"""Standalone ESM-C encoder (run in the ``codeepi-esmc`` environment).

The prediction pipeline consumes *precomputed* ESM-C tensors. This
module is the place that talks to the ESM-C API because the EvolutionaryScale
 ``esm`` client (Python 3.12+) and ``fair-esm``
(used by ESM-IF in the prediction env, Python 3.10) share the ``esm`` import
namespace and cannot be co-installed.

Run it from the separate ``codeepi-esmc`` environment::

    conda activate codeepi-esmc
    export ESMC_API_KEY="your_key_here"
    python -m codeepi.features.encode_esmc \
        --ag_fasta my_antigen.fasta \
        --antigen_chain A,B \
        --out cache/my_antigen_esmc.pt

    # then, in the codeepi env, feed the printed path back:
    ... --esmc_ag_pt cache/my_antigen_esmc.pt

"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from .fasta import map_records_to_chains, read_fasta


ESMC_DIM = 2560
ESMC_MODEL_DEFAULT = "esmc-6b-2024-12"


def call_api(sequence: str, model: str = ESMC_MODEL_DEFAULT) -> torch.Tensor:
    """Call the ESM-C API and return a ``(len(sequence), 2560)`` embedding.

    Requires the EvolutionaryScale ``esm`` client and an ``ESMC_API_KEY``
    environment variable.
    """
    key = os.environ.get("ESMC_API_KEY")
    if not key:
        raise RuntimeError(
            "ESMC_API_KEY environment variable is not set. "
            "Export it before running the encoder: "
            'export ESMC_API_KEY="your_key_here"')
    try:
        from esm.sdk import client as _esm_client_mod
        from esm.sdk.api import ESMProtein, LogitsConfig
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "codeepi.features.encode_esmc requires the EvolutionaryScale "
            "`esm` client. Install it in the dedicated `codeepi-esmc` "
            "environment (environment-esmc.yml); do NOT install it into the "
            "prediction env, where it would clobber fair-esm / ESM-IF."
        ) from e

    client = _esm_client_mod(model=model, token=key)
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True),
    )
    # embeddings shape: (1, L+2, dim) with BOS/EOS -- strip them
    emb = logits_output.embeddings[0]
    emb = emb[1:-1]  # strip BOS/EOS
    if emb.size(-1) != ESMC_DIM:
        raise RuntimeError(
            f"ESM-C API returned dim {emb.size(-1)}, expected {ESMC_DIM}"
        )
    return emb.float().cpu()


def build_ag_full_seqres(ag_fasta: str | Path, antigen_chain: str) -> str:
    """Reconstruct the exact ``ag_full_seqres`` used by the predictor.

    Identical recipe to ``student_antigen_ensemble``: read the FASTA, map
    records to chains, then concatenate per-chain SEQRES in ``antigen_chain``
    order.
    """
    ag_chains = [c.strip() for c in antigen_chain.split(",") if c.strip()]
    if not ag_chains:
        raise ValueError("--antigen_chain is empty")
    records = read_fasta(ag_fasta)
    ag_seqres = map_records_to_chains(records, ag_chains)
    return "".join(ag_seqres[c] for c in ag_chains)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Encode the full-length antigen SEQRES with the ESM-C "
                    "API and save a (L, 2560) .pt for precomputed prediction."
    )
    p.add_argument("--ag_fasta", required=True,
                   help="Antigen FASTA (one record per antigen chain). Same "
                        "file you pass to prediction as --ag_fasta.")
    p.add_argument("--antigen_chain", required=True,
                   help="Antigen chain ID(s). Multi-chain: comma-separated, "
                        'e.g. "A,B". Order defines the SEQRES concatenation '
                        "order and MUST match prediction.")
    p.add_argument("--out", required=True,
                   help="Output .pt path. Saved as a raw (L, 2560) tensor.")
    p.add_argument("--model", default=ESMC_MODEL_DEFAULT,
                   help=f"ESM-C model name (default {ESMC_MODEL_DEFAULT}).")
    return p


def main(argv=None) -> None:
    args = build_argparser().parse_args(argv)

    ag_full_seqres = build_ag_full_seqres(args.ag_fasta, args.antigen_chain)
    print(f"[encode_esmc] full-length antigen SEQRES: {len(ag_full_seqres)} "
          f"residues across chains {args.antigen_chain}")

    emb = call_api(ag_full_seqres, model=args.model)
    if emb.size(0) != len(ag_full_seqres):
        raise RuntimeError(
            f"ESM-C returned {emb.size(0)} rows but SEQRES length is "
            f"{len(ag_full_seqres)}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, str(out_path))

    print(f"[encode_esmc] saved ESM-C tensor {tuple(emb.shape)} -> {out_path.resolve()}")
    print()
    print("Next: run prediction in the `codeepi` env with this path, e.g.")
    print(f"    --esmc_ag_pt {out_path.resolve()}")


if __name__ == "__main__":
    main()
