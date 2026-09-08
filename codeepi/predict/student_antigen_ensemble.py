"""CodeEpi student 5-seed ensemble prediction (antigen-only).

Runs the 5 released student checkpoints
(``checkpoints_CodeEpi_v1.0/seeds/student_seed{1..5}.pt``) on a single antigen PDB and
reports the ensembled per-residue epitope probability
``p = mean_i p_seed_i``.

Two input modes (both supported):

* Single antigen: ``--pdb`` + ``--ag_fasta`` + ``--antigen_chain``. Writes
  the file layout below directly under ``--output``.
* Batch: ``--manifest <csv>`` (mutually exclusive with ``--pdb``). The 5
  checkpoints and their models are loaded ONCE and reused across every row;
  each row writes the same file layout under ``<output>/<name>/``. A single
  row failing is logged and skipped, not fatal to the batch.

Inputs: an antigen FASTA (SEQRES authority for ESM-C) + PDB + chain IDs.
ESM-C is computed on the full-length FASTA SEQRES; ESM-IF on the residues with
structures. Structure residues are mapped back to
SEQRES positions with the anchor aligner
(``codeepi.features.seqmap_align``) to crop ESM-C so it lines up
row-for-row with the ESM-IF surface slice.

The student was trained on the release ``Antigen1321.edge_index_egnn`` graph:
an 8 Å Cα radius graph over antigen surface residues with
``max_num_neighbors=64`` and 32-D edge features (16-D RBF distance + 16-D
sinusoidal positional encoding). 

Readme on `` /CodeEpi/docs/prediction.md ``

Output layout matches the single-seed CLI:

    <output>/residue_predictions.csv
    <output>/residue_predictions.json
    <output>/antigen_colored_by_probability.pdb
    <output>/config_used.yaml
    <output>/per_seed_probabilities.csv   (per-seed columns, for auditing)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Data

from ..features.esmc import load_precomputed
from ..features.esmif_antigen import encode_antigen
from ..features.fasta import map_records_to_chains, read_fasta
from ..features.pdb_parser import parse_pdb
from ..features.rsa import compute_rsa
from ..features.seqmap_align import (
    assert_aa_identity,
    build_chain_align,
    build_seqmap_report,
)
from ..features.surface import DEFAULT_RSA_THRESHOLD, compute_surface
from ..student.model import CodeEpiStudent
from ..utils.graph import build_egnn_edges
from ..utils.io import ensure_dir
from .postprocess import (
    dump_config_used,
    load_codebook_bin_edges,
    rank_and_bin,
    write_pdb_with_probability,
    write_residue_predictions_csv,
    write_residue_predictions_json,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="codeepi.predict.student_antigen_ensemble",
        description=(
            "CodeEpi student 5-seed ensemble antigen-only epitope prediction. "
        ),
    )
    p.add_argument("--pdb", default=None,
                   help="Input antigen-only PDB file. Single-antigen mode. "
                        "Mutually exclusive with --manifest.")
    p.add_argument("--ag_fasta", default=None,
                   help="Antigen FASTA (one record per antigen chain; the "
                        "SEQRES authority ESM-C is computed on). ESM-C runs on "
                        "the full-length FASTA sequence; surface residues from "
                        "the structure are mapped back to SEQRES (anchor) to "
                        "crop it, matching the training contract. "
                        "Required in single-antigen mode.")
    p.add_argument("--antigen_chain", default=None,
                   help="Antigen chain ID (comma-separated for multi-chain). "
                        "Required in single-antigen mode.")
    p.add_argument("--manifest", default=None,
                   help="CSV manifest for BATCH prediction (mutually exclusive "
                        "with --pdb). "
                        "one row failing does not abort the batch.")
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Paths to the 5 released student_seed{1..5}.pt files.")
    p.add_argument("--codebook", required=True,
                   help="Path to codebook_4bin.pt (from codeepi_ensemble_v1.0).")
    p.add_argument("--output", required=True,
                   help="Output directory. Single-antigen mode: the run "
                        "directory itself. Batch mode: the root under which "
                        "each row writes a <name>/ subdirectory.")
    p.add_argument("--seqmap_override", default=None,
                   help="Path to a hand-edited seqmap JSON (same schema as the "
                        "seqmap.json the CLI writes). When given, the "
                        "'seqres_concat_position' of each residue REPLACES the "
                        "automatic anchor mapping. Residues are matched by "
                        "(chain_id, atm_local_index). The overridden mapping "
                        "still has to pass the AA-identity / in-range / "
                        "injective / monotonic checks; the run aborts if any "
                        "fail. An audit copy is written to "
                        "seqmap.after_override.json.")
    p.add_argument("--esmc_ag_pt", default=None,
                   help="REQUIRED (single-antigen mode). Precomputed ESM-C "
                        ".pt of shape (L, 2560) for the FULL antigen SEQRES. "
                        "Produce it with the standalone encoder in the "
                        "`codeepi-esmc` env: "
                        "`python -m codeepi.features.encode_esmc --ag_fasta "
                        "... --antigen_chain ... --out X.pt`, then pass X.pt "
                        "here. In batch mode use the manifest's esmc_ag_pt "
                        "column instead.")
    p.add_argument("--rsa_threshold", type=float, default=DEFAULT_RSA_THRESHOLD)
    p.add_argument("--radius", type=float, default=8.0,
                   help="Ca radius (A) for the EGNN graph. Default 8.0 A "
                        "matches the release edge_index_egnn.")
    p.add_argument("--max_neighbors", type=int, default=64,
                   help="Max Ca neighbors per node. Default 64 matches the "
                        "release edge_index_egnn.")
    p.add_argument("--tau", type=float, default=0.05,
                   help="Softmax temperature for the prototype head.")
    p.add_argument("--d_proto", type=int, default=None,
                   help="Prototype dim; defaults to ckpt hyperparameters or 256.")
    p.add_argument("--threshold", type=float, default=0.45,
                   help="Epitope probability threshold (default 0.45). "
                        "Residues with probability >= threshold are marked "
                        "as epitope in the output CSV/JSON.")
    return p


def _resolve_d_proto(ckpt, cli_override):
    if cli_override is not None:
        return int(cli_override)
    hp = ckpt.get("hyperparameters", {}) if isinstance(ckpt, dict) else {}
    return int(hp.get("d_proto", ckpt.get("D_PROTO", 256)))


def _load_state(ckpt):
    """Accept both released ``model``+``head`` layout and repro ``state_dict``."""
    if not isinstance(ckpt, dict):
        raise ValueError("student checkpoint must be a dict")
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return dict(ckpt["state_dict"])
    if "model" in ckpt and "head" in ckpt:
        state = {f"backbone.{k}": v for k, v in ckpt["model"].items()}
        state.update({f"head.{k}": v for k, v in ckpt["head"].items()})
        return state
    raise ValueError(
        "student checkpoint layout not recognized (expected 'state_dict' or "
        "'model'+'head')"
    )


def _report_failed_and_abort(surf_report, seqmap_path, override_applied):
    """Print the failed residues and abort with a fix-it hint.

    Never predicts on a mis-aligned graph — this is the hard gate. The
    ``failed_residues`` list is already in the written JSON; here we surface
    the first few on stderr so the user does not have to open the file to
    see what to fix.
    """
    failed = surf_report.get("failed_residues", [])
    lines = [
        f"seqmap validation FAILED: {len(failed)}/"
        f"{surf_report['n_selected']} residue(s) did not pass the "
        f"AA-identity / in-range / injective / monotonic checks.",
        f"Full per-residue detail: {seqmap_path}",
        "Failing residues (first 20):",
    ]
    for e in failed[:20]:
        lines.append(
            f"  chain {e['chain_id']} pdb {e['pdb_residue_label']} "
            f"(aa={e['aa']}, atm_local={e['atm_local_index']}) -> "
            f"seqres {e['seqres_concat_position']} "
            f"[{e['anchor_mode']}]: {e['check_detail']}"
        )
    if len(failed) > 20:
        lines.append(f"  ... and {len(failed) - 20} more (see JSON).")
    if override_applied:
        lines += [
            "",
            "This was an OVERRIDE run: the positions you supplied still do "
            "not line up. Fix the 'seqres_concat_position' of the residues "
            "above in your override JSON and re-run.",
        ]
    else:
        lines += [
            "",
            "To fix: either correct --ag_fasta / --antigen_chain so the "
            "automatic alignment succeeds, OR copy this seqmap.json, edit the "
            "'seqres_concat_position' of each failing residue by hand, and "
            "re-run with --seqmap_override <edited.json>.",
        ]
    raise RuntimeError("\n".join(lines))


def _load_seqmap_override(path, ag_chains, ag_aligns, per_chain_surf_atm):
    """Read a hand-edited seqmap JSON and return per-chain lists of
    CHAIN-LOCAL SEQRES indices, aligned to ``per_chain_surf_atm``.

    The override is matched to the selected surface residues by
    ``(chain_id, atm_local_index)`` so row order / typos cannot silently
    shift the mapping. Every selected residue must be present exactly once;
    missing or extra residues are a hard error. Concatenated SEQRES
    positions in the file are converted back to chain-local indices using
    each chain's SEQRES offset (chains in ``ag_chains`` order).
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    surf = data.get("ag_surf", data)
    residues = surf.get("residues", [])
    by_key = {}
    for r in residues:
        key = (str(r.get("chain_id")), int(r.get("atm_local_index")))
        if key in by_key:
            raise ValueError(
                f"--seqmap_override: duplicate residue entry for chain "
                f"{key[0]!r} atm_local_index {key[1]}"
            )
        by_key[key] = r.get("seqres_concat_position")

    seqres_off = {}
    off = 0
    for cid, al in zip(ag_chains, ag_aligns):
        seqres_off[cid] = off
        off += al.n_seqres

    per_chain_local = []
    for cid, atm_locals in zip(ag_chains, per_chain_surf_atm):
        locals_for_chain = []
        for a in atm_locals:
            key = (cid, int(a))
            if key not in by_key:
                raise ValueError(
                    f"--seqmap_override: missing entry for chain {cid!r} "
                    f"atm_local_index {a}; the override must cover every "
                    f"selected surface residue."
                )
            concat = by_key.pop(key)
            if concat is None:
                locals_for_chain.append(None)
            else:
                locals_for_chain.append(int(concat) - seqres_off[cid])
        per_chain_local.append(locals_for_chain)
    if by_key:
        extra = sorted(by_key.keys())
        raise ValueError(
            f"--seqmap_override: {len(extra)} entr(y/ies) do not match any "
            f"selected surface residue, e.g. {extra[:5]}"
        )
    return per_chain_local


def _load_models(checkpoints, codebook, d_proto_override, device):
    """Load every seed checkpoint ONCE and return (models, seeds_meta).

    In batch mode this is called a single time and the returned models are
    reused across every antigen, so the per-row cost is features + forward
    only (no repeated torch.load / state_dict loading).
    """
    models = []
    seeds_meta: List[Dict] = []
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        d_proto = _resolve_d_proto(ckpt, d_proto_override)
        model = CodeEpiStudent.from_codebook_file(
            codebook_path=codebook, d_proto=d_proto,
        ).to(device)
        state = _load_state(ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(
                f"missing keys loading {ckpt_path}: {sorted(missing)}"
            )
        model.eval()
        models.append(model)
        seeds_meta.append({
            "checkpoint": str(Path(ckpt_path).resolve()),
            "seed": int(ckpt["seed"]) if isinstance(ckpt, dict) and "seed" in ckpt else None,
            "tau": float(ckpt.get("tau_final")) if isinstance(ckpt, dict) and "tau_final" in ckpt else None,
        })
        print(f"[codeepi.predict.student_ensemble] loaded {ckpt_path}  "
              f"seed={seeds_meta[-1]['seed']}")
    return models, seeds_meta


def _read_manifest(path):
    """Parse the batch manifest CSV into a list of per-antigen job dicts.

    Required columns: pdb, ag_fasta, antigen_chain. Optional: name,
    esmc_ag_pt, seqmap_override. Relative paths are resolved against the
    manifest file's own directory so a manifest is portable. Blank optional
    cells become None.
    """
    manifest_path = Path(path).resolve()
    base = manifest_path.parent

    def _resolve(val):
        val = (val or "").strip()
        if not val:
            return None
        pp = Path(val)
        return str(pp if pp.is_absolute() else (base / pp))

    jobs = []
    with manifest_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"pdb", "ag_fasta", "antigen_chain", "esmc_ag_pt"}
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            raise ValueError(
                f"--manifest {path}: missing required column(s) "
                f"{sorted(missing_cols)}; found {reader.fieldnames}."
            )
        for lineno, row in enumerate(reader, start=2): 
            pdb = _resolve(row.get("pdb"))
            ag_fasta = _resolve(row.get("ag_fasta"))
            chain = (row.get("antigen_chain") or "").strip()
            esmc_ag_pt = _resolve(row.get("esmc_ag_pt"))
            if not (pdb and ag_fasta and chain and esmc_ag_pt):
                raise ValueError(
                    f"--manifest {path} line {lineno}: pdb, ag_fasta, "
                    f"antigen_chain and esmc_ag_pt are all required (got "
                    f"pdb={pdb!r}, ag_fasta={ag_fasta!r}, "
                    f"antigen_chain={chain!r}, esmc_ag_pt={esmc_ag_pt!r}). "
                    f"Encode each antigen's ESM-C .pt first with "
                    f"`python -m codeepi.features.encode_esmc` in the "
                    f"codeepi-esmc env."
                )
            name = (row.get("name") or "").strip() or Path(pdb).stem
            jobs.append({
                "name": name,
                "pdb": pdb,
                "ag_fasta": ag_fasta,
                "antigen_chain": chain,
                "esmc_ag_pt": esmc_ag_pt,
                "seqmap_override": _resolve(row.get("seqmap_override")),
                "_lineno": lineno,
            })
    if not jobs:
        raise ValueError(f"--manifest {path}: no data rows found.")
    names = [j["name"] for j in jobs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(
            f"--manifest {path}: duplicate output name(s) {dup}; each row "
            f"must resolve to a unique 'name' (or unique PDB stem)."
        )
    return jobs


def _predict_one(args, models, seeds_meta, device, *, pdb, ag_fasta,
                 antigen_chain, output, esmc_ag_pt, seqmap_override):
    """Run the 5-seed ensemble on ONE antigen and write its outputs.

    Pre-loaded ``models`` / ``seeds_meta`` are shared across antigens; only
    the per-antigen inputs (pdb / ag_fasta / antigen_chain / output /
    esmc_ag_pt / seqmap_override) vary. All other knobs (rsa_threshold,
    radius, max_neighbors, tau, codebook) come from ``args``.
    """
    ag_chains = [c.strip() for c in antigen_chain.split(",") if c.strip()]
    if not ag_chains:
        raise ValueError("antigen_chain: at least one chain ID required.")

    out_dir = ensure_dir(output)

    ag_seqres = map_records_to_chains(read_fasta(ag_fasta), ag_chains)

    parsed = parse_pdb(pdb, chain_ids=ag_chains)
    seq = "".join(parsed["seq"])
    n_res = len(seq)
    coords = parsed["backbone"]
    resnum = parsed["resnum"]
    chain_of = parsed["chain_of"]
    chain_of_arr = np.asarray(chain_of)

    print(f"[codeepi.predict.student_ensemble] encoding {n_res} residues "
          f"from {pdb}")
    x_esmif, esmif_seqs = encode_antigen(pdb, ag_chains, device="cpu")
    if x_esmif.size(0) != n_res:
        raise RuntimeError(
            f"ESM-IF returned {x_esmif.size(0)} rows but PDB has {n_res} residues"
        )

    ag_aligns = []
    for cid in ag_chains:
        al = build_chain_align(pdb, cid, ag_seqres[cid])
        parse_seq = "".join(
            seq[i] for i in np.where(chain_of_arr == cid)[0]
        )
        assert_aa_identity(cid, al.atmseq, parse_seq, esmif_seqs.get(cid))
        ag_aligns.append(al)

    ag_full_seqres = "".join(ag_seqres[c] for c in ag_chains)
    if not esmc_ag_pt:
        raise ValueError(
            "A precomputed ESM-C tensor is REQUIRED but none was given "
            "(--esmc_ag_pt in single mode, or the manifest's esmc_ag_pt "
            "column in batch mode).\n"
            f"Encode the full-length antigen SEQRES ({len(ag_full_seqres)} "
            "residues) in the `codeepi-esmc` environment, e.g.:\n"
            f"    export ESMC_API_KEY=\"your_key_here\"\n"
            f"    python -m codeepi.features.encode_esmc "
            f"--ag_fasta {ag_fasta} --antigen_chain {antigen_chain} "
            f"--out my_antigen_esmc.pt\n"
            "then pass the printed .pt path back here via --esmc_ag_pt.")
    x_esmc_full = load_precomputed(esmc_ag_pt, expect_n=len(ag_full_seqres))
    if x_esmc_full.size(0) != len(ag_full_seqres):
        raise RuntimeError(
            f"ESM-C row count {x_esmc_full.size(0)} != full SEQRES length "
            f"{len(ag_full_seqres)}."
        )

    rsa = compute_rsa(pdb, chain_of=chain_of, resnum=resnum, seq=parsed["seq"])
    is_surface = compute_surface(rsa, threshold=args.rsa_threshold)

    surf_mask = np.asarray(is_surface, dtype=bool)
    surf_idx_np = np.where(surf_mask)[0].astype(np.int64)
    n_surf = int(surf_idx_np.size)
    if n_surf == 0:
        raise RuntimeError(
            f"No surface residues at rsa_threshold={args.rsa_threshold}; "
            f"lower --rsa_threshold or check that mkdssp ran successfully."
        )
    print(f"[codeepi.predict.student_ensemble] surface residues: "
          f"{n_surf}/{n_res} (rsa_threshold={args.rsa_threshold})")

    ag_bounds = np.cumsum([0] + [al.n_atm for al in ag_aligns])
    per_chain_surf_atm = [[] for _ in ag_chains]
    for s in surf_idx_np.tolist():
        c = int(np.searchsorted(ag_bounds, s, side="right") - 1)
        per_chain_surf_atm[c].append(int(s - ag_bounds[c]))

    override_local = None
    if seqmap_override:
        override_local = _load_seqmap_override(
            seqmap_override, ag_chains, ag_aligns, per_chain_surf_atm
        )
        print(f"[codeepi.predict.student_ensemble] applying seqmap override "
              f"from {seqmap_override}")

    surf_report = build_seqmap_report(
        ag_aligns, per_chain_surf_atm, "ag_surf",
        seqres_locals_override=override_local,
    )
    seqmap = {
        "pdb": str(Path(pdb).resolve()),
        "ag_chains": ag_chains,
        "override_applied": bool(seqmap_override),
        "override_source": (str(Path(seqmap_override).resolve())
                            if seqmap_override else None),
        "ag_surf": surf_report,
        "success": bool(surf_report["all_validated"]),
    }
    seqmap_name = ("seqmap.after_override.json" if seqmap_override
                   else "seqmap.json")
    with (out_dir / seqmap_name).open("w", encoding="utf-8") as fh:
        json.dump(seqmap, fh, indent=2)
    print(f"[codeepi.predict.student_ensemble] wrote {seqmap_name} "
          f"(success={seqmap['success']}, "
          f"n_failed={surf_report['n_failed']}/{surf_report['n_selected']}) "
          f"to {out_dir}")
    if not seqmap["success"]:
        _report_failed_and_abort(surf_report, out_dir / seqmap_name,
                                 bool(seqmap_override))

    surf_seqres = [e["seqres_concat_position"] for e in surf_report["residues"]]
    surf_seqres_np = np.asarray(surf_seqres, dtype=np.int64)

    x_esmif_s = x_esmif[torch.from_numpy(surf_idx_np).long()]
    x_esmc_s = x_esmc_full[torch.from_numpy(surf_seqres_np).long()]
    if x_esmc_s.size(0) != x_esmif_s.size(0):
        raise RuntimeError(
            f"ESM-C surface rows {x_esmc_s.size(0)} != ESM-IF surface rows "
            f"{x_esmif_s.size(0)}; anchor SEQRES crop and structure surface "
            f"disagree."
        )
    coords_s = coords[surf_idx_np]
    rsa_s = rsa[surf_idx_np]
    ca_s = torch.from_numpy(coords_s[:, 1, :]).float()
    edge_index, edge_attr = build_egnn_edges(
        ca_coords=ca_s,
        r=float(args.radius),
        max_num_neighbors=int(args.max_neighbors),
    )
    if edge_index.numel() == 0:
        raise RuntimeError(
            "radius_graph returned no edges on the surface subgraph; "
            "increase --radius or lower --rsa_threshold."
        )
    pos_s = torch.from_numpy(coords_s).float()
    data = Data(
        x_esmif=x_esmif_s.float(),
        x_esmc=x_esmc_s.float(),
        pos=pos_s,
        rsa=torch.from_numpy(rsa_s).float(),
        edge_index=edge_index.long(),
        edge_attr=edge_attr,
        num_nodes=n_surf,
    ).to(device)

    per_seed_probs: List[np.ndarray] = []
    for model, meta in zip(models, seeds_meta):
        with torch.no_grad():
            out = model(data, tau=float(args.tau))
            p = out["p_pred"].detach().cpu().numpy().astype(np.float32)
        per_seed_probs.append(p)
        print(f"[codeepi.predict.student_ensemble] seed={meta['seed']}  "
              f"mean_p={p.mean():.4f}")

    p_surf_ens = np.mean(np.stack(per_seed_probs, axis=0), axis=0).astype(np.float32)
    prob = np.zeros(n_res, dtype=np.float32)
    prob[surf_idx_np] = p_surf_ens

    rb = rank_and_bin(prob, edges=load_codebook_bin_edges(args.codebook))
    is_epitope = (prob >= args.threshold).tolist()
    records: List[Dict] = []
    for i in range(n_res):
        records.append({
            "residue_index": i,
            "pdb_residue_number": resnum[i],
            "chain_id": chain_of[i],
            "aa": parsed["seq"][i],
            "probability": float(prob[i]),
            "is_surface": bool(is_surface[i]),
            "rank": int(rb["rank"][i]),
            "probability_bin": int(rb["probability_bin"][i]),
            "is_epitope": bool(is_epitope[i]),
        })
    write_residue_predictions_csv(
        out_dir / "residue_predictions.csv",
        residue_index=list(range(n_res)),
        pdb_residue_number=resnum,
        chain_id=chain_of,
        aa=parsed["seq"],
        probability=prob,
        is_surface=is_surface,
        rank=rb["rank"],
        probability_bin=rb["probability_bin"],
        is_epitope=is_epitope,
    )
    write_residue_predictions_json(out_dir / "residue_predictions.json", records)
    prob_of = {(chain_of[i], resnum[i]): float(prob[i]) for i in range(n_res)}
    write_pdb_with_probability(
        in_pdb=Path(pdb),
        out_pdb=out_dir / "antigen_colored_by_probability.pdb",
        prob_of=prob_of,
    )

    per_seed_full = []
    for p_seed in per_seed_probs:
        pf = np.zeros(n_res, dtype=np.float32)
        pf[surf_idx_np] = p_seed
        per_seed_full.append(pf)
    with (out_dir / "per_seed_probabilities.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        header = ["residue_index", "pdb_residue_number", "chain_id", "aa",
                  "is_surface", "probability_mean"]
        header += [f"probability_seed{i+1}" for i in range(len(per_seed_full))]
        w.writerow(header)
        for i in range(n_res):
            row = [i, resnum[i], chain_of[i], parsed["seq"][i],
                   bool(is_surface[i]), float(prob[i])]
            row += [float(pf[i]) for pf in per_seed_full]
            w.writerow(row)

    cfg = {**vars(args), "seeds": seeds_meta}
    cfg.update({
        "pdb": str(pdb), "ag_fasta": str(ag_fasta),
        "antigen_chain": antigen_chain, "output": str(output),
        "esmc_ag_pt": (str(esmc_ag_pt) if esmc_ag_pt else None),
        "seqmap_override": (str(seqmap_override) if seqmap_override else None),
    })
    dump_config_used(out_dir / "config_used.yaml", cfg)
    print(f"[codeepi.predict.student_ensemble] wrote outputs to {out_dir}")
    return {"name": Path(output).name, "output": str(out_dir),
            "n_res": n_res, "n_surf": n_surf}


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    if len(args.checkpoints) < 2:
        raise ValueError(
            "--checkpoints needs at least 2 ckpts (5 for the release ensemble)."
        )

    if bool(args.manifest) == bool(args.pdb):
        raise ValueError(
            "provide exactly one of --pdb (single antigen) or --manifest "
            "(batch); got "
            + ("both" if args.pdb else "neither") + "."
        )

    jobs = None
    if not args.manifest:
        if not (args.ag_fasta and args.antigen_chain):
            raise ValueError(
                "single-antigen mode (--pdb) requires --ag_fasta and "
                "--antigen_chain."
            )
    else:
        for k in ("pdb", "ag_fasta", "antigen_chain", "esmc_ag_pt",
                  "seqmap_override"):
            if getattr(args, k):
                raise ValueError(
                    f"--{k} is a per-row manifest column in batch mode; do "
                    f"not also pass it on the command line."
                )
        jobs = _read_manifest(args.manifest)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, seeds_meta = _load_models(
        args.checkpoints, args.codebook, args.d_proto, device
    )

    if not args.manifest:
        _predict_one(
            args, models, seeds_meta, device,
            pdb=args.pdb, ag_fasta=args.ag_fasta,
            antigen_chain=args.antigen_chain, output=args.output,
            esmc_ag_pt=args.esmc_ag_pt, seqmap_override=args.seqmap_override,
        )
        return

    root = ensure_dir(args.output)
    print(f"[codeepi.predict.student_ensemble] BATCH: {len(jobs)} antigen(s) "
          f"from {args.manifest}; outputs under {root}")

    summary: List[Dict] = []
    n_ok = 0
    for j in jobs:
        print(f"\n[codeepi.predict.student_ensemble] === row "
              f"{j['_lineno'] - 1}/{len(jobs)}: {j['name']} ===")
        row_out = str(root / j["name"])
        try:
            res = _predict_one(
                args, models, seeds_meta, device,
                pdb=j["pdb"], ag_fasta=j["ag_fasta"],
                antigen_chain=j["antigen_chain"], output=row_out,
                esmc_ag_pt=j["esmc_ag_pt"],
                seqmap_override=j["seqmap_override"],
            )
            summary.append({"name": j["name"], "status": "ok", **res})
            n_ok += 1
        except Exception as exc:  
            print(f"[codeepi.predict.student_ensemble] FAILED {j['name']}: "
                  f"{type(exc).__name__}: {exc}")
            summary.append({
                "name": j["name"], "status": "failed",
                "output": row_out, "error": f"{type(exc).__name__}: {exc}",
            })

    summary_path = root / "batch_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump({"manifest": str(Path(args.manifest).resolve()),
                   "n_total": len(jobs), "n_ok": n_ok,
                   "n_failed": len(jobs) - n_ok, "rows": summary},
                  fh, indent=2)
    print(f"\n[codeepi.predict.student_ensemble] BATCH done: {n_ok}/{len(jobs)} "
          f"ok; summary -> {summary_path}")
    if n_ok == 0:
        raise RuntimeError("batch: every antigen failed; see batch_summary.json")


if __name__ == "__main__":
    main()
