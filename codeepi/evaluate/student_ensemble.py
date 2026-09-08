"""5-seed ensemble evaluation for CodeEpi student checkpoints.

This is the ONLY supported public evaluation path for the
``codeepi_ensemble_v1.0`` bundle. Given the five released
``student_seed{1..5}.pt`` checkpoints, it runs each on the val and test
release splits, averages the per-residue probabilities across seeds,
sweeps the macro / micro thresholds **once** on the ensembled val
probability (recording ``thr_val_search_macro`` / ``thr_val_search_micro``
to match the ``_thr_ens_valsearch.json``), and then reports:

  * ``test``:  metrics of the ensemble-averaged test probability at the
    shared threshold. 
  * ``per_seed_test_stats``:  each seed's own test probability evaluated
    at the shared threshold, aggregated to mean±std across seeds. 
  * ``per_seed_independent_eval``:  for each seed, sweep the threshold on
    that seed's OWN val, then evaluate that seed's OWN test at those
    thresholds. 

All blocks are reported in the PDB-ATOM (resolved) coordinate space —
every SEQRES residue with ATOM records in the antigen-only PDB, with
non-surface positions pinned to prob 0 upstream in ``eval_one_dual``.

Example
-------
    python -m codeepi.evaluate.student_ensemble \
        --checkpoints checkpoints_CodeEpi_v1.0/seeds/student_seed1.pt \
                      checkpoints_CodeEpi_v1.0/seeds/student_seed2.pt \
                      checkpoints_CodeEpi_v1.0/seeds/student_seed3.pt \
                      checkpoints_CodeEpi_v1.0/seeds/student_seed4.pt \
                      checkpoints_CodeEpi_v1.0/seeds/student_seed5.pt \
        --paths       configs/paths.yaml \
        --output      results/reproduce/student_5seed_ensemble_metrics.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from ..data.release import AntigenReleaseDataset
from ..student.model import CodeEpiStudent
from ..student.train_student import eval_one_dual
from ..utils.io import ensure_dir, load_yaml
from ..utils.metrics import (
    best_thr_macro,
    best_thr_micro,
    metrics_macro_at,
    metrics_micro_at,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="codeepi.evaluate.student_ensemble",
        description=(
            "Probability-level ensemble across N student checkpoints: "
            "average per-residue probs, search threshold once on val, "
            "then report macro / micro on test."
        ),
    )
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Paths to student_seedN.pt checkpoints (>=2).")
    p.add_argument("--codebook", default=None,
                   help="Path to codebook_4bin.pt "
                        "(default: paths.codebook_checkpoint).")
    p.add_argument("--paths", default=None,
                   help="paths.yaml with dataset_root / codebook_checkpoint. "
                        "Defaults to configs/paths.yaml or paths_template.yaml.")
    p.add_argument("--dataset_root", default=None,
                   help="Override paths.dataset_root (codeepi_pyg_release_v1.0).")
    p.add_argument("--tau", type=float, default=None,
                   help="Softmax temperature. Defaults to the first ckpt's "
                        "tau_final (0.05 for released bundles). All ckpts "
                        "must agree on tau_final; otherwise pass --tau.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--d_proto", type=int, default=None,
                   help="Prototype dim; defaults to first ckpt's D_PROTO or 256.")
    p.add_argument("--output", default=None,
                   help="Where to write the JSON. If omitted, prints to stdout.")
    return p


def _resolve_paths(args: argparse.Namespace) -> Dict[str, Any]:
    if args.paths and Path(args.paths).exists():
        return load_yaml(args.paths)
    repo = Path(__file__).resolve().parents[2]
    for cand in (repo / "configs" / "paths.yaml",
                 repo / "configs" / "paths_template.yaml"):
        if cand.exists():
            return load_yaml(str(cand))
    return {}


def _load_ckpt(path: str) -> Dict[str, Any]:
    """Load a student ckpt. 
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(
            f"Unexpected checkpoint layout in {path}; expected dict."
        )
    has_release_layout = "model" in ckpt and "head" in ckpt
    has_repro_layout = "state_dict" in ckpt and isinstance(
        ckpt["state_dict"], dict
    )
    if not (has_release_layout or has_repro_layout):
        raise ValueError(
            f"Unexpected checkpoint layout in {path}; need either "
            f"('model' + 'head') or 'state_dict'."
        )
    return ckpt


def _state_from_ckpt(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """Return a state_dict ready for ``CodeEpiStudent.load_state_dict``."""
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return dict(ckpt["state_dict"])
    state = {f"backbone.{k}": v for k, v in ckpt["model"].items()}
    state.update({f"head.{k}": v for k, v in ckpt["head"].items()})
    return state


def _metric_block(y: np.ndarray, p: np.ndarray, gidx: np.ndarray,
                  thr_macro: float, thr_micro: float) -> Dict[str, Any]:
    macro = metrics_macro_at(y, p, gidx, thr_macro)
    micro = metrics_micro_at(y, p, thr_micro)
    return {
        "n_graphs": int(macro["n_graphs"]),
        "macro": macro,
        "micro": micro,
    }


def _block_stats_summary(blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a list of per-seed metric blocks to mean/std/per_seed.

    macro: mcc / f1 / bacc / agiou (per-antigen, at thr_macro).
    micro: mcc / f1 / bacc / agiou / auprc / auroc / auroc_01 (pooled, at thr_micro).
    """
    def _stats(vals: List[float]) -> Dict[str, float]:
        arr = np.array(vals, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "per_seed": [float(v) for v in arr],
        }

    return {
        "test_macro_mcc":   _stats([b["macro"]["mcc"]   for b in blocks]),
        "test_macro_f1":    _stats([b["macro"]["f1"]    for b in blocks]),
        "test_macro_bacc":  _stats([b["macro"]["bacc"]  for b in blocks]),
        "test_macro_agiou": _stats([b["macro"]["agiou"] for b in blocks]),
        "test_micro_mcc":     _stats([b["micro"]["mcc"]      for b in blocks]),
        "test_micro_f1":      _stats([b["micro"]["f1"]       for b in blocks]),
        "test_micro_bacc":    _stats([b["micro"]["bacc"]     for b in blocks]),
        "test_micro_agiou":   _stats([b["micro"]["agiou"]    for b in blocks]),
        "test_micro_auprc":   _stats([b["micro"]["auprc"]    for b in blocks]),
        "test_micro_auroc":   _stats([b["micro"]["auroc"]    for b in blocks]),
        "test_micro_auroc01": _stats([b["micro"]["auroc_01"] for b in blocks]),
    }


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    paths = _resolve_paths(args)

    if len(args.checkpoints) < 2:
        raise ValueError("--checkpoints needs at least 2 ckpts for ensemble.")

    dataset_root = args.dataset_root or paths.get("dataset_root")
    codebook_pt = args.codebook or paths.get("codebook_checkpoint")
    if not dataset_root:
        raise ValueError("dataset_root not provided (CLI or paths YAML).")
    if not codebook_pt:
        raise ValueError("codebook not provided (CLI or paths YAML).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = [_load_ckpt(p) for p in args.checkpoints]
    # Released checkpoints store D_PROTO / tau_final at the top level.
    # Repro checkpoints (train_student.py) store them under "hyperparameters".
    # Read from top level first, fall back to nested dict.
    def _hp(ckpt: Dict[str, Any], key: str, default):
        if key in ckpt:
            return ckpt[key]
        return ckpt.get("hyperparameters", {}).get(key, default)

    d_proto = int(args.d_proto or _hp(ckpts[0], "D_PROTO", _hp(ckpts[0], "d_proto", 256)))
    taus = [float(_hp(c, "tau_final", 0.05)) for c in ckpts]
    if args.tau is not None:
        tau = float(args.tau)
    else:
        tau_ref = taus[0]
        if not all(math.isclose(t, tau_ref, rel_tol=0.0, abs_tol=1e-6) for t in taus):
            raise ValueError(
                f"Checkpoints disagree on tau_final: {taus}. "
                f"Pass --tau to force a common value."
            )
        tau = tau_ref

    val_ds = AntigenReleaseDataset(root=str(dataset_root), split="val")
    test_ds = AntigenReleaseDataset(root=str(dataset_root), split="test")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # Per-seed prob tensors, on resolved (PDB-ATOM) coordinates.
    p_val_resolved_seeds: List[np.ndarray] = []
    p_test_resolved_seeds: List[np.ndarray] = []
    y_val_r_ref = y_test_r_ref = None
    g_val_r_ref = g_test_r_ref = None
    seeds_meta: List[Any] = []

    for i, (ckpt_path, ckpt) in enumerate(zip(args.checkpoints, ckpts)):
        model = CodeEpiStudent.from_codebook_file(
            codebook_path=str(codebook_pt),
            d_proto=d_proto,
        ).to(device)
        state = _state_from_ckpt(ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(
                f"missing keys loading {ckpt_path}: {sorted(missing)}"
            )
        if unexpected and i == 0:
            print(f"[codeepi.evaluate.student_ensemble] ignoring unused "
                  f"legacy keys: {sorted(unexpected)}")
        model.eval()

        dual_v = eval_one_dual(model, val_loader, tau=tau, device=device)
        dual_t = eval_one_dual(model, test_loader, tau=tau, device=device)
        y_v_r, p_v_r, g_v_r = dual_v["resolved"]
        y_t_r, p_t_r, g_t_r = dual_t["resolved"]

        if y_val_r_ref is None:
            y_val_r_ref, g_val_r_ref = y_v_r, g_v_r
            y_test_r_ref, g_test_r_ref = y_t_r, g_t_r
        else:
            aligned = (
                np.array_equal(y_v_r, y_val_r_ref)
                and np.array_equal(g_v_r, g_val_r_ref)
                and np.array_equal(y_t_r, y_test_r_ref)
                and np.array_equal(g_t_r, g_test_r_ref)
            )
            if not aligned:
                raise RuntimeError(
                    f"Label / graph-index alignment differs at "
                    f"{ckpt_path}; ensemble requires deterministic split "
                    f"ordering across ckpts."
                )
        p_val_resolved_seeds.append(p_v_r)
        p_test_resolved_seeds.append(p_t_r)
        seeds_meta.append(int(ckpt["seed"]) if "seed" in ckpt else None)
        print(f"[ensemble] loaded {ckpt_path}  seed={seeds_meta[-1]}  "
              f"tau={tau:.3f}  val_res_n={len(y_v_r)}  "
              f"test_res_n={len(y_t_r)}")

    # ------------------------------------------------------------------
    # Threshold selection (user-specified protocol):
    #   average per-seed val probabilities per antigen residue,
    #   sweep thr on that averaged resolved-space val,
    #   apply the same thr to the averaged test prob.
    # ------------------------------------------------------------------
    p_val_ens_resolved = np.mean(np.stack(p_val_resolved_seeds, axis=0), axis=0).astype(np.float32)
    p_test_ens_resolved = np.mean(np.stack(p_test_resolved_seeds, axis=0), axis=0).astype(np.float32)

    thr_macro, val_macro_mcc_at_thr = best_thr_macro(
        y_val_r_ref, p_val_ens_resolved, g_val_r_ref
    )
    thr_micro, _ = best_thr_micro(y_val_r_ref, p_val_ens_resolved)

    test_block_resolved = _metric_block(y_test_r_ref, p_test_ens_resolved, g_test_r_ref,
                                         thr_macro, thr_micro)

    print(f"[ensemble] thr_val_search_macro={thr_macro:.4f} "
          f"thr_val_search_micro={thr_micro:.4f}  "
          f"val_macro_mcc_at_thr={val_macro_mcc_at_thr:.4f}")

    # Per-seed test metrics AT THE ENSEMBLE thr (mean/std across seeds).
    per_seed_resolved = []
    for i in range(len(args.checkpoints)):
        per_seed_resolved.append(
            _metric_block(y_test_r_ref, p_test_resolved_seeds[i], g_test_r_ref,
                          thr_macro, thr_micro)
        )

    per_seed_resolved_summary = _block_stats_summary(per_seed_resolved)

    # ------------------------------------------------------------------
    # Per-seed INDEPENDENT evaluation: each seed picks its own thr on its
    # own resolved val, then evaluates its own test at those thresholds.
    # ------------------------------------------------------------------
    per_seed_indep_resolved: List[Dict[str, Any]] = []
    per_seed_indep_thresholds: List[Dict[str, Any]] = []
    for i in range(len(args.checkpoints)):
        thr_m_r, _ = best_thr_macro(
            y_val_r_ref, p_val_resolved_seeds[i], g_val_r_ref
        )
        thr_mi_r, _ = best_thr_micro(y_val_r_ref, p_val_resolved_seeds[i])
        per_seed_indep_resolved.append(
            _metric_block(y_test_r_ref, p_test_resolved_seeds[i], g_test_r_ref,
                          thr_m_r, thr_mi_r)
        )
        print(f"[per-seed-indep] slot={i+1} seed={seeds_meta[i]}  "
              f"thr_r_macro={thr_m_r:.4f} thr_r_micro={thr_mi_r:.4f}  "
              f"test_r_macro_mcc={per_seed_indep_resolved[-1]['macro']['mcc']:.4f}")
        per_seed_indep_thresholds.append({
            "seed_slot": i + 1,
            "seed": seeds_meta[i],
            "resolved": {
                "thr_val_search_macro": float(thr_m_r),
                "thr_val_search_micro": float(thr_mi_r),
            },
        })

    per_seed_indep_resolved_summary = _block_stats_summary(per_seed_indep_resolved)

    out: Dict[str, Any] = {
        "mode": "ensemble_prob_mean",
        "n_seeds": len(args.checkpoints),
        "seeds": seeds_meta,
        "checkpoints": [str(Path(p).resolve()) for p in args.checkpoints],
        "tau": float(tau),
        # PDB-ATOM (resolved) results — every SEQRES residue with ATOM
        # records in the antigen-only PDB (non-surface positions pinned to
        # prob 0 upstream in eval_one_dual).
        # Threshold nomenclature matches _thr_ens_valsearch.json.
        "thr_val_search_macro": float(thr_macro),
        "thr_val_search_micro": float(thr_micro),
        "thr_macro": float(thr_macro),
        "thr_micro": float(thr_micro),
        "test": test_block_resolved,
        "per_seed_test_stats": per_seed_resolved_summary,
        "per_seed_independent_eval": {
            "protocol": (
                "for reference only - each seed picks its own thr on its "
                "own resolved val, then evaluates its own test."
            ),
            "per_seed_thresholds": per_seed_indep_thresholds,
            "test_resolved": per_seed_indep_resolved_summary,
        },
    }

    print(json.dumps(out, indent=2))

    if args.output:
        outp = Path(args.output)
        ensure_dir(outp.parent)
        with open(outp, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[codeepi.evaluate.student_ensemble] wrote {outp}")


if __name__ == "__main__":
    main()
