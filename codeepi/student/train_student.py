"""Training entry point for the CodeEpi full student model.

Training recipe:

  * 3-layer EGNN backbone + prototype head with a fixed 4-bin codebook
  * asymmetric focal loss on probability + L2 anchor regularizer
  * cosine ``tau`` schedule, AdamW + cosine LR, parameter-space EMA
  * per-antigen macro-MCC threshold sweep on val for early stopping

Data comes from ``codeepi.data.release.AntigenReleaseDataset``, which
consumes the ``Antigen1321/`` PyG dataset under the CodeEpi PyG Release v1.0.

Example
-------
    python -m codeepi.student.train_student \
        --config configs/train/student.yaml \
        --paths configs/paths.yaml \
        --seed_index 1

The CLI can also override individual paths / hyperparameters without editing
the YAML.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from ..data.release import AntigenReleaseDataset
from ..utils.io import load_yaml, ensure_dir
from ..utils.metrics import (
    best_thr_macro,
    best_thr_micro,
    metrics_macro_at,
    metrics_micro_at,
    scatter_batch_surface_to_resolved,
)
from ..utils.seed import seed_everything
from .losses import EMA, afl_p, cosine_tau
from .model import CodeEpiStudent


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="codeepi.student.train_student",
        description="Train the full CodeEpi student model (codebook + anchor).",
    )
    p.add_argument("--config", type=str, required=True,
                   help="YAML file with train / model / eval settings.")
    p.add_argument("--paths", type=str, required=True,
                   help="YAML file with dataset_root and checkpoint paths.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override the training seed. If unset, uses --seed_index "
                        "to pick from the config's seeds list.")
    p.add_argument("--seed_index", type=int, default=None,
                   help="1-based index into config.seeds; ignored if --seed is set.")
    p.add_argument("--codebook", type=str, default=None,
                   help="Override paths.codebook_checkpoint (path to the "
                        "teacher-derived 4-bin codebook .pt).")
    p.add_argument("--dataset_root", type=str, default=None,
                   help="Override paths.dataset_root (codeepi_pyg_release_v1.0 directory).")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Override paths.output_dir. A seed subdirectory is created.")
    p.add_argument("--tag", type=str, default="student",
                   help="Filename tag written into results/ckpt names.")
    return p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def resolve_seed(cfg: Dict[str, Any], args: argparse.Namespace) -> Tuple[int, int]:
    """Return ``(seed, seed_index_1based)`` from CLI / config."""
    seeds = list(cfg.get("seeds", []))
    if args.seed is not None:
        s = int(args.seed)
        if seeds and s not in seeds:
            raise ValueError(
                f"--seed={s} is not in config.seeds={seeds}. Add it to the "
                f"config's seeds list first (the seed_index used in output "
                f"filenames is derived from that list)."
            )
        idx = seeds.index(s) + 1
        return s, idx
    if args.seed_index is not None:
        i = int(args.seed_index)
        if not seeds:
            raise ValueError("--seed_index given but config.seeds is empty")
        if not (1 <= i <= len(seeds)):
            raise ValueError(f"--seed_index must be in [1, {len(seeds)}], got {i}")
        return int(seeds[i - 1]), i
    raise ValueError("Provide either --seed or --seed_index.")


@torch.no_grad()
def eval_one_dual(
    model: CodeEpiStudent,
    loader: DataLoader,
    tau: float,
    device: torch.device,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:

    model.eval()
    Y_r, P_r, GI_r = [], [], []
    g_offset = 0
    for b in loader:
        b = b.to(device)
        out = model(b, tau=tau)
        p_surf_all = out["p_pred"].detach().cpu().numpy()
        bb = b.batch.detach().cpu().numpy()
        num_graphs = int(bb.max()) + 1 if bb.size > 0 else 0

        surf_idx_in_resolved_cpu = b.surf_idx_in_resolved.detach().cpu().numpy()
        n_resolved_vec = b.n_resolved.detach().cpu().numpy().reshape(-1).tolist()
        y_resolved_all = b.y_resolved.detach().cpu().numpy()
        surf_idx_in_resolved_pg: List[np.ndarray] = []
        y_resolved_pg: List[np.ndarray] = []
        y_res_cursor = 0
        for gi in range(num_graphs):
            surf_idx_in_resolved_pg.append(surf_idx_in_resolved_cpu[bb == gi])
            n_res_i = int(n_resolved_vec[gi])
            y_resolved_pg.append(y_resolved_all[y_res_cursor : y_res_cursor + n_res_i])
            y_res_cursor += n_res_i
        p_resolved_cat, gidx_res_cat = scatter_batch_surface_to_resolved(
            p_surf_all=p_surf_all,
            batch_of_surface_nodes=bb,
            surf_idx_in_resolved_per_graph=surf_idx_in_resolved_pg,
            n_resolved_per_graph=[int(x) for x in n_resolved_vec],
            graph_offset=g_offset,
        )
        P_r.append(p_resolved_cat.astype(np.float32))
        Y_r.append(np.concatenate(y_resolved_pg).astype(np.int64))
        GI_r.append(gidx_res_cat.astype(np.int64))

        g_offset += num_graphs

    return {
        "resolved": (
            np.concatenate(Y_r).astype(np.int64) if Y_r else np.zeros(0, dtype=np.int64),
            np.concatenate(P_r).astype(np.float32) if P_r else np.zeros(0, dtype=np.float32),
            np.concatenate(GI_r).astype(np.int64) if GI_r else np.zeros(0, dtype=np.int64),
        ),
    }


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_yaml(args.config)
    paths = load_yaml(args.paths)

    dataset_root = args.dataset_root or paths.get("dataset_root")
    codebook_pt = args.codebook or paths.get("codebook_checkpoint")
    output_dir = Path(args.output_dir or paths.get("output_dir", "outputs"))
    if not dataset_root:
        raise ValueError("dataset_root not provided (CLI or paths YAML).")
    if not codebook_pt:
        raise ValueError("codebook_checkpoint not provided (CLI or paths YAML).")

    seed, seed_index = resolve_seed(cfg, args)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = ensure_dir(output_dir / f"seed_{seed_index}")
    arm = f"{args.tag}_seed{seed_index}"
    print(f"[CodeEpi][train_student] arm={arm} seed={seed} device={device}")
    print(f"[CodeEpi][train_student] config={args.config}")
    print(f"[CodeEpi][train_student] paths={args.paths}")
    print(f"[CodeEpi][train_student] dataset_root={dataset_root}")
    print(f"[CodeEpi][train_student] codebook={codebook_pt}")
    print(f"[CodeEpi][train_student] output={run_dir}")

    model_cfg = cfg.get("model", {})
    model = CodeEpiStudent.from_codebook_file(
        codebook_path=str(codebook_pt),
        stem_dim=int(model_cfg.get("stem_dim", 511)),
        node_dim=int(model_cfg.get("node_dim", 512)),
        node_layers=int(model_cfg.get("node_layers", 3)),
        edge_dim=int(model_cfg.get("edge_dim", 32)),
        dropout=float(model_cfg.get("dropout", 0.4)),
        proj_dropout=float(model_cfg.get("proj_dropout", 0.1)),
        head_dropout=float(model_cfg.get("head_dropout", 0.1)),
        d_proto=int(model_cfg.get("d_proto", 256)),
    ).to(device)

    tcfg = cfg.get("train", {})
    lr = float(tcfg.get("lr", 1e-4))
    lr_min = float(tcfg.get("lr_min", 1e-6))
    wd = float(tcfg.get("weight_decay", 1e-5))
    max_epochs = int(tcfg.get("max_epochs", 60))
    patience = int(tcfg.get("patience", 15))
    batch_size = int(tcfg.get("batch_size", 4))
    num_workers = int(tcfg.get("num_workers", 0))
    pos_weight = float(tcfg.get("pos_weight", 2.0))
    gamma_neg = float(tcfg.get("gamma_neg", 2.0))
    label_smoothing = float(tcfg.get("label_smoothing", 0.2))
    ema_decay = float(tcfg.get("ema_decay", 0.9995))
    lam_anchor = float(tcfg.get("lam_anchor", 1e-4))
    tau_init = float(tcfg.get("tau_init", 0.5))
    tau_final = float(tcfg.get("tau_final", 0.05))

    print(
        f"[CodeEpi][train_student] hp: lr={lr} wd={wd} pw={pos_weight} "
        f"gn={gamma_neg} ls={label_smoothing} ema={ema_decay} "
        f"lam_anchor={lam_anchor} tau=[{tau_init}->{tau_final}] "
        f"max_epochs={max_epochs} patience={patience} bs={batch_size}"
    )

    train_ds = AntigenReleaseDataset(root=str(dataset_root), split="train")
    val_ds = AntigenReleaseDataset(root=str(dataset_root), split="val")
    test_ds = AntigenReleaseDataset(root=str(dataset_root), split="test")
    print(
        f"[CodeEpi][train_student] splits: "
        f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}"
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=lr_min)

    ema = EMA(model, decay=ema_decay)

    best_val_macro = -2.0
    best_thr_v_macro = 0.5
    best_epoch = -1
    bad_epochs = 0
    best_state: Dict[str, torch.Tensor] | None = None
    ep_done = 0

    for ep in range(max_epochs):
        ep_done = ep
        tau = cosine_tau(ep, max_epochs, tau_init=tau_init, tau_final=tau_final)
        model.train()

        loss_agg = {"code": 0.0, "anchor": 0.0, "nb": 0}
        for b in train_loader:
            b = b.to(device)
            out = model(b, tau=tau)
            p = out["p_pred"]
            y = b.y_node.float()
            l_code = afl_p(p, y, pos_weight, gamma_neg, label_smoothing)
            l_anchor = model.head.anchor_distance()
            loss = l_code + lam_anchor * l_anchor
            opt.zero_grad()
            loss.backward()
            opt.step()
            ema.update(model)
            loss_agg["code"] += float(l_code.item())
            loss_agg["anchor"] += float(l_anchor.item())
            loss_agg["nb"] += 1
        nb = max(loss_agg["nb"], 1)

        live_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict())
        dual_v_tick = eval_one_dual(model, val_loader, tau=tau_final, device=device)
        y_v, p_v, g_v = dual_v_tick["resolved"]
        thr_ma, val_ma = best_thr_macro(y_v, p_v, g_v)
        thr_mi, val_mi = best_thr_micro(y_v, p_v)
        dual_t_tick = eval_one_dual(model, test_loader, tau=tau_final, device=device)
        y_t, p_t, g_t = dual_t_tick["resolved"]
        test_ma_epoch = metrics_macro_at(y_t, p_t, g_t, thr_ma)
        test_mi_epoch = metrics_micro_at(y_t, p_t, thr_mi)
        with torch.no_grad():
            anc_dist = float(model.head.anchor_distance().item())
        cur_lr = opt.param_groups[0]["lr"]
        print(
            f"[ep {ep:3d}] tau={tau:.3f} "
            f"code={loss_agg['code']/nb:.4f} anc={loss_agg['anchor']/nb:.4f} "
            f"|| W - C||^2={anc_dist:.4f}  "
            f"test_ma={test_ma_epoch['mcc']:.4f}@{thr_ma:.2f}  "
            f"test_mi={test_mi_epoch['mcc']:.4f}@{thr_mi:.2f}  "
            f"lr={cur_lr:.2e}"
        )

        if val_ma > best_val_macro:
            best_val_macro = val_ma
            best_thr_v_macro = thr_ma
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        model.load_state_dict(live_state)
        sched.step()

        if bad_epochs >= patience:
            print(f"[CodeEpi][train_student] early stop @ ep{ep}  "
                  f"best_epoch={best_epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    dual_v = eval_one_dual(model, val_loader, tau=tau_final, device=device)
    dual_t = eval_one_dual(model, test_loader, tau=tau_final, device=device)
    y_v_r, p_v_r, g_v_r = dual_v["resolved"]
    y_t_r, p_t_r, g_t_r = dual_t["resolved"]

    thr_ma_resolved = best_thr_v_macro
    thr_mi_resolved, _ = best_thr_micro(y_v_r, p_v_r)

    with torch.no_grad():
        anc_final = float(model.head.anchor_distance().item())

    result = {
        "arm": arm,
        "seed": seed,
        "seed_index": seed_index,
        "tag": args.tag,
        "epochs_run": ep_done + 1,
        "hyperparameters": {
            "lr": lr,
            "lr_min": lr_min,
            "weight_decay": wd,
            "pos_weight": pos_weight,
            "gamma_neg": gamma_neg,
            "label_smoothing": label_smoothing,
            "ema_decay": ema_decay,
            "lam_anchor": lam_anchor,
            "tau_init": tau_init,
            "tau_final": tau_final,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "d_proto": int(model.head.d_proto),
            "K": int(model.head.K),
        },
        "codebook_checkpoint": str(codebook_pt),
        "dataset_root": str(dataset_root),
        "best_epoch": int(best_epoch),
        "best_val_macro": float(best_val_macro),
        "thr_val_macro": float(thr_ma_resolved),
        "thr_val_micro": float(thr_mi_resolved),
        "final_test_macro": metrics_macro_at(y_t_r, p_t_r, g_t_r, thr_ma_resolved),
        "final_test_micro": metrics_micro_at(y_t_r, p_t_r, thr_mi_resolved),
        "final_anchor_distance": anc_final,
    }

    json_out = run_dir / f"{args.tag}_seed{seed_index}.json"
    pt_out = run_dir / f"{args.tag}_seed{seed_index}.pt"
    with open(json_out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    torch.save({**result, "state_dict": best_state}, pt_out)

    print(
        f"[CodeEpi][train_student] done: "
        f"test_mcc={result['final_test_macro']['mcc']:.4f}  "
        f"test_f1={result['final_test_macro']['f1']:.4f}  "
        f"test_bacc={result['final_test_macro']['bacc']:.4f}  "
        f"test_agiou={result['final_test_macro']['agiou']:.4f}  "
        f"test_auroc={result['final_test_micro']['auroc']:.4f}  "
        f"test_auroc01={result['final_test_micro']['auroc_01']:.4f}  "
        f"test_auprc={result['final_test_micro']['auprc']:.4f}"
    )
    print(f"[CodeEpi][train_student] wrote {json_out}")
    print(f"[CodeEpi][train_student] wrote {pt_out}")


if __name__ == "__main__":
    main()
