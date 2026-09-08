"""Training entry point for the CodeEpi (group-aware) teacher.

The teacher is trained per rep antigen group over the ``Complex1723``
release: each mini-batch item packs one representative plus every sibling
complex that shares its antigen SEQRES. The rep-fused antigen surface and
its epitope label come from ``Antigen1321`` (release-shipped fused masks),
and per-member ESM-IF / ESM-C / internal graphs / 4.5 A CDR-surface
contacts come from ``Complex1723``.

After training finishes, the trainer reloads ``ckpts/best.pt`` and dumps a
``teacher_residue_table.pt`` in the run directory containing per-rep
teacher probabilities and the 256-d post-QKV antibody context antigen embeddings 
from the train split. That file is consumed by ``codeepi.codebook.build_codebook``; 
it is an intermediate artifact and does not enter git or the release checkpoint
archives. Disable with ``--no_export_residue_table``.

Example
-------
    python -m codeepi.teacher.train_teacher \
        --config configs/train/teacher.yaml \
        --paths configs/paths.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, matthews_corrcoef
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from ..utils.io import ensure_dir, load_yaml
from ..utils.seed import seed_everything
from .group_dataset import (
    GroupTeacherDataset,
    ReleaseMemberGraphLoader,
    build_group_index,
    collate_groups,
)
from .model import CodeEpiTeacher


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="codeepi.teacher.train_teacher",
        description="Train the CodeEpi teacher (group-aware).",
    )
    p.add_argument("--config", type=str, required=True,
                   help="YAML file with train / model settings.")
    p.add_argument("--paths", type=str, required=True,
                   help="YAML file with dataset_root and output_dir.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override the training seed.")
    p.add_argument("--dataset_root", type=str, default=None,
                   help="Override paths.dataset_root (codeepi_pyg_release_v1.0 directory).")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Override paths.output_dir.")
    p.add_argument("--tag", type=str, default="teacher",
                   help="Filename tag written into results / checkpoint names.")
    p.add_argument("--no_export_residue_table", action="store_true",
                   help="Skip the post-training export of "
                        "teacher_residue_table.pt (the codebook-construction "
                        "intermediate; produced by default from best.pt on "
                        "the train split).")
    return p


# ---------------------------------------------------------------------------
# metric helpers (per-rep macro MCC)
# ---------------------------------------------------------------------------

def _sweep_macro_mcc(
    per_graph_y: List[np.ndarray],
    per_graph_p: List[np.ndarray],
    thr_lo: float = 0.05,
    thr_hi: float = 0.95,
    step: float = 0.01,
) -> Tuple[float, float]:
    """Sweep a global threshold; return (best_thr, per-rep mean MCC)."""
    best_thr, best_mcc = 0.5, -2.0
    for t in np.arange(thr_lo, thr_hi + 1e-9, step):
        per = []
        for yi, pi in zip(per_graph_y, per_graph_p):
            pred = (pi >= t).astype(int)
            if len(np.unique(pred)) < 2 or len(np.unique(yi)) < 2:
                per.append(0.0)
            else:
                per.append(matthews_corrcoef(yi, pred))
        m = float(np.mean(per)) if per else 0.0
        if m > best_mcc:
            best_mcc, best_thr = m, float(t)
    return best_thr, best_mcc


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
def _evaluate_loader(model: CodeEpiTeacher, loader: DataLoader,
                     device: torch.device
                     ) -> Tuple[List[np.ndarray], List[np.ndarray],
                                List[np.ndarray], List[np.ndarray]]:
    """Return four parallel per-rep lists:
      * per_y_surf    : (n_surf,)     surface-only labels
      * per_p_surf    : (n_surf,)     surface-only predicted probs
      * per_y_resolved: (n_resolved,) PDB-ATOM labels (non-surf=0)
      * per_p_resolved: (n_resolved,) probs scattered to PDB-ATOM space

    Order across all four lists is identical: element ``i`` refers to the
    same rep antigen.
    """
    model.eval()
    per_y_surf, per_p_surf = [], []
    per_y_resolved, per_p_resolved = [], []
    for batch in loader:
        surf_idx_in_res_per_rep = batch.get("surf_idx_in_resolved_per_rep", None)
        n_resolved_per_rep = batch.get("n_resolved_per_rep", None)
        y_resolved_per_rep = batch.get("y_resolved_per_rep", None)
        if (surf_idx_in_res_per_rep is None or n_resolved_per_rep is None
                or y_resolved_per_rep is None):
            raise KeyError(
                "teacher batch missing PDB-ATOM side-channel "
                "(surf_idx_in_resolved_per_rep / n_resolved_per_rep / "
                "y_resolved_per_rep); rebuild collate_groups from the "
                "current group_dataset."
            )
        batch = _move_batch(batch, device)
        out = model(batch)
        probs = torch.sigmoid(out["logits"]).cpu().numpy()
        y = out["y_true"].cpu().numpy()
        bg = out["batch_g"].cpu().numpy()
        if bg.size == 0:
            continue
        for gi in range(int(bg.max()) + 1):
            m = bg == gi
            yi_s = y[m]
            pi_s = probs[m]
            per_y_surf.append(yi_s)
            per_p_surf.append(pi_s)

            sir = surf_idx_in_res_per_rep[gi]
            sir_np = (sir.cpu().numpy() if torch.is_tensor(sir)
                      else np.asarray(sir))
            n_res_i = int(n_resolved_per_rep[gi])
            y_res_i = y_resolved_per_rep[gi]
            y_res_np = (y_res_i.cpu().numpy() if torch.is_tensor(y_res_i)
                        else np.asarray(y_res_i))
            p_res = np.zeros(n_res_i, dtype=np.float32)
            p_res[sir_np[: pi_s.shape[0]]] = pi_s
            per_p_resolved.append(p_res)
            per_y_resolved.append(y_res_np.astype(np.int64))

    return per_y_surf, per_p_surf, per_y_resolved, per_p_resolved


# ---------------------------------------------------------------------------
# residue-table export (codebook input)
# ---------------------------------------------------------------------------

@torch.no_grad()
def export_teacher_residue_table(
    model: CodeEpiTeacher,
    loader: DataLoader,
    device: torch.device,
    out_path: Path,
    source_checkpoint: str,
) -> Dict[str, Any]:
    """Dump per-train-rep-position teacher probs + 256-d post-QKV emb.

    Written schema (bit-compatible with ``codeepi.codebook.build_codebook``)::

        {
          "teacher_prob"       : Tensor (M,)      float32
          "after_qkv_emb_256"  : Tensor (M, 256)  float32
          "y_true"             : Tensor (M,)      float32
          "abdbid"             : list[str]        length B, per-rep
          "sample_index"       : Tensor (M,)      long, 0..B-1 into abdbid
          "batch_g"            : Tensor (M,)      long, per-node graph index
                                                  within its own batch
          "split"              : list[str]        length M, all "train"
          "format_version"    : "1.0"
          "source_checkpoint"  : str
        }

    "M" counts one entry per rep-fused-surface node across the train reps.
    """
    model.eval()
    all_prob: list[torch.Tensor] = []
    all_emb: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []
    all_batch: list[torch.Tensor] = []
    all_abdbid: list[str] = []
    all_sample_idx: list[torch.Tensor] = []
    running_offset = 0

    for batch in loader:
        rep_ids = list(batch.get("rep_abdbids", []))
        batch = _move_batch(batch, device)
        out = model(batch, return_embeddings=True)
        logits = out["logits"].detach().cpu()
        emb = out["after_qkv_emb_256"].detach().cpu().float()
        y = out["y_true"].detach().cpu().float()
        bg = out["batch_g"].detach().cpu().long()

        all_prob.append(torch.sigmoid(logits).float())
        all_emb.append(emb)
        all_y.append(y)
        all_batch.append(bg)

        n_g = int(bg.max().item()) + 1 if bg.numel() > 0 else 0
        if not rep_ids:
            rep_ids = [""] * n_g
        elif len(rep_ids) != n_g:
            rep_ids = list(rep_ids) + [""] * max(0, n_g - len(rep_ids))
            rep_ids = rep_ids[:n_g]

        if bg.numel() > 0:
            all_sample_idx.append(bg + running_offset)
        running_offset += len(rep_ids)
        all_abdbid.extend(rep_ids)

    teacher_prob = torch.cat(all_prob, dim=0) if all_prob else torch.zeros(0)
    after_qkv = (torch.cat(all_emb, dim=0)
                 if all_emb else torch.zeros(0, model.gnn_out))
    y_true = torch.cat(all_y, dim=0) if all_y else torch.zeros(0)
    batch_g = (torch.cat(all_batch, dim=0)
               if all_batch else torch.zeros(0, dtype=torch.long))
    sample_idx = (torch.cat(all_sample_idx, dim=0)
                  if all_sample_idx else torch.zeros(0, dtype=torch.long))
    M = int(teacher_prob.numel())

    payload: Dict[str, Any] = {
        "teacher_prob": teacher_prob,
        "after_qkv_emb_256": after_qkv,
        "y_true": y_true,
        "batch_g": batch_g,
        "sample_index": sample_idx,
        "abdbid": all_abdbid,
        "split": ["train"] * M,
        "format_version": "1.0",
        "source_checkpoint": str(source_checkpoint),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out_path))
    return payload


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_yaml(args.config)
    paths = load_yaml(args.paths)

    dataset_root = args.dataset_root or paths.get("dataset_root")
    output_dir = Path(args.output_dir or paths.get("output_dir", "outputs"))
    if not dataset_root:
        raise ValueError("dataset_root not provided (CLI or paths YAML).")

    tcfg = cfg.get("train", {})
    mcfg = cfg.get("model", {})

    seed = args.seed if args.seed is not None else int(tcfg.get("seed", 2503059))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = ensure_dir(output_dir / f"{args.tag}_seed{seed}")
    (run_dir / "ckpts").mkdir(parents=True, exist_ok=True)
    print(f"[CodeEpi][train_teacher] tag={args.tag} seed={seed} device={device}")
    print(f"[CodeEpi][train_teacher] dataset_root={dataset_root}")
    print(f"[CodeEpi][train_teacher] output={run_dir}")

    # -- build the group index and per-split rep lists --
    print("[CodeEpi][train_teacher] building group index from release ...",
          flush=True)
    t_gi = time.time()
    group_index = build_group_index(dataset_root)
    split_of = group_index["split_of"]
    train_ids = [r for r, s in split_of.items() if s == "train"]
    val_ids = [r for r, s in split_of.items() if s == "val"]
    test_ids = [r for r, s in split_of.items() if s == "test"]
    print(f"[CodeEpi][train_teacher] reps train/val/test = "
          f"{len(train_ids)}/{len(val_ids)}/{len(test_ids)} "
          f"(built in {time.time() - t_gi:.1f}s)")

    member_loader = ReleaseMemberGraphLoader(dataset_root)
    train_ds = GroupTeacherDataset(train_ids, group_index, member_loader)
    val_ds = GroupTeacherDataset(val_ids, group_index, member_loader)
    test_ds = GroupTeacherDataset(test_ids, group_index, member_loader)

    batch_size = int(tcfg.get("batch_size", 2))
    num_workers = int(tcfg.get("num_workers", 2))
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  collate_fn=collate_groups)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    model = CodeEpiTeacher(
        fusion_type=str(mcfg.get("fusion_type", "projected_esmc")),
        gnn_hidden=int(mcfg.get("gnn_hidden", 384)),
        gnn_out=int(mcfg.get("gnn_out", 256)),
        num_heads=int(mcfg.get("num_heads", 8)),
        ffn_hidden=int(mcfg.get("ffn_hidden", 512)),
        cls_h1=int(mcfg.get("cls_h1", 128)),
        cls_h2=int(mcfg.get("cls_h2", 64)),
        dropout=float(mcfg.get("dropout", 0.1)),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[CodeEpi][train_teacher] params={n_params/1e6:.2f}M")

    pos_w = torch.tensor([float(tcfg.get("pos_weight", 8.0))], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = Adam(model.parameters(), lr=float(tcfg.get("lr", 1e-3)))
    sched = StepLR(opt, step_size=int(tcfg.get("lr_step", 20)),
                   gamma=float(tcfg.get("lr_gamma", 0.5)))

    max_epochs = int(tcfg.get("max_epochs", 100))
    patience = int(tcfg.get("patience", 15))

    best_val = -1.0
    best_epoch = -1
    best_state: Dict[str, torch.Tensor] | None = None
    no_improve = 0
    history: List[Dict[str, Any]] = []

    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        ep_loss = 0.0
        nb = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            out = model(batch)
            loss = loss_fn(out["logits"], out["y_true"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        sched.step()
        train_loss = ep_loss / max(nb, 1)

        gy_v_s, gp_v_s, gy_v_r, gp_v_r = _evaluate_loader(model, val_loader, device)
        gy_t_s, gp_t_s, gy_t_r, gp_t_r = _evaluate_loader(model, test_loader, device)
        # Best-ckpt selection: PDB-ATOM (resolved) macro MCC — the primary
        # reporting coordinate (v1.1) and what the student also uses.
        thr_v, val_macro = _sweep_macro_mcc(gy_v_r, gp_v_r)
        thr_v_surf, val_macro_surf = _sweep_macro_mcc(gy_v_s, gp_v_s)

        def _macro_at(yl: List[np.ndarray], pl: List[np.ndarray], thr: float) -> float:
            per = []
            for yi, pi in zip(yl, pl):
                pred = (pi >= thr).astype(int)
                if len(np.unique(pred)) < 2 or len(np.unique(yi)) < 2:
                    per.append(0.0)
                else:
                    per.append(matthews_corrcoef(yi, pred))
            return float(np.mean(per)) if per else 0.0

        # thr_v was swept on PDB-ATOM val; apply it to PDB-ATOM test as primary.
        test_macro_resolved = _macro_at(gy_t_r, gp_t_r, thr_v)
        test_macro = _macro_at(gy_t_s, gp_t_s, thr_v_surf)

        y_all_v = np.concatenate(gy_v_s) if gy_v_s else np.array([])
        p_all_v = np.concatenate(gp_v_s) if gp_v_s else np.array([])
        y_all_t = np.concatenate(gy_t_s) if gy_t_s else np.array([])
        p_all_t = np.concatenate(gp_t_s) if gp_t_s else np.array([])
        auprc_v = (float(average_precision_score(y_all_v, p_all_v))
                   if y_all_v.size and y_all_v.max() > 0 else 0.0)
        auprc_t = (float(average_precision_score(y_all_t, p_all_t))
                   if y_all_t.size and y_all_t.max() > 0 else 0.0)

        rec = {
            "epoch": ep, "train_loss": train_loss,
            # PRIMARY: PDB-ATOM (resolved) early-stop metric
            "val_macro_mcc": val_macro, "val_thr": thr_v,
            "test_macro_mcc_resolved": test_macro_resolved,
            "val_macro_mcc_surface": val_macro_surf,
            "test_macro_mcc_surface": test_macro,
            "val_thr_surface": thr_v_surf,
            "val_auprc_surface": auprc_v,
            "test_auprc_surface": auprc_t,
            "lr": opt.param_groups[0]["lr"], "time_s": time.time() - t0,
        }
        history.append(rec)
        # Val is used for early-stop and threshold selection
        print(f"[ep {ep:3d}] loss={train_loss:.4f} "
              f"test_mcc(pdbatom)={test_macro_resolved:.4f}@thr{thr_v:.2f} "
              f"|| test_mcc(surf)={test_macro:.4f}@thr{thr_v_surf:.2f} "
              f"test_auprc(surf)={auprc_t:.4f} "
              f"lr={rec['lr']:.2e} t={rec['time_s']:.1f}s",
              flush=True)

        if val_macro > best_val + 1e-6:
            best_val = val_macro
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            torch.save({
                "state_dict": best_state,
                "epoch": ep,
                # PRIMARY: PDB-ATOM (resolved) — matches early-stop criterion
                "val_macro_mcc": val_macro,
                "val_thr": thr_v,
                "test_macro_mcc_resolved": test_macro_resolved,
                "val_macro_mcc_surface": val_macro_surf,
                "test_macro_mcc_surface": test_macro,
                "val_thr_surface": thr_v_surf,
                "config": cfg,
                "seed": seed,
            }, run_dir / "ckpts" / "best.pt")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[CodeEpi][train_teacher] early stop @ ep{ep}")
                break

    summary = {
        "tag": args.tag,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_mcc": float(best_val),
        "args": vars(args),
        "config": cfg,
    }
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(run_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    best_test_pdbatom = next(
        (h["test_macro_mcc_resolved"] for h in history if h["epoch"] == best_epoch),
        float("nan"),
    )
    best_test_surface = next(
        (h["test_macro_mcc_surface"] for h in history if h["epoch"] == best_epoch),
        float("nan"),
    )
    print(f"[CodeEpi][train_teacher] done: best_epoch={best_epoch}  "
          f"test_mcc(pdbatom)={best_test_pdbatom:.4f}  "
          f"test_mcc(surf)={best_test_surface:.4f}")

    ckpt_path = run_dir / "ckpts" / "best.pt"
    if args.no_export_residue_table:
        print("[CodeEpi][train_teacher] --no_export_residue_table given; "
              "skipping teacher_residue_table.pt export.")
    elif best_state is None or not ckpt_path.exists():
        print("[CodeEpi][train_teacher] no best.pt was written (no epoch "
              "improved on val); skipping teacher_residue_table.pt export.")
    else:
        print(f"[CodeEpi][train_teacher] reloading {ckpt_path} for "
              f"teacher_residue_table.pt export ...")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        export_loader = DataLoader(train_ds, shuffle=False, **common)
        out_path = run_dir / "teacher_residue_table.pt"
        payload = export_teacher_residue_table(
            model=model,
            loader=export_loader,
            device=device,
            out_path=out_path,
            source_checkpoint=str(ckpt_path.relative_to(run_dir)),
        )
        print(f"[CodeEpi][train_teacher] wrote {out_path} "
              f"(M={int(payload['teacher_prob'].numel())} residues, "
              f"emb dim={int(payload['after_qkv_emb_256'].shape[-1])})")


if __name__ == "__main__":
    main()
