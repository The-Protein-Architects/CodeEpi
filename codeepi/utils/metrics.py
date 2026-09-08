"""Metric helpers.

Two evaluation modes are used by the codebook student:

* **micro**: pool residues across all antigens, then compute the metric on the
  pooled arrays. 
* **macro (per-antigen)**: compute the metric per antigen, then average.

"""
from __future__ import annotations
from typing import Iterable, List, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)


def scatter_surface_to_resolved(
    p_surf: np.ndarray,
    surf_idx_in_resolved: np.ndarray,
    n_resolved: int,
) -> np.ndarray:
    p_resolved = np.zeros(int(n_resolved), dtype=np.float32)
    p_resolved[np.asarray(surf_idx_in_resolved, dtype=np.int64)] = np.asarray(p_surf, dtype=np.float32)
    return p_resolved


def scatter_batch_surface_to_resolved(
    p_surf_all: np.ndarray,
    batch_of_surface_nodes: np.ndarray,
    surf_idx_in_resolved_per_graph: List[np.ndarray],
    n_resolved_per_graph: List[int],
    graph_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    P, GI = [], []
    for gi, (surf_idx_res, n_res) in enumerate(zip(surf_idx_in_resolved_per_graph, n_resolved_per_graph)):
        mask = batch_of_surface_nodes == gi
        p_i = p_surf_all[mask]
        p_resolved = scatter_surface_to_resolved(p_i, surf_idx_res, int(n_res))
        P.append(p_resolved)
        GI.append(np.full(int(n_res), gi + graph_offset, dtype=np.int64))
    return (
        np.concatenate(P) if P else np.zeros(0, dtype=np.float32),
        np.concatenate(GI) if GI else np.zeros(0, dtype=np.int64),
    )


def per_graph_mcc_at(
    y: np.ndarray,
    p: np.ndarray,
    gidx: np.ndarray,
    thr: float,
) -> float:
    pred = (p >= thr).astype(np.int64)
    mccs = []
    for gi in np.unique(gidx):
        m = gidx == gi
        yi = y[m]
        if len(np.unique(yi)) < 2:
            continue
        pri = pred[m]
        if 0 < pri.sum() < len(pri):
            mccs.append(matthews_corrcoef(yi, pri))
        else:
            mccs.append(0.0)
    return float(np.mean(mccs)) if mccs else 0.0


def per_graph_f1_at(
    y: np.ndarray,
    p: np.ndarray,
    gidx: np.ndarray,
    thr: float,
) -> float:
    pred = (p >= thr).astype(np.int64)
    scores = []
    for gi in np.unique(gidx):
        m = gidx == gi
        yi = y[m]
        if len(np.unique(yi)) < 2:
            continue
        scores.append(float(f1_score(yi, pred[m], zero_division=0)))
    return float(np.mean(scores)) if scores else 0.0


def per_graph_bacc_at(
    y: np.ndarray,
    p: np.ndarray,
    gidx: np.ndarray,
    thr: float,
) -> float:
    pred = (p >= thr).astype(np.int64)
    scores = []
    for gi in np.unique(gidx):
        m = gidx == gi
        yi = y[m]
        if len(np.unique(yi)) < 2:
            continue
        scores.append(float(balanced_accuracy_score(yi, pred[m])))
    return float(np.mean(scores)) if scores else 0.0


def per_graph_agiou_at(
    y: np.ndarray,
    p: np.ndarray,
    gidx: np.ndarray,
    thr: float,
) -> float:
    pred = (p >= thr).astype(np.int64)
    scores = []
    for gi in np.unique(gidx):
        m = gidx == gi
        yi = y[m]
        pri = pred[m]
        if yi.sum() == 0:
            continue
        inter = int((yi & pri).sum())
        union = int(((yi | pri) > 0).sum())
        if union == 0:
            continue
        scores.append(inter / union)
    return float(np.mean(scores)) if scores else 0.0


def _linspace_thresholds(thr_lo: float, thr_hi: float, step: float) -> np.ndarray:
    n = int(round((thr_hi - thr_lo) / step)) + 1
    return np.linspace(thr_lo, thr_hi, n)


def sweep_threshold_macro(
    y: np.ndarray,
    p: np.ndarray,
    gidx: np.ndarray,
    thr_lo: float = 0.05,
    thr_hi: float = 0.95,
    step: float = 0.01,
) -> Tuple[float, float]:
    best_thr, best_mcc = 0.5, -2.0
    thrs = _linspace_thresholds(thr_lo, thr_hi, step)
    for t in thrs:
        m = per_graph_mcc_at(y, p, gidx, float(t))
        if m > best_mcc:
            best_mcc, best_thr = m, float(t)
    return best_thr, best_mcc


def sweep_threshold_micro(
    y: np.ndarray,
    p: np.ndarray,
    thr_lo: float = 0.0,
    thr_hi: float = 1.0,
    step: float = 0.01,
) -> Tuple[float, float]:
    best_thr, best_mcc = 0.5, -2.0
    thrs = _linspace_thresholds(thr_lo, thr_hi, step)
    for t in thrs:
        pred = (p >= t).astype(np.int64)
        if 0 < pred.sum() < len(pred):
            m = matthews_corrcoef(y, pred)
            if m > best_mcc:
                best_mcc, best_thr = float(m), float(t)
    return best_thr, best_mcc


def partial_auroc_at(y: np.ndarray, p: np.ndarray, max_fpr: float = 0.1) -> float:
    if len(np.unique(y)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y, p)
    if fpr[-1] < max_fpr:
        area = float(np.trapz(tpr, fpr))
        return area / max_fpr
    tpr_at = float(np.interp(max_fpr, fpr, tpr))
    mask = fpr <= max_fpr
    fpr_slice = np.append(fpr[mask], max_fpr)
    tpr_slice = np.append(tpr[mask], tpr_at)
    area = float(np.trapz(tpr_slice, fpr_slice))
    return area / max_fpr


def micro_iou_at(y: np.ndarray, p: np.ndarray, thr: float) -> float:
    pred = (p >= thr).astype(np.int64)
    yb = (y == 1)
    prb = (pred == 1)
    inter = int((yb & prb).sum())
    union = int((yb | prb).sum())
    return float(inter / union) if union > 0 else 0.0


def micro_metrics_at(
    y: np.ndarray,
    p: np.ndarray,
    thr: float,
) -> dict:
    n_pos = int((y == 1).sum())
    n = int(len(y))
    if len(np.unique(y)) < 2:
        return {
            "mcc": 0.0, "f1": 0.0, "bacc": 0.0, "agiou": 0.0,
            "auprc": 0.0, "auroc": 0.0, "auroc_01": 0.0,
            "n": n, "n_pos": n_pos, "n_neg": n - n_pos,
        }
    pred = (p >= thr).astype(np.int64)
    return {
        "mcc": float(matthews_corrcoef(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "agiou": micro_iou_at(y, p, thr),
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "auroc_01": partial_auroc_at(y, p, max_fpr=0.1),
        "n": n,
        "n_pos": n_pos,
        "n_neg": n - n_pos,
    }


def macro_metrics_at(
    y: np.ndarray,
    p: np.ndarray,
    gidx: np.ndarray,
    thr: float,
) -> dict:
    n = int(len(y))
    n_pos = int((y == 1).sum())
    return {
        "mcc":  per_graph_mcc_at(y, p, gidx, thr),
        "f1":   per_graph_f1_at(y, p, gidx, thr),
        "bacc": per_graph_bacc_at(y, p, gidx, thr),
        "agiou": per_graph_agiou_at(y, p, gidx, thr),
        "n": n,
        "n_pos": n_pos,
        "n_neg": n - n_pos,
        "n_graphs": int(len(np.unique(gidx))),
    }


best_thr_macro = sweep_threshold_macro
best_thr_micro = sweep_threshold_micro
metrics_macro_at = macro_metrics_at
metrics_micro_at = micro_metrics_at

__all__ = [
    "scatter_surface_to_resolved",
    "scatter_batch_surface_to_resolved",
    "per_graph_mcc_at",
    "per_graph_f1_at",
    "per_graph_bacc_at",
    "per_graph_agiou_at",
    "micro_iou_at",
    "partial_auroc_at",
    "sweep_threshold_macro",
    "sweep_threshold_micro",
    "micro_metrics_at",
    "macro_metrics_at",
    "best_thr_macro",
    "best_thr_micro",
    "metrics_macro_at",
    "metrics_micro_at",
]
