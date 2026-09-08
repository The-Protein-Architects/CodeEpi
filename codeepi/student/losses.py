"""Loss functions and EMA helper for the CodeEpi student.
"""
from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn


def afl_p(
    p: torch.Tensor,
    y: torch.Tensor,
    pos_weight: float,
    gamma_neg: float,
    label_smoothing: float,
) -> torch.Tensor:
    p_c = p.clamp(1e-6, 1 - 1e-6)
    y_s = y * (1 - label_smoothing) + 0.5 * label_smoothing
    bce = -(y_s * torch.log(p_c) + (1 - y_s) * torch.log(1 - p_c))
    w_pos = torch.where(
        y >= 0.5,
        torch.tensor(pos_weight, device=p.device, dtype=p.dtype),
        torch.tensor(1.0, device=p.device, dtype=p.dtype),
    )
    bce = bce * w_pos
    w_focal = torch.where(
        y >= 0.5,
        torch.ones_like(p_c),
        p_c.pow(gamma_neg),
    )
    return (bce * w_focal).mean()


class EMA:
    _DEFAULT_EXCLUDE = ("head.C_anchor", "head.bin_score")

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9995,
        exclude: Iterable[str] = _DEFAULT_EXCLUDE,
    ) -> None:
        self.decay = float(decay)
        self.exclude = frozenset(exclude)
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.exclude or not v.dtype.is_floating_point:
                self.shadow[k].copy_(v.detach())
            else:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow


def cosine_tau(
    epoch: int,
    total_epochs: int,
    tau_init: float = 0.5,
    tau_final: float = 0.05,
) -> float:
    import math

    denom = max(total_epochs - 1, 1)
    return float(
        tau_final
        + 0.5 * (tau_init - tau_final) * (1.0 + math.cos(math.pi * epoch / denom))
    )
