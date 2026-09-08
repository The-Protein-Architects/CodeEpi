"""CodeEpi student model.

The student is an antigen-only network. It consumes:

  * ``x_esmif``   (N, 512)   ESM-IF antigen-only encoding
  * ``x_esmc``    (N, 2560)  ESM-C encoding
  * ``pos``       (N, 3, 3)  backbone atoms (N, Ca, C)
  * ``rsa``       (N,)       relative solvent accessibility
  * ``edge_index``(2, E)     antigen graph (8A Ca radius)
  * ``edge_attr`` (E, 32)    16-D RBF distance + 16-D positional encoding

It uses a fixed teacher-derived 4-bin codebook to map per-residue
representations to epitope probabilities.

The architecture is a 3-layer EGNN backbone followed by a
prototype-attention head that reads out
``p = sum_k softmax(sim(z, W) / tau)_k * bin_score_k``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.layers import DihedralFeatures, EGNNLayer


D_PROTO_DEFAULT = 256


class CodeEpiStudentBackbone(nn.Module):
    """Antigen-only backbone: ESM-C projection + ESM-IF + dihedral + RSA -> 3xEGNN.
    """

    def __init__(
        self,
        stem_dim: int = 511,
        node_dim: int = 512,
        node_layers: int = 3,
        edge_dim: int = 32,
        dropout: float = 0.4,
        proj_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem_dim = stem_dim
        self.node_dim = node_dim

        self.esmc_proj = nn.Sequential(
            nn.LayerNorm(2560),
            nn.Linear(2560, 512),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.LayerNorm(512),
        )
        self.ln_esmif = nn.LayerNorm(512)
        self.fusion = nn.Sequential(
            nn.Linear(1024, stem_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.LayerNorm(stem_dim),
        )
        self.dihedral_features = DihedralFeatures(stem_dim)
        self.egnn_layers = nn.ModuleList()
        for _ in range(node_layers):
            self.egnn_layers.append(
                EGNNLayer(
                    input_nf=node_dim,
                    output_nf=node_dim,
                    hidden_nf=node_dim,
                    edges_in_d=edge_dim,
                    act_fn=nn.GELU(),
                    residual=True,
                    attention=True,
                    normalize=True,
                    coords_agg="mean",
                    dropout=dropout,
                    ffn=True,
                    batch_norm=True,
                )
            )

    def forward(self, data) -> torch.Tensor:
        x_esmif = data.x_esmif
        x_esmc = data.x_esmc
        pos = data.pos
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        rsa_val = data.rsa
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(x_esmif.size(0), dtype=torch.long, device=x_esmif.device)
        )

        h_c = self.esmc_proj(x_esmc)
        h_if = self.ln_esmif(x_esmif)
        h = self.fusion(torch.cat([h_if, h_c], dim=-1))
        h = h + self.dihedral_features(pos)
        h = torch.cat([h, rsa_val.unsqueeze(-1)], dim=-1)  # -> node_dim

        coord = pos[:, 1, :].clone()
        for layer in self.egnn_layers:
            h, coord, _ = layer(h, coord, edge_index, batch, edge_attr=edge_attr)
        return h


class ProtoHead(nn.Module):
    """Prototype-attention head over a fixed 4-bin teacher-derived codebook.
    """

    def __init__(
        self,
        K: int,
        bin_score: torch.Tensor,
        C_init: torch.Tensor,
        d_in: int = 512,
        d_proto: int = D_PROTO_DEFAULT,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if C_init.shape != (K, d_proto):
            raise ValueError(
                f"ProtoHead: C_init shape {tuple(C_init.shape)} != ({K}, {d_proto})"
            )
        if bin_score.numel() != K:
            raise ValueError(
                f"ProtoHead: bin_score numel {bin_score.numel()} != K={K}"
            )
        self.K = int(K)
        self.d_proto = int(d_proto)
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_proto),
            nn.LayerNorm(d_proto),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.W = nn.Parameter(C_init.clone().float())
        self.register_buffer("C_anchor", C_init.clone().float())
        self.register_buffer("bin_score", bin_score.float().clone())

    def forward(self, h: torch.Tensor, tau: float) -> Dict[str, torch.Tensor]:
        z = self.proj(h)
        sim = F.normalize(z, dim=-1) @ F.normalize(self.W, dim=-1).t()
        a = F.softmax(sim / tau, dim=-1)
        p = (a * self.bin_score.unsqueeze(0)).sum(-1).clamp(1e-6, 1 - 1e-6)
        return {"alpha": a, "p_pred": p, "z": z}

    def anchor_distance(self) -> torch.Tensor:
        return (F.normalize(self.W, dim=-1) - self.C_anchor).pow(2).sum()


class CodeEpiStudent(nn.Module):
    def __init__(
        self,
        backbone: CodeEpiStudentBackbone,
        head: ProtoHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, data, tau: float) -> Dict[str, torch.Tensor]:
        h = self.backbone(data)
        out = self.head(h, tau=tau)
        out["h"] = h
        return out

    @classmethod
    def from_codebook_file(
        cls,
        codebook_path: str,
        stem_dim: int = 511,
        node_dim: int = 512,
        node_layers: int = 3,
        edge_dim: int = 32,
        dropout: float = 0.4,
        proj_dropout: float = 0.1,
        head_dropout: float = 0.1,
        d_proto: int = D_PROTO_DEFAULT,
    ) -> "CodeEpiStudent":
        pr = torch.load(codebook_path, map_location="cpu", weights_only=False)
        K = int(pr["K"])
        bin_score = pr["bin_score"]
        C_init = pr.get("C_weighted_mean")
        if C_init is None:
            C_init = pr["C_mean"]
        backbone = CodeEpiStudentBackbone(
            stem_dim=stem_dim,
            node_dim=node_dim,
            node_layers=node_layers,
            edge_dim=edge_dim,
            dropout=dropout,
            proj_dropout=proj_dropout,
        )
        head = ProtoHead(
            K=K,
            bin_score=bin_score,
            C_init=C_init,
            d_in=node_dim,
            d_proto=d_proto,
            dropout=head_dropout,
        )
        return cls(backbone=backbone, head=head)
