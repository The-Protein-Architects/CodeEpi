"""CodeEpi teacher: group-aware antibody-antigen cross-attention.

The teacher aggregates antibody CDR context across every complex that shares
an antigen SEQRES (a *rep antigen group*) and predicts per-residue epitope
probabilities on the rep-fused antigen surface. The architecture matches 
the CodeEpi paper (Section 2.3)

    1. ESM-IF + ESM-C fusion (T2 ``ProjectedESMCFusion``) on both CDR and
       antigen sides, 2560+512 -> 512.
    2. Two-layer GCN encoders (512 -> 384 -> 256) applied independently to
       CDR and antigen residues; no cross-molecule messages.
    3. Group aggregation to the rep-fused surface:
         * antigen: mean over K members onto ``(B, N_rep, 256)``.
         * antibody: query-conditioned segment-softmax pool (over 4.5 A
           real contacts) for K>=2 groups; per-position mean for K=1
           groups.
    4. Cross-attention with rep antigen residues as query, group-fused CDR
       residues as key/value.
    5. Two-hidden-layer MLP classifier -> logit per rep-fused-surface node.

Input contract
--------------
``forward`` accepts either:

    (a) A dict produced by ``codeepi.teacher.group_dataset.collate_groups``.
        This is the training / codebook-export path.
    (b) A PyG ``Data`` object with per-complex fields. This is
        the inference / release-prediction path and is treated as a batch
        of ``size==1`` groups (rep = the complex itself), which reduces to
        the plain-mean fallback in the  architecture.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


def _scatter_sum(src: torch.Tensor, index: torch.Tensor,
                 dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(0, index, src)


def _scatter_max_1d(src: torch.Tensor, index: torch.Tensor,
                    dim_size: int) -> torch.Tensor:
    neg_inf = torch.finfo(src.dtype).min
    out = torch.full((dim_size,), neg_inf, dtype=src.dtype, device=src.device)
    return out.scatter_reduce_(0, index, src, reduce="amax",
                               include_self=True)


# ---------------------------------------------------------------------------
# fusion module 
# ---------------------------------------------------------------------------

class ProjectedESMCFusion(nn.Module):
    """Project ESM-C (2560) to 512, concat with ESM-IF, fuse to 512."""

    def __init__(self, esmif_dim: int = 512, esmc_dim: int = 2560,
                 out_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln_esmif = nn.LayerNorm(esmif_dim)
        self.esmc_proj = nn.Sequential(
            nn.LayerNorm(esmc_dim),
            nn.Linear(esmc_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x_esmif: torch.Tensor, x_esmc: torch.Tensor) -> torch.Tensor:
        h_if = self.ln_esmif(x_esmif)
        h_c = self.esmc_proj(x_esmc)
        return self.fusion(torch.cat([h_if, h_c], dim=-1))


# ---------------------------------------------------------------------------
# cross-attention block 
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_hidden: int,
                 dropout: float) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                          batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, q_x: torch.Tensor, kv_x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q_n = self.ln_q(q_x)
        kv_n = self.ln_kv(kv_x)
        attn_out, _ = self.attn(q_n, kv_n, kv_n,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)
        z = q_x + self.dropout1(attn_out)
        z = z + self.dropout2(self.ffn(self.ln_ffn(z)))
        return z


# ---------------------------------------------------------------------------
# scatter helpers on a (B, max_pos) grid
# ---------------------------------------------------------------------------

def _scatter_mean_2d(values: torch.Tensor, group_idx: torch.Tensor,
                     pos_idx: torch.Tensor, B: int, max_pos: int, D: int,
                     eps: float = 1e-6):

    device, dtype = values.device, values.dtype
    out = torch.zeros(B, max_pos, D, device=device, dtype=dtype)
    cnt = torch.zeros(B, max_pos, 1, device=device, dtype=dtype)
    if values.shape[0] == 0:
        return out, torch.zeros(B, max_pos, dtype=torch.bool, device=device)
    flat_idx = group_idx * max_pos + pos_idx
    out_flat = out.view(B * max_pos, D)
    out_flat.index_add_(0, flat_idx, values)
    cnt_flat = cnt.view(B * max_pos, 1)
    cnt_flat.index_add_(0, flat_idx, torch.ones_like(cnt_flat[flat_idx]))
    mask = (cnt > 0).squeeze(-1)
    out = out / cnt.clamp(min=eps)
    return out, mask


def _segment_softmax_pool(values: torch.Tensor, group_idx: torch.Tensor,
                          pos_idx: torch.Tensor, scores: torch.Tensor,
                          B: int, max_pos: int, D: int):
    """Softmax-weighted pool inside each ``(group, pos)`` bucket."""
    device, dtype = values.device, values.dtype
    if values.shape[0] == 0:
        return (torch.zeros(B, max_pos, D, device=device, dtype=dtype),
                torch.zeros(B, max_pos, dtype=torch.bool, device=device))
    flat = group_idx * max_pos + pos_idx
    N = B * max_pos
    s_max = _scatter_max_1d(scores, flat, N)
    s_max_per = s_max[flat]
    exp_s = (scores - s_max_per).exp()
    denom = _scatter_sum(exp_s, flat, N).clamp(min=1e-9)
    alpha = exp_s / denom[flat]
    weighted = values * alpha.unsqueeze(-1)
    out_flat = torch.zeros(N, D, dtype=values.dtype, device=device)
    out_flat.index_add_(0, flat, weighted)
    out = out_flat.view(B, max_pos, D)
    cnt = _scatter_sum(torch.ones_like(scores), flat, N).view(B, max_pos)
    mask = cnt > 0
    return out, mask


# ---------------------------------------------------------------------------
# CodeEpi teacher (group-aggregate)
# ---------------------------------------------------------------------------

class CodeEpiTeacher(nn.Module):
    """group-aware teacher"""

    def __init__(
        self,
        fusion_type: str = "projected_esmc",
        esmif_dim: int = 512,
        esmc_dim: int = 2560,
        gnn_hidden: int = 384,
        gnn_out: int = 256,
        num_heads: int = 8,
        ffn_hidden: int = 512,
        cls_h1: int = 128,
        cls_h2: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if fusion_type != "projected_esmc":
            raise ValueError(
                f"CodeEpiTeacher: only fusion_type='projected_esmc' is packaged; "
                f"got {fusion_type!r}"
            )
        self.ab_fusion = ProjectedESMCFusion(esmif_dim, esmc_dim, esmif_dim, dropout)
        self.ag_fusion = ProjectedESMCFusion(esmif_dim, esmc_dim, esmif_dim, dropout)
        self.ab_conv1 = GCNConv(esmif_dim, gnn_hidden)
        self.ab_conv2 = GCNConv(gnn_hidden, gnn_out)
        self.ag_conv1 = GCNConv(esmif_dim, gnn_hidden)
        self.ag_conv2 = GCNConv(gnn_hidden, gnn_out)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.cross_block = CrossAttentionBlock(gnn_out, num_heads, ffn_hidden, dropout)
        self.classifier = nn.Sequential(
            nn.Linear(gnn_out, cls_h1),
            nn.LayerNorm(cls_h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(cls_h1, cls_h2),
            nn.LayerNorm(cls_h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(cls_h2, 1),
        )
        # query-conditioned score MLP for the size>1 softmax pool.
        self.score_mlp = nn.Sequential(
            nn.Linear(gnn_out * 2, gnn_out),
            nn.ReLU(inplace=True),
            nn.Linear(gnn_out, 1),
        )
        self.gnn_out = gnn_out

    # -- input adaptation --
    @staticmethod
    def _pyg_batch_to_group_dict(data) -> Dict[str, torch.Tensor]:
        """Adapt a per-complex PyG batch (used at inference) into the
        group-batch dict the forward consumes. Each complex becomes
        its own rep group of size 1; rep-surf-fused space is identical to
        the complex's own antigen surface, so ag_pairs is the identity map
        and ab_pairs comes straight from ``edge_index_contact``.
        """
        device = data.x_ag.device
        # per-graph batch vectors
        x_ab_batch = getattr(data, "x_ab_batch", None)
        x_ag_batch = getattr(data, "x_ag_batch", None)
        if x_ag_batch is None:
            x_ag_batch = torch.zeros(int(data.x_ag.shape[0]),
                                     dtype=torch.long, device=device)
        if x_ab_batch is None:
            x_ab_batch = torch.zeros(int(data.x_ab.shape[0]),
                                     dtype=torch.long, device=device)
        B = int(x_ag_batch.max().item()) + 1 if x_ag_batch.numel() > 0 else 1

        counts_ag = torch.bincount(x_ag_batch, minlength=B)
        counts_ab = torch.bincount(x_ab_batch, minlength=B)
        offsets_ag = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            counts_ag.cumsum(0)[:-1],
        ])
        offsets_ab = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            counts_ab.cumsum(0)[:-1],
        ])

        rep_surf_lens = counts_ag.to(torch.long)
        max_rep = int(rep_surf_lens.max().item()) if rep_surf_lens.numel() else 0

        n_g_total = int(data.x_ag.shape[0])
        arange_g = torch.arange(n_g_total, dtype=torch.long, device=device)
        rep_pos_ag = arange_g - offsets_ag[x_ag_batch]
        ag_pairs = torch.stack([x_ag_batch.long(), rep_pos_ag, arange_g], dim=1)

        ec = getattr(data, "edge_index_contact",
                     torch.zeros(2, 0, dtype=torch.long, device=device))
        if ec.numel() > 0:
            cdr_glob = ec[0].long()
            surf_glob = ec[1].long()
            gi = x_ag_batch[surf_glob]
            rep_pos_ab = surf_glob - offsets_ag[gi]
            ab_pairs = torch.stack([gi, rep_pos_ab, cdr_glob], dim=1)
        else:
            ab_pairs = torch.zeros(0, 3, dtype=torch.long, device=device)

        y = getattr(data, "y_ag_epitope", None)
        y_pad = torch.zeros(B, max_rep, dtype=torch.float32, device=device)
        y_mask = torch.zeros(B, max_rep, dtype=torch.bool, device=device)
        if y is not None and torch.is_tensor(y) and y.numel() > 0:
            y_f = y.float()
            gi = x_ag_batch
            pos = arange_g - offsets_ag[gi]
            y_pad[gi, pos] = y_f
        gi = x_ag_batch
        pos = arange_g - offsets_ag[gi]
        y_mask[gi, pos] = True

        return {
            "x_g": data.x_ag,
            "x_b": data.x_ab,
            "x_g_esmc": data.x_ag_esmc,
            "x_b_esmc": data.x_ab_esmc,
            "edge_index_g": data.edge_index_ag.long(),
            "edge_index_b": data.edge_index_ab.long(),
            "x_b_batch": x_ab_batch.long(),
            "ag_pairs": ag_pairs,
            "ab_pairs": ab_pairs,
            "ab_pairs_self": ab_pairs,
            "ab_pairs_other": torch.zeros(0, 3, dtype=torch.long, device=device),
            "rep_surf_lens": rep_surf_lens,
            "y_g_pad": y_pad,
            "y_mask": y_mask,
        }

    # -- pooling stages --
    def _gcn_and_ag(self, batch: Dict[str, torch.Tensor]):
        x_b = self.ab_fusion(batch["x_b"], batch["x_b_esmc"])
        x_g = self.ag_fusion(batch["x_g"], batch["x_g_esmc"])

        x_b = self.act(self.ab_conv1(x_b, batch["edge_index_b"]))
        x_b = self.dropout(x_b)
        x_b = self.ab_conv2(x_b, batch["edge_index_b"])

        x_g = self.act(self.ag_conv1(x_g, batch["edge_index_g"]))
        x_g = self.dropout(x_g)
        x_g = self.ag_conv2(x_g, batch["edge_index_g"])

        B = int(batch["rep_surf_lens"].shape[0])
        max_rep = int(batch["y_mask"].shape[1])
        D = self.gnn_out
        ag_pairs = batch["ag_pairs"]
        if ag_pairs.numel() > 0:
            ag_vals = x_g[ag_pairs[:, 2]]
        else:
            ag_vals = x_g.new_zeros(0, D)
        ag_q, _ = _scatter_mean_2d(ag_vals, ag_pairs[:, 0], ag_pairs[:, 1],
                                   B, max_rep, D)
        return x_b, ag_q, B, max_rep

    def _ab_pool_split(self, x_b: torch.Tensor, batch: Dict[str, torch.Tensor],
                      ag_q: torch.Tensor, B: int, max_rep: int):
        D = self.gnn_out
        device = x_b.device
        ap = batch["ab_pairs"]

        # size==1 fallback: plain mean of contact CDR residues
        if ap.numel() == 0:
            ab_self = x_b.new_zeros(B, max_rep, D)
            mask_self = torch.zeros(B, max_rep, dtype=torch.bool, device=device)
        else:
            vals = x_b[ap[:, 2]]
            ab_self, mask_self = _scatter_mean_2d(vals, ap[:, 0], ap[:, 1],
                                                  B, max_rep, D)

        # size>1: query-conditioned segment softmax pool
        if ap.numel() == 0:
            ab_soft = x_b.new_zeros(B, max_rep, D)
            mask_soft = torch.zeros(B, max_rep, dtype=torch.bool, device=device)
        else:
            c_t = x_b[ap[:, 2]]
            q_t = ag_q[ap[:, 0], ap[:, 1]]
            score = self.score_mlp(torch.cat([q_t, c_t], dim=-1)).squeeze(-1)
            ab_soft, mask_soft = _segment_softmax_pool(
                c_t, ap[:, 0], ap[:, 1], score, B, max_rep, D,
            )

        ap_other = batch.get(
            "ab_pairs_other",
            torch.zeros(0, 3, dtype=torch.long, device=device),
        )
        is_multi = torch.zeros(B, dtype=torch.bool, device=device)
        if ap_other.numel() > 0:
            is_multi[ap_other[:, 0].unique()] = True
        sel = is_multi.view(B, 1, 1)
        ab_kv = torch.where(sel, ab_soft, ab_self)
        sel_m = is_multi.view(B, 1)
        mask_out = torch.where(sel_m, mask_soft, mask_self)
        return ab_kv, mask_out

    def forward(
        self,
        data: Any,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Predict per-rep-fused-surface-residue logits.

        ``data`` may be either a ``collate_groups`` dict (training / export)
        or a per-complex PyG ``Data`` (inference). The result dict layout is
        stable across both paths and matches therelease contract:

            {'logits': (M,), 'y_true': (M,), 'batch_g': (M,) long,
             (optional) 'after_qkv_emb_256': (M, gnn_out)}
        """
        if isinstance(data, dict):
            batch = data
        else:
            batch = self._pyg_batch_to_group_dict(data)

        x_b, ag_q, B, max_rep = self._gcn_and_ag(batch)
        ab_kv, ab_kv_mask = self._ab_pool_split(x_b, batch, ag_q, B, max_rep)
        ab_kpm = ~ab_kv_mask
        all_invalid = ab_kpm.all(dim=1)
        if all_invalid.any():
            ab_kpm[all_invalid, 0] = False

        z_g = self.cross_block(ag_q, ab_kv, key_padding_mask=ab_kpm)
        y_mask = batch["y_mask"]
        logits_full = self.classifier(z_g).squeeze(-1)
        logits = logits_full[y_mask]
        y_true = batch["y_g_pad"][y_mask].float()
        gi_grid = torch.arange(B, device=y_mask.device).unsqueeze(1).expand_as(y_mask)
        batch_g = gi_grid[y_mask]

        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "y_true": y_true,
            "batch_g": batch_g,
        }
        if return_embeddings:
            out["after_qkv_emb_256"] = z_g[y_mask]
        return out
