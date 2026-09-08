from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.nn import InstanceNorm


# --------------------------------------------------------------------------
# Backbone dihedral features
# --------------------------------------------------------------------------
class _NormalizeLastDim(nn.Module):
    def __init__(self, features: int, epsilon: float = 1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim=True)
        sigma = torch.sqrt(x.var(dim=-1, keepdim=True) + self.epsilon)
        return self.gain * (x - mu) / (sigma + self.epsilon) + self.bias


class DihedralFeatures(nn.Module):

    def __init__(self, node_embed_dim: int):
        super().__init__()
        self.node_embedding = nn.Linear(6, node_embed_dim, bias=True)
        self.norm_nodes = _NormalizeLastDim(node_embed_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            V = self._dihedrals(X)
        V = V.squeeze(1)
        V = self.node_embedding(V)
        V = self.norm_nodes(V)
        return V

    @staticmethod
    def _dihedrals(X: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        if X.dim() == 4:
            X = X[..., :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)
        else:
            X = X[:, :3, :]

        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        n_2 = F.normalize(torch.cross(u_2, u_1, dim=-1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0, dim=-1), dim=-1)

        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        D = F.pad(D, (1, 2), "constant", 0)
        D = D.view(D.size(0), D.size(1) // 3, 3)
        return torch.cat((torch.cos(D), torch.sin(D)), dim=2)


# --------------------------------------------------------------------------
# EGNN
# --------------------------------------------------------------------------
class EGNNLayer(nn.Module):
    def __init__(
        self,
        input_nf: int,
        output_nf: int,
        hidden_nf: int,
        edges_in_d: int = 0,
        act_fn: nn.Module | None = None,
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        coords_agg: str = "mean",
        tanh: bool = False,
        dropout: float = 0.0,
        ffn: bool = False,
        batch_norm: bool = True,
    ):
        super().__init__()
        if act_fn is None:
            act_fn = nn.SiLU()
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.ffn = ffn
        self.batch_norm = batch_norm

        in_edge = input_nf * 2 + 1 + edges_in_d
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_edge, hidden_nf), act_fn, nn.Dropout(dropout),
            nn.Linear(hidden_nf, hidden_nf), act_fn, nn.Dropout(dropout),
        )
        if attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

        coord_layer = nn.Linear(hidden_nf, 1, bias=False)
        nn.init.xavier_uniform_(coord_layer.weight, gain=0.001)
        coord_blocks = [
            nn.Linear(hidden_nf, hidden_nf), act_fn, nn.Dropout(dropout),
            coord_layer,
        ]
        if tanh:
            coord_blocks.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_blocks)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn, nn.Dropout(dropout),
            nn.Linear(hidden_nf, output_nf),
        )

        if batch_norm:
            self.norm_node = InstanceNorm(output_nf, affine=True)
            self.norm_coord = InstanceNorm(3, affine=True)

        if ffn:
            self.ff1 = nn.Linear(output_nf, output_nf * 2)
            self.ff2 = nn.Linear(output_nf * 2, output_nf)
            self.act_ff = act_fn
            self.drop_ff = nn.Dropout(dropout)
            if batch_norm:
                self.norm_ff1 = InstanceNorm(output_nf, affine=True)
                self.norm_ff2 = InstanceNorm(output_nf, affine=True)

    def _coord2radial(self, edge_index, coord):
        row, col = edge_index
        diff = coord[row] - coord[col]
        dist2 = (diff ** 2).sum(dim=-1, keepdim=True)
        dist2 = torch.clamp(dist2, min=self.epsilon, max=100.0)
        if self.normalize:
            norm = (dist2.sqrt().detach() + self.epsilon)
            diff = diff / norm
            diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
        return dist2, diff

    def _ff_block(self, x):
        return self.ff2(self.drop_ff(self.act_ff(self.ff1(x))))

    def forward(self, h, coord, edge_index, batch, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self._coord2radial(edge_index, coord)

        e_in = [h[row], h[col], radial]
        if edge_attr is not None:
            e_in.append(edge_attr)
        e = torch.cat(e_in, dim=-1)
        e = self.edge_mlp(e)
        if self.attention:
            att = self.att_mlp(e)
            e = e * att

        coord_update = torch.clamp(self.coord_mlp(e), -1.0, 1.0)
        trans = coord_diff * coord_update
        trans = torch.where(torch.isfinite(trans), trans, torch.zeros_like(trans))
        agg_coord = scatter(trans, row, dim=0, dim_size=coord.size(0), reduce=self.coords_agg)
        coord = coord + agg_coord
        coord = torch.where(torch.isfinite(coord), coord, torch.zeros_like(coord))
        if self.batch_norm:
            coord = self.norm_coord(coord, batch)

        agg_node = scatter(e, row, dim=0, dim_size=h.size(0), reduce="sum")
        x_in = torch.cat([h, agg_node], dim=-1)
        if node_attr is not None:
            x_in = torch.cat([x_in, node_attr], dim=-1)
        h_new = self.node_mlp(x_in)
        if self.batch_norm:
            h_new = self.norm_node(h_new, batch)
        if self.residual and h_new.shape[-1] == h.shape[-1]:
            h_new = h + h_new

        if self.ffn:
            if self.batch_norm:
                h_new = self.norm_ff1(h_new, batch)
            h_new = h_new + self._ff_block(h_new)
            if self.batch_norm:
                h_new = self.norm_ff2(h_new, batch)

        return h_new, coord, e
