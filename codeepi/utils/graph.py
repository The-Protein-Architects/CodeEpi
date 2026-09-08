"""Graph feature helpers.

Two conventions used across the codebase, matching the release dataset:

* 16-D Gaussian RBF expansion of pairwise Ca distances (Jing et al. 2019).
* 16-D sinusoidal positional encoding on ``edge_index[0] - edge_index[1]``.

"""
from __future__ import annotations
import math
from typing import Optional

import torch
from torch_geometric.nn import radius_graph


def rbf(
    D: torch.Tensor,
    D_min: float = 0.0,
    D_max: float = 20.0,
    D_count: int = 16,
) -> torch.Tensor:
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = D.unsqueeze(-1)
    return torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)


def get_posenc(
    edge_index: torch.Tensor,
    num_posenc: int = 16,
) -> torch.Tensor:
    d = (edge_index[0] - edge_index[1]).float()
    frequency = torch.exp(
        torch.arange(0, num_posenc, 2, dtype=torch.float32, device=d.device)
        * -(math.log(10000.0) / num_posenc)
    )
    angles = d.unsqueeze(-1) * frequency
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)


def build_egnn_edges(
    ca_coords: torch.Tensor,
    r: float = 8.0,
    max_num_neighbors: int = 64,
    num_rbf: int = 16,
    num_posenc: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the student EGNN graph from Ca coordinates.

    Returns ``(edge_index, edge_attr)`` where ``edge_attr`` is
    ``[E, num_rbf + num_posenc]``. Matches the ``Antigen1321`` release edges.
    """
    edge_index = radius_graph(
        ca_coords, r=r, loop=False, max_num_neighbors=max_num_neighbors
    )
    if edge_index.shape[1] == 0:
        edge_attr = ca_coords.new_zeros(0, num_rbf + num_posenc)
        return edge_index, edge_attr
    diff = ca_coords[edge_index[0]] - ca_coords[edge_index[1]]
    dist = diff.norm(dim=-1)
    edge_rbf = rbf(dist, D_count=num_rbf)
    edge_posenc = get_posenc(edge_index, num_posenc=num_posenc)
    edge_attr = torch.cat([edge_rbf, edge_posenc], dim=-1)
    return edge_index, edge_attr
