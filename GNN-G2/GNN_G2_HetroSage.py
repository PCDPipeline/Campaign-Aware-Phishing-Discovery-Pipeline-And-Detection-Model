"""
G2 Heterogeneous GraphSAGE (3-layer) — model definition + inference helper.

This file is intentionally dataset-agnostic:
- We assume the user already has:
  (1) a base hetero graph (PyG HeteroData) compatible with training schema
  (2) a trained model checkpoint
  (3) feature tensors constructed using the provided frozen vocab ordering

Node types expected:
  - "URL"
  - "HAR"
  - "Registrar"

Edge types expected:
  - ("Registrar", "registers", "URL")
  - ("URL", "linked_to", "HAR")
  - ("HAR", "linked_to", "URL")
"""

from __future__ import annotations

from typing import Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

__all__ = ["HeteroGNN", "infer_urls", "append_subgraph_to_base"]


def append_subgraph_to_base(
    base: Any,
    new_url_x: torch.Tensor,                    # [N_new, D_url]
    new_har_x: torch.Tensor,                    # [M_new, D_har]
    edge_url_har: torch.Tensor,                 # [2, E] local indices
    edge_har_url: torch.Tensor,                 # [2, E] local indices
    edge_attr: Optional[torch.Tensor] = None,   # [E, 1] (optional)
    add_dummy_registrar: bool = True,
) -> Tuple[Any, int]:
    """
    Append user-provided URL/HAR nodes and edges to the provided base graph.

    Inputs:
      - base: torch_geometric.data.HeteroData (the released training/base graph)
      - new_url_x: URL features for user nodes (N_new x D_url)
      - new_har_x: HAR features for user nodes (M_new x D_har)
      - edge_url_har: edges (URL -> HAR) using *local* indexing:
            URL ids in [0..N_new-1], HAR ids in [0..M_new-1]
      - edge_har_url: edges (HAR -> URL) using *local* indexing
      - edge_attr: optional edge attributes aligned with E edges (E x 1)
      - add_dummy_registrar: if True, inject a single all-zero Registrar node and connect it
            to every new URL node (useful when registrar info is unavailable for user data).

    Returns:
      - merged: combined HeteroData graph
      - url_offset: starting global index of the newly appended URL nodes
    """
    merged = base.clone()

    url_offset = merged["URL"].num_nodes
    har_offset = merged["HAR"].num_nodes
    reg_offset = merged["Registrar"].num_nodes

    merged["URL"].x = torch.cat([merged["URL"].x, new_url_x], dim=0)
    merged["HAR"].x = torch.cat([merged["HAR"].x, new_har_x], dim=0)

    if add_dummy_registrar:
        reg_dim = merged["Registrar"].x.size(1)
        dummy_reg_x = torch.zeros((1, reg_dim), dtype=merged["Registrar"].x.dtype)
        merged["Registrar"].x = torch.cat([merged["Registrar"].x, dummy_reg_x], dim=0)

        dummy_reg_id = reg_offset
        new_url_global = torch.arange(url_offset, url_offset + new_url_x.size(0), dtype=torch.long)

        reg_src = torch.full((new_url_global.numel(),), dummy_reg_id, dtype=torch.long)
        reg_dst = new_url_global
        reg_edge = torch.stack([reg_src, reg_dst], dim=0)  # [2, N_new]

        merged[("Registrar", "registers", "URL")].edge_index = torch.cat(
            [merged[("Registrar", "registers", "URL")].edge_index, reg_edge],
            dim=1,
        )

    e_url_har = edge_url_har.clone()
    e_har_url = edge_har_url.clone()

    e_url_har[0] += url_offset
    e_url_har[1] += har_offset
    e_har_url[0] += har_offset
    e_har_url[1] += url_offset

    merged[("URL", "linked_to", "HAR")].edge_index = torch.cat(
        [merged[("URL", "linked_to", "HAR")].edge_index, e_url_har],
        dim=1,
    )
    merged[("HAR", "linked_to", "URL")].edge_index = torch.cat(
        [merged[("HAR", "linked_to", "URL")].edge_index, e_har_url],
        dim=1,
    )

    if edge_attr is not None:
        for etype in [("URL", "linked_to", "HAR"), ("HAR", "linked_to", "URL")]:
            if "edge_attr" in merged[etype]:
                merged[etype].edge_attr = torch.cat([merged[etype].edge_attr, edge_attr], dim=0)
            else:
                merged[etype].edge_attr = edge_attr

    return merged, url_offset


class HeteroGNN(torch.nn.Module):
    """3-layer HeteroConv GraphSAGE model (G2). Must match training architecture."""
    def __init__(self, hidden_channels: int = 64, out_channels: int = 2):
        super().__init__()

        conv_dict = {
            ("Registrar", "registers", "URL"): SAGEConv((-1, -1), hidden_channels),
            ("URL", "linked_to", "HAR"): SAGEConv((-1, -1), hidden_channels),
            ("HAR", "linked_to", "URL"): SAGEConv((-1, -1), hidden_channels),
        }

        self.conv1 = HeteroConv(conv_dict, aggr="mean")
        self.conv2 = HeteroConv(conv_dict, aggr="mean")
        self.conv3 = HeteroConv(conv_dict, aggr="mean")
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        registrar_x = x_dict["Registrar"]

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict["Registrar"] = registrar_x

        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict["Registrar"] = registrar_x

        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict["Registrar"] = registrar_x

        return self.lin(x_dict["URL"])


@torch.no_grad()
def infer_urls(
    data: Any,
    model: torch.nn.Module,
    url_node_ids: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on URL nodes.

    Returns:
        pred: ndarray of predicted labels (0/1)
        p_phish: ndarray of phishing probabilities (P(y=1))
    """
    model.eval()
    logits = model(data.x_dict, data.edge_index_dict)  # [num_URL, 2]
    probs = F.softmax(logits, dim=1)

    probs_sel = probs if url_node_ids is None else probs[url_node_ids]
    pred = probs_sel.argmax(dim=1).cpu().numpy()
    p_phish = probs_sel[:, 1].cpu().numpy()
    return pred, p_phish
