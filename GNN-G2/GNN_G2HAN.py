"""
G2-HAN (Heterogeneous Attention Network) — model definition + inference helper.

Expected node types:
  - "URL"
  - "HAR"
  - "Registrar"

Expected edge types (must match training metadata):
  - ("Registrar", "registers", "URL")
  - ("URL", "linked_to", "HAR")
  - ("HAR", "linked_to", "URL")
"""

from __future__ import annotations

from typing import Tuple, Optional, Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import HANConv

__all__ = ["HANModel", "infer_urls", "append_subgraph_to_base"]


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
    Returns:
      - merged: combined HeteroData graph
      - url_offset: starting global index of newly appended URL nodes
    """
    merged = base.clone()

    url_offset = merged["URL"].num_nodes
    har_offset = merged["HAR"].num_nodes
    reg_offset = merged["Registrar"].num_nodes

    # Append nodes
    merged["URL"].x = torch.cat([merged["URL"].x, new_url_x], dim=0)
    merged["HAR"].x = torch.cat([merged["HAR"].x, new_har_x], dim=0)

    # Optional: dummy registrar node connected to all new URLs
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

    # Remap local -> global edges
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

    # Edge attributes (optional)
    if edge_attr is not None:
        for etype in [("URL", "linked_to", "HAR"), ("HAR", "linked_to", "URL")]:
            if "edge_attr" in merged[etype]:
                merged[etype].edge_attr = torch.cat([merged[etype].edge_attr, edge_attr], dim=0)
            else:
                merged[etype].edge_attr = edge_attr

    return merged, url_offset


class HANModel(torch.nn.Module):
    """
    3-layer HANConv model for heterogeneous graphs.

    Notes:
    - HANConv outputs per-node-type embeddings.
    - We return logits for URL nodes only (binary classification by default).
    - If Registrar has a different input dimension than URL/HAR, we optionally
    """
    def __init__(
        self,
        metadata,
        in_channels_dict: Dict[str, int],
        hidden_channels: int = 64,
        out_channels: int = 2,
        heads: int = 8,
        keep_registrar_static: bool = True,
    ):
        super().__init__()
        self.keep_registrar_static = keep_registrar_static
        self.hidden_channels = hidden_channels
        self.heads = heads

        self.conv1 = HANConv(
            in_channels=in_channels_dict,
            out_channels=hidden_channels,
            metadata=metadata,
            heads=heads,
        )

        hidden_dict_1 = {k: hidden_channels * heads for k in in_channels_dict.keys()}

        self.conv2 = HANConv(
            in_channels=hidden_dict_1,
            out_channels=hidden_channels,
            metadata=metadata,
            heads=heads,
        )

        hidden_dict_2 = {k: hidden_channels * heads for k in in_channels_dict.keys()}

        self.conv3 = HANConv(
            in_channels=hidden_dict_2,
            out_channels=out_channels,
            metadata=metadata,
            heads=1,
        )

    def forward(self, x_dict, edge_index_dict):
        # Preserve Registrar features if requested (e.g., dummy registrar node)
        registrar_x_orig = x_dict.get("Registrar", None)

        #transform Registrar input to hidden size for stable propagation
        registrar_x_fixed = None
        if registrar_x_orig is not None and self.registrar_transform is not None:
            registrar_x_fixed = self.registrar_transform(registrar_x_orig)

        # ---- Layer 1 ----
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.elu(v) for k, v in x_dict.items()}

        if self.keep_registrar_static and registrar_x_fixed is not None:
            # After conv1, Registrar would normally be hidden*heads;
            # we inject a stable representation instead if desired.
            x_dict["Registrar"] = registrar_x_fixed.repeat(1, self.heads)

        # ---- Layer 2 ----
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.elu(v) for k, v in x_dict.items()}

        if self.keep_registrar_static and registrar_x_fixed is not None:
            x_dict["Registrar"] = registrar_x_fixed.repeat(1, self.heads)

        # ---- Layer 3 (logits) ----
        x_dict = self.conv3(x_dict, edge_index_dict)

        # Restore original Registrar 
        if self.keep_registrar_static and registrar_x_orig is not None:
            x_dict["Registrar"] = registrar_x_orig

        # Return URL logits only (shape: [num_URL, out_channels])
        return x_dict["URL"]


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
