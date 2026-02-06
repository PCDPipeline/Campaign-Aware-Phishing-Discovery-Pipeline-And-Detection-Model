
import torch

from GNN_G2_HetroSage import HeteroGNN, infer_urls, append_subgraph_to_base


BASE_GRAPH_PATH = "G2.pt"
MODEL_PATH = "G2_Heterogeneous_SAGEConv(3_layer).pt"


def main():
    base = torch.load(BASE_GRAPH_PATH, weights_only=False)

    model = HeteroGNN(hidden_channels=64, out_channels=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # ------------------------------------------------------------------
    # USER MUST PROVIDE THESE (constructed using vocab.json ordering)
    # ------------------------------------------------------------------
    new_url_x = None        # torch.Tensor [N_new, D_url]
    new_har_x = None        # torch.Tensor [M_new, D_har]
    edge_url_har = None     # torch.Tensor [2, E] local indices (URL->HAR)
    edge_har_url = None     # torch.Tensor [2, E] local indices (HAR->URL)
    edge_attr = None        # torch.Tensor [E, 1] optional

    if any(v is None for v in [new_url_x, new_har_x, edge_url_har, edge_har_url]):
        raise ValueError(
            "You must construct: new_url_x, new_har_x, edge_url_har, edge_har_url "
            "(and optionally edge_attr) before running this script."
        )

    merged, url_offset = append_subgraph_to_base(
        base,
        new_url_x,
        new_har_x,
        edge_url_har,
        edge_har_url,
        edge_attr=edge_attr,
        add_dummy_registrar=True,
    )

    new_url_ids = torch.arange(url_offset, url_offset + new_url_x.size(0), dtype=torch.long)
    pred, p_phish = infer_urls(merged, model, url_node_ids=new_url_ids)

    print("Pred shape:", pred.shape)
    print("p_phish shape:", p_phish.shape)
    print("First 10 preds:", pred[:10])
    print("First 10 p_phish:", p_phish[:10])


if __name__ == "__main__":
    main()
