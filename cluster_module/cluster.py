from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from core.logger import logger
import core.utils as utils


# ---------------------------------------------------------------------------
# Runtime-configurable thresholds (set by caller / CLI / config file)
# ---------------------------------------------------------------------------
# Leave these as None so experiments can set them explicitly at runtime.
# Example:
#   config = ClusteringConfig(
#       distance_threshold=DISTANCE_THRESHOLD,
#       min_degree=MIN_DEGREE,
#       min_fallback_path_len=MIN_FALLBACK_PATH_LEN,
#   )
DISTANCE_THRESHOLD: Optional[float] = None        # e.g., 8.0
MIN_DEGREE: Optional[int] = None                  # e.g., 2
MIN_FALLBACK_PATH_LEN: Optional[int] = None       # e.g., 9


@dataclass(frozen=True)
class ClusteringConfig:
    """
    Configuration for content/HAR-based clustering.

    All thresholds are intentionally runtime-configurable:
    set them via the caller, CLI flags, environment variables, or an external config file.
    """
    distance_threshold: Optional[float] = None
    min_degree: Optional[int] = None
    min_fallback_path_len: Optional[int] = None


def _resolve_config(config: ClusteringConfig) -> Tuple[float, int, int]:
    """
    Resolve thresholds from:
      1) explicit ClusteringConfig values
      2) module-level variables (set at runtime)
    Raises if any required threshold is missing.
    """
    distance_threshold = config.distance_threshold if config.distance_threshold is not None else DISTANCE_THRESHOLD
    min_degree = config.min_degree if config.min_degree is not None else MIN_DEGREE
    min_fallback_path_len = (
        config.min_fallback_path_len if config.min_fallback_path_len is not None else MIN_FALLBACK_PATH_LEN
    )

    missing = []
    if distance_threshold is None:
        missing.append("distance_threshold (DISTANCE_THRESHOLD)")
    if min_degree is None:
        missing.append("min_degree (MIN_DEGREE)")
    if min_fallback_path_len is None:
        missing.append("min_fallback_path_len (MIN_FALLBACK_PATH_LEN)")

    if missing:
        raise ValueError(
            "Clustering thresholds are not set. Please set them at runtime via ClusteringConfig or module variables: "
            + ", ".join(missing)
        )

    return float(distance_threshold), int(min_degree), int(min_fallback_path_len)


def make_clusters(data: pd.DataFrame, config: ClusteringConfig = ClusteringConfig()) -> pd.DataFrame:
    """
    End-to-end clustering:
      1) Explode HAR logs and content_files to map requests to content (md5) where possible.
      2) Build per-sid set of identifiers (md5-derived har_id or fallback path-derived har_id).
      3) Compute pairwise distances between sids.
      4) Build graph edges based on distance threshold; prune weak nodes; use connected components as clusters.

    Returns a filtered DataFrame containing only clustered sids, with a 'cluster' column.
    """
    df = data.copy()
    df["cluster"] = -1

    try:
        dist_thr, min_deg, min_path_len = _resolve_config(config)
        logger.info(
            "Beginning clustering with thresholds: "
            f"distance_threshold={dist_thr}, min_degree={min_deg}, min_fallback_path_len={min_path_len}"
        )

        exploded = _expand_records(df)
        distance_df, df_filtered = _create_distance_matrix(
            df, exploded, distance_threshold=dist_thr, min_fallback_path_len=min_path_len
        )
        _, clustered_df = _cluster_via_graph(distance_df, df_filtered, distance_threshold=dist_thr, min_degree=min_deg)

        clustered_df.to_csv("data.csv", index=False)
        logger.info("Clustering finished.")
        return clustered_df

    except Exception as e:
        logger.exception(f"FAILED to cluster: {e}")
        return df


def _expand_records(data: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a request-level DataFrame by exploding:
      - har_logs (per request): expects dicts with keys like id, url, status
      - content_files: expects dicts with keys like md5, requestId

    Output includes:
      - sid, root_url, request_id, request_url, response_status, md5, request_url_path
    """
    if "har_logs" not in data.columns or "content_files" not in data.columns:
        raise KeyError("Input DataFrame must include 'har_logs' and 'content_files' columns.")

    har_exploded = data.explode("har_logs", ignore_index=True)
    har_df = har_exploded["har_logs"].apply(pd.Series).rename(
        columns={"id": "request_id", "url": "request_url", "status": "response_status"}
    )
    har_exploded = pd.concat([har_exploded.drop(columns=["har_logs"]), har_df], axis=1)

    content_exploded = data[["sid", "content_files"]].explode("content_files", ignore_index=True)
    content_df = content_exploded["content_files"].apply(pd.Series)[["md5", "requestId"]]
    content_exploded = pd.concat([content_exploded[["sid"]], content_df], axis=1)

    har_exploded["request_id"] = har_exploded["request_id"].astype(str)
    content_exploded["requestId"] = content_exploded["requestId"].astype(str)

    merged = har_exploded.merge(
        content_exploded,
        left_on=["sid", "request_id"],
        right_on=["sid", "requestId"],
        how="left",
    ).drop(columns=["requestId", "content_files"], errors="ignore")

    merged["request_url_path"] = merged["request_url"].apply(
        lambda x: utils.extract_full_path(x) if isinstance(x, str) else "/"
    )
    return merged


def _create_distance_matrix(
    data: pd.DataFrame,
    exploded: pd.DataFrame,
    *,
    distance_threshold: float,
    min_fallback_path_len: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds a symmetric distance matrix between valid sids based on shared identifiers.

    Identifier strategy:
      - Primary: md5 (content) mapped into integer ids via factorization
      - Fallback: request_url_path (for requests without md5), only if path length >= min_fallback_path_len
    """
    exp = exploded.copy()

    exp["har_id"] = pd.factorize(exp["md5"])[0]

    missing_mask = exp["har_id"] == -1
    if missing_mask.any():
        start_id = int(exp["har_id"].max()) + 1
        eligible_paths = exp.loc[
            missing_mask & (exp["request_url_path"].astype(str).str.len() >= min_fallback_path_len),
            "request_url_path",
        ].dropna().unique()

        fallback_map = {path: start_id + i for i, path in enumerate(eligible_paths)}
        exp.loc[missing_mask, "har_id"] = exp.loc[missing_mask, "request_url_path"].map(fallback_map)

    exp = exp[exp["har_id"].notna() & (exp["har_id"] != -1)].copy()

    exp["root_domain"] = exp["root_url"].apply(utils.get_registered_domain)
    exp["request_domain"] = exp["request_url"].apply(utils.get_registered_domain)
    exp = exp[exp["root_domain"] == exp["request_domain"]].reset_index(drop=True)
    exp = exp.drop(columns=["root_domain", "request_domain"], errors="ignore")

    valid_sids = exp["sid"].dropna().unique()
    if len(valid_sids) == 0:
        raise ValueError("No valid sids remain after filtering (domain match + har_id assignment).")

    sid_to_ids: Dict[str, Set[int]] = exp.groupby("sid")["har_id"].apply(lambda s: set(map(int, s))).to_dict()

    n = len(valid_sids)
    dist = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            d = _distance_from_sets(sid_to_ids, valid_sids[i], valid_sids[j])
            dist[i, j] = d
            dist[j, i] = d

    df_filtered = data[data["sid"].isin(valid_sids)].copy().reset_index(drop=True)
    df_filtered = df_filtered.drop(columns=["content_files", "har_logs"], errors="ignore")

    distance_df = pd.DataFrame(dist, index=valid_sids, columns=valid_sids)
    return distance_df, df_filtered


def _distance_from_sets(
    sid_to_ids: Dict[str, Set[int]], sid1: str, sid2: str, d_min: float = 1.0, d_max: float = 10.0
) -> float:
    """
    Distance in [1,10] computed from overlap:
      distance = 10 - 9 * (|A ∩ B| / max(|A|, |B|, 1))
    """
    a = sid_to_ids.get(sid1, set())
    b = sid_to_ids.get(sid2, set())

    common = len(a.intersection(b))
    denom = max(len(a), len(b), 1)

    d = d_max - (d_max - d_min) * (common / denom)
    return float(max(d_min, min(d_max, d)))


def _cluster_via_graph(
    distance_df: pd.DataFrame,
    data: pd.DataFrame,
    *,
    distance_threshold: float,
    min_degree: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs a graph over sids using the distance matrix, prunes weak nodes,
    and assigns cluster ids based on connected components.
    """
    sids = list(distance_df.index)
    mat = distance_df.values

    G = nx.Graph()
    G.add_nodes_from(sids)

    n = len(sids)
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i, j] < distance_threshold:
                G.add_edge(sids[i], sids[j])

    weak = [node for node, deg in G.degree() if deg < min_degree]
    if weak:
        G.remove_nodes_from(weak)
    logger.info(f"Removed {len(weak)} weakly connected nodes (degree < {min_degree}).")

    components = list(nx.connected_components(G))
    cluster_map: Dict[str, int] = {}
    for cid, comp in enumerate(components):
        for sid in comp:
            cluster_map[str(sid)] = cid

    clustered_sids = set(cluster_map.keys())
    clustered_data = data[data["sid"].astype(str).isin(clustered_sids)].copy().reset_index(drop=True)
    clustered_data["cluster"] = clustered_data["sid"].astype(str).map(cluster_map)

    distance_df = distance_df.copy()
    distance_df["cluster"] = distance_df.index.astype(str).map(cluster_map)

    logger.info(f"Formed {len(components)} clusters from {len(clustered_sids)} sids.")
    return distance_df, clustered_data
