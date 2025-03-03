"""Module providing PyGenStabilityOne and its helper functions."""

import copy
import numpy as np
import pandas as pd
import networkx as nx
from joblib import load
from pygenstability import run
from karateclub.graph_embedding import Graph2Vec

_STRATEGIES = {"GradientBoosting"}


def _labels_to_coms(labels):
    coms_dict = {key: [] for key in set(labels)}
    for i, v in enumerate(labels):
        coms_dict[v].append(i)
    return list(coms_dict.values())


def _get_features(g):
    results_df = pd.DataFrame()

    g_copy = copy.deepcopy(g)

    for min_count_val in range(5, -1, -1):
        try:
            g2v = Graph2Vec(seed=42, dimensions=256, min_count=min_count_val)
            g2v.fit([g_copy])
            break
        except RuntimeError:
            pass
    emb = g2v.get_embedding()[0]

    emb_arr = np.array([emb])
    results_df["emb_mean"] = np.mean(emb_arr, axis=1)
    results_df["emb_median"] = np.median(emb_arr, axis=1)
    results_df["emb_std"] = np.std(emb_arr, axis=1)
    results_df["emb_min"] = np.min(emb_arr, axis=1)
    results_df["emb_max"] = np.max(emb_arr, axis=1)

    extra_features = {
        "global_efficiency": nx.global_efficiency(g),
        "n": len(g.nodes()),
        "m": len(g.edges()),
        "assortativity": nx.degree_assortativity_coefficient(g),
        "average_clustering": nx.average_clustering(g),
        "average_degree": 2 * len(g.edges()) / len(g.nodes()),
    }

    for feature, feature_val in extra_features.items():
        results_df.loc[0, feature] = feature_val

    return results_df.fillna(0)


def _find_coms(
    g: nx.Graph,
    strategy: str = "GradientBoosting",
    min_scale: int = -1,
    max_scale: int = 0.75,
    n_scale: int = 30,
    constructor: str = "continuous_normalized",
):
    results = run(
        nx.adjacency_matrix(g),
        min_scale=min_scale,
        max_scale=max_scale,
        n_scale=n_scale,
        constructor=constructor,
        tqdm_disable=True,
    )

    if len(results["selected_partitions"]) == 1:
        return _labels_to_coms(
            results["community_id"][results["selected_partitions"][0]]
        )

    if strategy not in _STRATEGIES:
        strategy = "GradientBoosting"

    scale_estimator = load(f"estimators/{strategy}.joblib")
    ml_scale = scale_estimator.predict(_get_features(g))[0]
    closest_scale_idx = min(
            results["selected_partitions"],
            key=lambda scale_idx: abs(ml_scale - results["scales"][scale_idx]),
            )

    return _labels_to_coms(results["community_id"][closest_scale_idx])


def _get_connected(g: nx.Graph):
    g_copy = copy.deepcopy(g)
    used = set()

    while not nx.is_connected(g_copy):
        comps = sorted(({node for node in component if node not in used}
                        for component in nx.connected_components(g_copy)),
                        key=len, reverse=True)

        if len(comps[0]) == 0 or len(comps[1]) == 0:
            comps = sorted(nx.connected_components(g_copy), key=len, reverse=True)

        n0 = max(comps[0], key=lambda c: g_copy.degree[c])
        n1 = max(comps[1], key=lambda c: g_copy.degree[c])
        g_copy.add_edge(n0, n1)
        used.add(n0)
        used.add(n1)

    return g_copy


def pygenstability_one(
    g: nx.Graph,
    strategy: str = "GradientBoosting",
    min_scale: int = -1,
    max_scale: int = 0.75,
    n_scale: int = 30,
    constructor: str = "continuous_normalized",
):
    """A community detection approach that uses machine learning to choose the best
    scale from those selected by PyGenStability's optimal scale selection method."""

    if nx.is_connected(g):
        return _find_coms(g, strategy, min_scale, max_scale, n_scale, constructor)

    for _ in range(5):
        try:
            return _find_coms(_get_connected(g),
                              strategy, min_scale, max_scale, n_scale, constructor)
        except:
            pass
