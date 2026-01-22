# graph_utils.py
import os
import networkx as nx
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def sif_to_nx(sif_file: str) -> nx.DiGraph:
    "Changing a SIF file into a NetworkX DiGraph."

    G = nx.DiGraph()
    with open(sif_file) as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split()
            if len(parts) < 3: continue  # skip malformed lines
            source, _, target = parts
            G.add_edge(source, target)
    return G


def jaccard_index(G_true: nx.DiGraph, G_pred: nx.DiGraph) -> float:
    """Calculates the Jaccard Index between the edge sets of two graphs.
    It is a fraction of correctly predicted edges"""

    E_true = set(G_true.edges())
    E_pred = set(G_pred.edges())
    if not E_true and not E_pred: return 1.0
    return len(E_true & E_pred) / len(E_true | E_pred)


def precision_recall_f1(G_true: nx.DiGraph, G_pred: nx.DiGraph) -> tuple[float, float, float]:
    """Calculates precision, recall, and F1-score between the edge sets of two graphs"""
    nodes = list(G_true.nodes())

    A_true = nx.to_numpy_array(G_true, nodelist=nodes)

    # Make a copy so you don't mutate the original graph
    G_pred_aligned = G_pred.copy()
    G_pred_aligned.add_nodes_from(nodes)

    A_pred = nx.to_numpy_array(G_pred_aligned, nodelist=nodes)

    # Flatten for sklearn
    y_true = A_true.flatten()
    y_pred = A_pred.flatten()

    # Handle the case where y_pred has no positives
    if y_pred.sum() == 0 and y_true.sum() == 0:
        return 1.0, 1.0, 1.0  # perfect match: no edges predicted, none in truth
    else:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return precision, recall, f1
