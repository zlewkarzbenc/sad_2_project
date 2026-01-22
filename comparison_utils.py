# comparison_utils.py
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from graph_utils import jaccard_index, precision_recall_f1, sif_to_nx
from bn import *

"""
Utilities for evaluating reconstructed Boolean networks by comparing predicted
graphs against ground‑truth networks.
"""

def draw_compare_bn(G_true: nx.DiGraph, G_pred: nx.DiGraph, 
                    path: str, filename: str, show=False)->None:
    """
    Draws a comparison plot between the predicted and ground‑truth Boolean networks and saves a PNG file with the comparison plot.
    Args:
    G_true : nx.DiGraph
        Ground‑truth network as a directed graph.
    G_pred : nx.DiGraph
        Predicted network reconstructed by BNFinder.
    path : str
        Path to the directory containing the .sif file and where output graphs will be saved.
    filename : str
        Base name of the dataset (used for saving the plot).
    show : bool, optional
        Whether to display the plot interactively. Default is False.
    """

    
    G_combined = nx.compose(G_pred, G_true)  # includes all nodes from both
    pos = nx.kamada_kawai_layout(G_combined)

    nx.draw(G_pred, pos, with_labels=True, node_color="lightblue", arrowsize=20)
    nx.draw(G_true, pos, with_labels=True, edge_color='green', style='dashed')

    plt.title(f"Predicted network vs Ground truth edges (green) - {filename}")
    figure_path = path + "_graphs"
    os.makedirs(figure_path, exist_ok=True)
    plt.savefig(os.path.join(figure_path, f"{filename}_comparison.png"), bbox_inches="tight")
    if show: plt.show()
    plt.close()



def compare_results_multiple(origin_dict: dict, path: str)->pd.DataFrame:
    """
    Computes evaluation metrics for multiple reconstructed networks.
    Args:
    origin_dict : dict
        Dictionary mapping BN objects to lists of filenames.
        Example: { bn_object : ["bn_0_m_synch_t_1_s_50_f_3", ...] }
    path : str
        Path to the directory containing .sif files.
    Returns:
    pd.DataFrame
        DataFrame indexed by dataset name with columns:
        ["Jaccard", "Precision", "Recall", "F1"].
    """


    metrics = {"Jaccard": [], "Precision": [], "Recall": [], "F1": []}
    index = []

    for bn in origin_dict:

        for filename in origin_dict[bn]:
            index.append(filename)
            file_metrics = compare_results_single(bn, path, filename)

            for k, v in file_metrics.items():
                metrics[k].append(v)

    df_metrics = pd.DataFrame(metrics, index=index)
    df_metrics.index.name = 'Dataset'
    return df_metrics


def compare_results_single(bn, path: str, filename: str, show=False)->dict:
    """
    Computes similarity metrics between a ground‑truth BN and a reconstructed network.
    Args:
    bn : BN
        Ground‑truth Boolean Network object
    path : str
        Path to the directory containing the .sif file.
    filename : str
        Base name of the dataset (without extension).
    show : bool, optional
        Whether to display the comparison plot. Default is False.
        
    Returns:
    dict
        Dictionary with keys:
        - "Jaccard"
        - "Precision"
        - "Recall"
        - "F1"
    """


    metrics = {}
    
    bn_nx = bn.to_nx_graph()
    file_nx = sif_to_nx(os.path.join(path, filename + ".sif"))
    
    metrics["Jaccard"] = (jaccard_index(bn_nx, file_nx))

    p, r, f1 = precision_recall_f1(bn_nx, file_nx)
    metrics["Precision"] = (p)
    metrics["Recall"] = (r)
    metrics["F1"] = (f1)

    draw_compare_bn(bn_nx, file_nx, path, filename, show=show)

    return metrics
