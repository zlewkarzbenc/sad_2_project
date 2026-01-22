# comparison_utils.py
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from graph_utils import shd, jaccard_index, precision_recall_f1, multiple_sif_to_nx
from bn import *

def draw_compare_bn(G_true, G_pred, title):
    G_combined = nx.compose(G_pred, G_true)  # includes all nodes from both
    pos = nx.spring_layout(G_combined, seed=42)

    nx.draw(G_pred, pos, with_labels=True, node_color="lightblue", arrowsize=20)
    nx.draw(G_true, pos, with_labels=True, edge_color='green', style='dashed', alpha=0.5)

    plt.title(f"Ground truth edges (green dashed) vs Predicted network (blue) - {title}")
    plt.show()

def compare_results(bns, BDE_path, MDL_path):
    nx_BDE = multiple_sif_to_nx(BDE_path)
    nx_MDL = multiple_sif_to_nx(MDL_path)
    print(nx_BDE)

    metrics = {"SHD_BDE": [], "Jaccard_BDE": [], "Precision_BDE": [], "Recall_BDE": [], "F1_BDE": [],
            "SHD_MDL": [], "Jaccard_MDL": [], "Precision_MDL": [], "Recall_MDL": [], "F1_MDL": []}

    for i, bn in enumerate(bns):
        bn_nx = bn.to_nx_graph()

        key = f'dataset_{i}_BDE'
        metrics["SHD_BDE"].append(shd(bn_nx, nx_BDE[key]))
        metrics["Jaccard_BDE"].append(jaccard_index(bn_nx, nx_BDE[key]))

        p, r, f1 = precision_recall_f1(bn_nx, nx_BDE[key])
        metrics["Precision_BDE"].append(p)
        metrics["Recall_BDE"].append(r)
        metrics["F1_BDE"].append(f1)

        draw_compare_bn(bn_nx, nx_BDE[key], key)

        key = f'dataset_{i}_MDL'
        metrics["SHD_MDL"].append(shd(bn_nx, nx_MDL[key]))
        metrics["Jaccard_MDL"].append(jaccard_index(bn_nx, nx_MDL[key]))
        
        p, r, f1 = precision_recall_f1(bn_nx, nx_MDL[key])
        metrics["Precision_MDL"].append(p)
        metrics["Recall_MDL"].append(r)
        metrics["F1_MDL"].append(f1)

        draw_compare_bn(bn_nx, nx_MDL[key], key)

    df_metrics = pd.DataFrame(metrics, index=[f'BN_{i}' for i in range(len(bns))])
    df_metrics.index.name = 'BN_index'
    df_metrics.to_csv('bnf_metrics.csv')
    return df_metrics



# def compare_results(bns, BDE_path, MDL_path):
#     nx_BDE = multiple_sif_to_nx(BDE_path)
#     nx_MDL = multiple_sif_to_nx(MDL_path)
#     print(nx_BDE)

#     metrics = {"SHD_BDE": [], "Jaccard_BDE": [], "Precision_BDE": [], "Recall_BDE": [], "F1_BDE": [],
#             "SHD_MDL": [], "Jaccard_MDL": [], "Precision_MDL": [], "Recall_MDL": [], "F1_MDL": []}

#     for i in range(len(bns)):
#         bn = bns[i]
#         bn_nx = bn.to_nx_graph()

#         key = f'dataset_{i}_BDE'
#         metrics["SHD_BDE"].append(shd(bn_nx, nx_BDE[key]))
#         metrics["Jaccard_BDE"].append(jaccard_index(bn_nx, nx_BDE[key]))

#         p, r, f1 = precision_recall_f1(bn_nx, nx_BDE[key])
#         metrics["Precision_BDE"].append(p)
#         metrics["Recall_BDE"].append(r)
#         metrics["F1_BDE"].append(f1)

#         draw_compare_bn(bn_nx, nx_BDE[key], key)

#         key = f'dataset_{i}_MDL'
#         metrics["SHD_MDL"].append(shd(bn_nx, nx_MDL[key]))
#         metrics["Jaccard_MDL"].append(jaccard_index(bn_nx, nx_MDL[key]))
        
#         p, r, f1 = precision_recall_f1(bn_nx, nx_MDL[key])
#         metrics["Precision_MDL"].append(p)
#         metrics["Recall_MDL"].append(r)
#         metrics["F1_MDL"].append(f1)

#         draw_compare_bn(bn_nx, nx_MDL[key], key)

#     df_metrics = pd.DataFrame(metrics, index=[f'BN_{i}' for i in range(len(bns))])
#     df_metrics.index.name = 'BN_index'
#     df_metrics.to_csv('bnf_metrics.csv')
#     return df_metrics



def compare_results_single(bns, BNF_path):
    nx_BNF = multiple_sif_to_nx(BNF_path)

    metrics = {"SHD": [], "Jaccard": [], "Precision": [], "Recall": [], "F1": []}

    bn = bns[0]
    bn_nx = bn.to_nx_graph()

    key = f'dataset_0'
    metrics["SHD_BDE"].append(shd(bn_nx, nx_BNF[key]))
    metrics["Jaccard_BDE"].append(jaccard_index(bn_nx, nx_BNF[key]))

    p, r, f1 = precision_recall_f1(bn_nx, nx_BNF[key])
    metrics["Precision_BDE"].append(p)
    metrics["Recall_BDE"].append(r)
    metrics["F1_BDE"].append(f1)

    draw_compare_bn(bn_nx, nx_BNF[key], key)

    df_metrics = pd.DataFrame(metrics, index=[f'BN_{i}' for i in range(len(bns))])
    df_metrics.index.name = 'BN_index'
    df_metrics.to_csv('bnf_metrics.csv')
    return df_metrics
