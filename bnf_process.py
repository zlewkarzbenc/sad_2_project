from fileinput import filename
import os
import subprocess
from unittest import result
import pandas as pd
import re
import csv
import random
from bn import *
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

"Module for processing Boolean network trajectories and interfacing with BNFinder."

def trajectories_to_bnfinder_txt(bn, trajectories, output_dir, filename) -> None:
    """
    Convert trajectories directly into a BNFinder-compatible TXT file,
    skipping CSV creation.

    Args:
        bn: BN class object
        trajectories: List of trajectories; each trajectory is a list of states.
        output_dir: Directory to save the output TXT file.
        filename: Output filename (should end with .txt).
    """

    num_nodes = bn.num_nodes
    nodes = bn.node_names

    # Build condition labels and per-node value streams
    conditions = []
    node_values = [[] for _ in range(num_nodes)]

    # Experiment conditions
    for exp_idx, traj in enumerate(trajectories, start=1):
        for t_idx, state in enumerate(traj, start=1):
            conditions.append(f"s{exp_idx}:t{t_idx}")
            for i, value in enumerate(state):
                node_values[i].append(str(value))

    # Write to BNFinder TXT format
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    with open(out_path, "w") as f:
        f.write("#default 0 1\n")
        f.write("conditions " + " ".join(conditions) + "\n")

        for node, values in zip(nodes, node_values):
            f.write(node + " " + " ".join(values) + "\n")


def dataset_for_nets(bns, output_dir, num_of_traj=None,
                    mode=None, num_steps=None, frequency_choice=None) -> None:
    '''
    Generate datasets for a list of Boolean networks and save them
    in BNFinder TXT format.
    Args:
        bns: List of BN instances.
        output_dir: Directory to save the output TXT files.
    Output:

    '''

    metadata = []
    for i, bn in enumerate(bns):
        dataset = []

        attractors = bn.get_attractors()

        if not num_of_traj: num_of_traj = random.randint(1, 5)

        for j in range(num_of_traj):

            init = tuple(random.randint(0,1) for _ in range(bn.num_nodes))

            # choose sync or async
            if not mode: 
                p_sync = random.random()
                if p_sync < 0.5:
                    mode = 'async'
                else:
                    mode = 'sync'

            # choose length of trajectory
            if not num_steps: num_steps = random.randint(10, 100)
            traj = bn.simulate_trajectory(init, steps=num_steps, mode=mode)

            # add trajectory (change frequency)
            if not frequency_choice: frequency_choice = random.randint(1, 5)
            final_traj = BN.sample_trajectory(traj, freq=frequency_choice)
            dataset.append(final_traj)

            # number of attractor states in the trajectory
            flat_set = set()
            for attr in attractors:
                flat_set.update(attr)

            num_of_att_states = sum(1 for state in final_traj if state in flat_set)

            metadata.append({'name': f'dataset_{i}_traj_{j}',
                        'mode': mode,
                        'length': num_steps,
                        'frequency': frequency_choice,
                        'attractors': round(num_of_att_states/len(final_traj), 2)})
        
        trajectories_to_bnfinder_txt(bn, dataset, output_dir, f'dataset_{i}.txt')

    # Write metadata to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)




def run_bnfinder_4_one(python2_path, bnfinder_path, input_txt, output_sif, score):
    '''
    Run BNFinder on a single input file.
    Args:
        python2_path: Path to Python 2 interpreter.
        bnfinder_path: Path to BNFinder script.
        input_txt: Path to input TXT file.
        output_sif: Path to output SIF file.
        score: Scoring method to use.
    '''
    cmd = [
        python2_path, bnfinder_path,
        "-e", input_txt,
        "-s", score,
        "-n", output_sif, 
        "-v"
    ]
    
    print("Executing:", " ".join(cmd))
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result



def run_bnfinder_and_collect(input_folder: str, output_folder: str, 
                    python2_path, bnfinder_path, score: str, csv_filename: str)->None:
    '''
    Run BNFinder on all files in the input folder and collect total scores into a CSV file.
    Args:
        input_folder: Folder with input TXT files.
        output_folder: Folder to save output SIF files.
        python2_path: Path to Python 2 interpreter.
        bnfinder_path: Path to BNFinder script.
        score: Scoring method to use.
        csv_filename: Filename for the output CSV file (without extension).
    '''
    total_scores = []
    os.makedirs(output_folder, exist_ok=True)
    # Running BNFinder for each file
    for name in os.listdir(input_folder):
        input_txt = os.path.join(input_folder, name)
        output_sif = os.path.join(output_folder, name.replace(".txt", ".sif"))
        
        result = run_bnfinder_4_one(python2_path, bnfinder_path, input_txt, output_sif, score)
        stdout = result.stdout
        
        # Extracting total score from BNFinder output
        total_score = float(
            re.search(
                r'Total score of optimal network:\s*([0-9.]+)',
                stdout
            ).group(1)
        )
        total_scores.append((name, total_score))
    # Writing total scores to CSV
    csv_path = os.path.join(output_folder, f"{csv_filename}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "total_score"])  # Header
        writer.writerows(total_scores)


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

def multiple_sif_to_nx(sif_folder: str) -> dict[str, nx.DiGraph]:
    "Changing multiple SIF files in a folder into NetworkX DiGraphs."

    mode = sif_folder[-3:]

    graphs = {}
    for filename in os.listdir(sif_folder):
        if filename.endswith(".sif"):
            path = os.path.join(sif_folder, filename)
            graph_name = filename[:-4] + '_' + mode  # Remove .sif extension
            graphs[graph_name] = sif_to_nx(path)
    return graphs

def shd(G_true: nx.DiGraph, G_pred: nx.DiGraph) -> int:
    "Calculates the Structural Hamming Distance (SHD) between two graphs."

    all_edges = set(G_true.edges()).union(set(G_pred.edges()))
    diff = sum(1 for e in all_edges if e not in G_true.edges() or e not in G_pred.edges())
    return diff

def jaccard_index(G_true: nx.DiGraph, G_pred: nx.DiGraph) -> float:
    """Calculates the Jaccard Index between the edge sets of two graphs.
    It is a fraction of correctly predicted edges"""

    E_true = set(G_true.edges())
    E_pred = set(G_pred.edges())
    if not E_true and not E_pred: return 1.0
    return len(E_true & E_pred) / len(E_true | E_pred)


def precision_recall_f1(G_true: nx.DiGraph, G_pred: nx.DiGraph) -> tuple[float, float, float]:
    """Calculates precision, recall, and F1-score between the edge sets of two graphs"""
    # Ensure all nodes are included
    nodes = list(G_true.nodes())
    
    # Convert to adjacency matrices
    A_true = nx.to_numpy_array(G_true, nodelist=nodes)
    A_pred = nx.to_numpy_array(G_pred, nodelist=nodes) if len(G_pred.nodes()) > 0 else np.zeros_like(A_true)

    # Flatten for sklearn
    y_true = A_true.flatten()
    y_pred = A_pred.flatten()

    # Handle the case where y_pred has no positives
    if y_pred.sum() == 0 and y_true.sum() == 0:
        return 1.0, 1.0, 1.0  # perfect match: no edges predicted, none in truth
    else:
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return precision, recall, f1
    

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

    metrics = {"SHD_BDE": [], "Jaccard_BDE": [], "Precision_BDE": [], "Recall_BDE": [], "F1_BDE": [],
            "SHD_MDL": [], "Jaccard_MDL": [], "Precision_MDL": [], "Recall_MDL": [], "F1_MDL": []}

    for i in range(len(bns)):
        bn = bns[i]
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
