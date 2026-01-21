from fileinput import filename
import os
import subprocess
from unittest import result
import pandas as pd
import re
import csv
import random
from bn import *

"Module for processing Boolean network trajectories and interfacing with BNFinder."

def trajectories_to_bnfinder_txt(trajectories, output_dir, filename) -> None:
    """
    Convert trajectories directly into a BNFinder-compatible TXT file,
    skipping CSV creation.

    Args:
        trajectories: List of trajectories; each trajectory is a list of states.
                      Each state is a tuple (x1, ..., xn).
        output_dir: Directory to save the output TXT file.
        filename: Output filename (should end with .txt).
    """

    num_nodes = len(trajectories[0][0])
    nodes = [f"X{i+1}" for i in range(num_nodes)]

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


def dataset_for_nets(bns, output_dir) -> None:
    '''
    Generate datasets for a list of Boolean networks and save them
    in BNFinder TXT format.
    Args:
        bns: List of BN instances.
        output_dir: Directory to save the output TXT files.
    '''

    for i, bn in enumerate(bns):
        dataset = []
        init = tuple(random.randint(0,1) for _ in range(bn.num_nodes))

        num_of_traj = random.randint(1, 5)

        for _ in range(num_of_traj):

            # choose sync or async
            p_sync = random.random()
            if p_sync < 0.5:
                mode = 'async'
            else:
                mode = 'sync'

            # choose length of trajectory
            num_steps = random.randint(50, 200)
            traj = bn.simulate_trajectory(init, steps=num_steps, mode=mode)

            # add trajectory (change frequency)
            frequency_choice = random.randint(1, 5)
            dataset.append(BN.sample_trajectory(traj, freq=frequency_choice))

        trajectories_to_bnfinder_txt(dataset, output_dir, f'dataset_{i}.txt')



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
