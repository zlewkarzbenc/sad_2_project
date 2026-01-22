# data_generation.py
import os
import random
import pandas as pd
from bn import *

"""
Module for generating synthetic datasets for Boolean Network experiments.
"""

def trajectories_to_bnfinder_txt(bn, trajectories, output_dir: str, filename: str) -> None:
    """
    Convert trajectories directly into a BNFinder-compatible TXT file.

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



def ofat_combinations(param_lists: list[list])->list[tuple]]:
    """
    Generates OFAT (one-factor-at-a-time) parameter combinations.

    Args:
        param_lists (list[list]): A list of parameter lists. Each inner list
            contains possible values for one parameter.

    Returns:
        list[tuple]: List of parameter combinations in the form:
            (mode, param1, param2, ...)
    """
    ## Take the middle value of each parameter list (baseline for OFAT)
    mids = [lst[len(lst)//2] for lst in param_lists]
    modes = ["synch", "asynch"]
    results = set()
    
    # vary one parameter at a time while keeping others at their mid values
    for idx, lst in enumerate(param_lists):
        for val in lst:
            combo = mids.copy()
            combo[idx] = val # replace only the parameter being varied
            # # Add both synchronous and asynchronous variants
            for mode in modes:
                results.add((mode, *combo))

    return list(results)



def generate_dataset(bn, index: int, attractor_set: set,
                     mode: str, num_steps: int, num_traj: int, freq: int) -> None:
    """
    Generates a dataset by simulating trajectories from random initial states.

    Args:
        bn (BN): Boolean Network object.
        index (int): Index of the BN (used for naming).
        attractor_set (set): Set of attractor states for the BN.
        mode (str): Update mode ("synch" or "asynch").
        num_steps (int): Number of simulation steps per trajectory.
        num_traj (int): Number of trajectories to generate.
        freq (int): Sampling frequency for downsampling trajectories.

    Returns:
        tuple:
            dataset (list[list[tuple[int]]]): List of sampled trajectories.
            metadata (list[dict]): Metadata entries for each trajectory.
    """

    
    dataset = []
    metadata = []

    for j in range(num_traj):

        # Random initial point
        init = tuple(random.randint(0, 1) for _ in range(bn.num_nodes))

        # Complete trajectory
        traj = bn.simulate_trajectory(init, steps=num_steps, mode=mode)

        # Sampling
        final_traj = BN.sample_trajectory(traj, freq=freq)
        dataset.append(final_traj)

        # Number of attractors
        num_att_states = sum(1 for state in final_traj if state in attractor_set)
        att_ratio = round(num_att_states / len(final_traj), 2) * 100

        metadata.append({
            'name': f'bn_{index}_m_{mode}_t_{num_traj}_s_{num_steps}_f_{freq}_a_{att_ratio}_traj_{j}',
            'mode': mode,
            'length': num_steps,
            'frequency': freq,
            'attractors': att_ratio
        })

    return dataset, metadata


def datasets_for_gridsearch(bns, output_dir: str, ofat_params) -> None:
    """
    Generates datasets for multiple Boolean Networks across OFAT parameter sets.

    Args:
        bns (list[BN]): List of Boolean Network objects.
        output_dir (str): Directory where datasets will be saved.
        ofat_params (list[tuple]): List of parameter combinations produced by
            `ofat_combinations`.

    Returns:
        dict: Mapping BN --> list of dataset base filenames (without extension).
    """

    
    metadata = []
    origin_bn = {} # a dictionary that stores the filenames associated with each network

    for i, bn in enumerate(bns):

        origin_bn[bn] = []

        attractors = bn.get_attractors()
        flat_set = set().union(*attractors)

        for param_set in ofat_params:

            mode, num_steps, num_traj, freq = param_set
            dataset, meta = generate_dataset(bn, i, flat_set, mode, num_steps, num_traj, freq)

            # Saving dataset for the given combination
            filename = f'bn_{i}_m_{mode}_t_{num_traj}_s_{num_steps}_f_{freq}.txt'
            trajectories_to_bnfinder_txt(bn, dataset, output_dir, filename)

            metadata.extend(meta)
            origin_bn[bn].append(filename.split('.')[0])

    # Write metadata to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv('metadata_nets.csv', index=False)
    
    return origin_bn


def dataset_for_model(bn, mode: str, num_steps: int, 
                      num_traj: int, freq: int, output_dir: str, filename: str)->None:
    """
    Generates a dataset for a single Boolean Network model.

    Args:
        bn (BN): Boolean Network object.
        mode (str): Update mode ("synch" or "asynch").
        num_steps (int): Number of simulation steps.
        num_traj (int): Number of trajectories to generate.
        freq (int): Sampling frequency.
        output_dir (str): Directory where the dataset will be saved.
        filename (str): Output filename for the BNFinder TXT file.
    """

    attractors = bn.get_attractors()
    attractor_set = set().union(*attractors)

    dataset, metadata = generate_dataset(bn, 0, attractor_set, mode, num_steps, num_traj, freq)
    trajectories_to_bnfinder_txt(bn, dataset, output_dir, filename)
    
    pd.DataFrame(metadata).to_csv('metadata_model.csv', index=False)
