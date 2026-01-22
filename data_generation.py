# data_generation.py
import os
import random
import pandas as pd
from bn import *

def trajectories_to_bnfinder_txt(bn, trajectories, output_dir, filename) -> None:
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



def ofat_combinations(param_lists):
    mids = [lst[len(lst)//2] for lst in param_lists]
    modes = ["synch", "asynch"]
    results = set()

    for idx, lst in enumerate(param_lists):
        for val in lst:
            combo = mids.copy()
            combo[idx] = val
            for mode in modes:
                results.add((mode, *combo))

    return list(results)



def generate_dataset(bn, index, attractor_set, mode, num_steps, num_traj, freq) -> None:
    
    dataset = []
    metadata = []

    for j in range(num_traj):

        # losowy punkt startowy
        init = tuple(random.randint(0, 1) for _ in range(bn.num_nodes))

        # pełna trajektoria
        traj = bn.simulate_trajectory(init, steps=num_steps, mode=mode)

        # próbkowanie
        final_traj = BN.sample_trajectory(traj, freq=freq)
        dataset.append(final_traj)

        # liczba stanów atraktorowych
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


def datasets_for_gridsearch(bns, output_dir, ofat_params) -> None:
    
    metadata = []
    origin_bn = {} # słownik przechowujący nazwy plików dla każdej sieci

    for i, bn in enumerate(bns):

        origin_bn[bn] = []

        attractors = bn.get_attractors()
        flat_set = set().union(*attractors)

        for param_set in ofat_params:

            mode, num_steps, num_traj, freq = param_set
            dataset, meta = generate_dataset(bn, i, flat_set, mode, num_steps, num_traj, freq)

            # zapis datasetu dla tej kombinacji
            filename = f'bn_{i}_m_{mode}_t_{num_traj}_s_{num_steps}_f_{freq}.txt'
            trajectories_to_bnfinder_txt(bn, dataset, output_dir, filename)

            metadata.extend(meta)
            origin_bn[bn].append(filename.split('.')[0])

    # Write metadata to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv('metadata_nets.csv', index=False)
    
    return origin_bn


def dataset_for_model(bn, mode, num_steps, num_traj, freq, output_dir, filename):
    attractors = bn.get_attractors()
    attractor_set = set().union(*attractors)

    dataset, metadata = generate_dataset(bn, 0, attractor_set, mode, num_steps, num_traj, freq)
    trajectories_to_bnfinder_txt(bn, dataset, output_dir, filename)
    
    pd.DataFrame(metadata).to_csv('metadata_model.csv', index=False)
