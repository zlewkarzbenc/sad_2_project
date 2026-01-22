# data_generation.py
import os
import random
import pandas as pd
from bn import *

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
# def ofat_combinations(param_lists):
#     """
#     param_lists: lista list parametrów, [m_list, s_list, d_list]
#     zwraca listę kombinacji OFAT
#     """
#     mids = [lst[len(lst)//2] for lst in param_lists]  # środkowe wartości
#     results = set()

#     for idx, lst in enumerate(param_lists):
#         for val in lst:
#             combo = mids.copy()
#             combo[idx] = val
#             results.add(tuple(combo))

#     return list(results)

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


def datasets_for_gridsearch(bns, output_dir, ofat_params) -> None:
    modes = ['sync', 'async']
    metadata = []

    for i, bn in enumerate(bns):

        attractors = bn.get_attractors()
        flat_set = set().union(*attractors)



        for param_set in ofat_params:

            mode, num_steps, num_traj, freq = param_set
            dataset = []
            for j in range(num_traj):

                # losowy punkt startowy
                init = tuple(random.randint(0, 1) for _ in range(bn.num_nodes))

                # pełna trajektoria
                traj = bn.simulate_trajectory(init, steps=num_steps, mode=mode)

                # próbkowanie
                final_traj = BN.sample_trajectory(traj, freq=freq)
                dataset.append(final_traj)

                # liczba stanów atraktorowych
                num_att_states = sum(1 for state in final_traj if state in flat_set)

                metadata.append({
                    'name': f'dataset_{i}_traj_{j}_m_{mode}_t_{num_traj}_s_{num_steps}_f_{freq}',
                    'mode': mode,
                    'length': num_steps,
                    'frequency': freq,
                    'attractors': round(num_att_states / len(final_traj), 2)
                })

            # zapis datasetu dla tej kombinacji
            filename = f'dataset_{i}_m_{mode}_t_{num_traj}_s_{num_steps}_f_{freq}.txt'
            trajectories_to_bnfinder_txt(bn, dataset, output_dir, filename)

    # Write metadata to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv('metadata_nets.csv', index=False)

# def dataset_for_nets(bns, output_dir, num_of_traj=None,
#                     mode=None, num_steps=None, frequency_choice=None) -> None:
#     '''
#     Generate datasets for a list of Boolean networks and save them
#     in BNFinder TXT format.
#     Args:
#         bns: List of BN instances.
#         output_dir: Directory to save the output TXT files.
#     Output:

#     '''

#     metadata = []
#     for i, bn in enumerate(bns):
#         dataset = []

#         attractors = bn.get_attractors()

#         if not num_of_traj: num_of_traj = random.randint(1, 5)

#         for j in range(num_of_traj):

#             init = tuple(random.randint(0,1) for _ in range(bn.num_nodes))

#             # choose sync or async
#             if not mode: 
#                 p_sync = random.random()
#                 if p_sync < 0.5:
#                     mode = 'async'
#                 else:
#                     mode = 'sync'

#             # choose length of trajectory
#             if not num_steps: num_steps = random.randint(10, 100)
#             traj = bn.simulate_trajectory(init, steps=num_steps, mode=mode)

#             # add trajectory (change frequency)
#             if not frequency_choice: frequency_choice = random.randint(1, 5)
#             final_traj = BN.sample_trajectory(traj, freq=frequency_choice)
#             dataset.append(final_traj)

#             # number of attractor states in the trajectory
#             flat_set = set()
#             for attr in attractors:
#                 flat_set.update(attr)

#             num_of_att_states = sum(1 for state in final_traj if state in flat_set)

#             metadata.append({'name': f'dataset_{i}_traj_{j}',
#                         'mode': mode,
#                         'length': num_steps,
#                         'frequency': frequency_choice,
#                         'attractors': round(num_of_att_states/len(final_traj), 2)})
        
#         trajectories_to_bnfinder_txt(bn, dataset, output_dir, f'dataset_{i}.txt')

#     # Write metadata to CSV
#     metadata_df = pd.DataFrame(metadata)
#     metadata_df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
