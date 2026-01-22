# from fileinput import filename
# import os
# import subprocess
# from unittest import result
# import pandas as pd
# import re
# import csv
# import random
# from bn import *
# from sklearn.metrics import precision_score, recall_score, f1_score
# import numpy as np

"Stub module. All functions have been moved to the following files:"

"""
(New module)
Dataset creation, trajectory conversion, and OFAT combinations

data_generation.py:

    trajectories_to_bnfinder_txt(bn, trajectories, output_dir, filename) -> None:
    ofat_combinations(param_lists):
    datasets_for_gridsearch(bns, output_dir, ofat_params) -> None:

    # ofat_combinations(param_lists): # STARE
    # dataset_for_nets(bns, output_dir, num_of_traj=None, # STARE
"""


"""
(New module)
Running BNFinder on files

bnfinder_runner.py:

    run_bnfinder_4_one(python2_path, bnfinder_path, input_txt, output_sif, score):
    run_bnfinder_and_collect(input_folder: str, output_folder: str, 
                    python2_path, bnfinder_path, score: str, csv_filename: str)->None:
"""


"""
(New module)
Graph conversions and metric calculations

graph_utils.py:

    sif_to_nx(sif_file: str) -> nx.DiGraph:
    multiple_sif_to_nx(sif_folder: str) -> dict[str, nx.DiGraph]:
    shd(G_true: nx.DiGraph, G_pred: nx.DiGraph) -> int:
    jaccard_index(G_true: nx.DiGraph, G_pred: nx.DiGraph) -> float:
    precision_recall_f1(G_true: nx.DiGraph, G_pred: nx.DiGraph) -> tuple[float, float, float]:
    draw_compare_bn(G_true, G_pred, title):
    
"""


"""
(New module)
Comparing and visualizing BN results.

comparison_utils.py:
    
    draw_compare_bn(G_true, G_pred, title):
    compare_results(bns, BDE_path, MDL_path):
    compare_results_single(bns, BNF_path):
    

    # compare_results(bns, BDE_path, MDL_path):    #STARE
"""