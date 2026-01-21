import os
import subprocess
from unittest import result
import pandas as pd
import re
import csv

"Module for processing Boolean network trajectories and interfacing with BNFinder."

def transitions_csv_to_bnf(csv_folder: str, output_folder: str) -> None:
    """
    Convert CSV files with state transitions into BNFinder-compatible TXT files.
    Each CSV file should have rows as time points and columns as variables.
    Args:
        csv_folder: Folder containing input CSV files.
        output_folder: Folder to save output TXT files.
    """

    for name in os.listdir(csv_folder):
        csv_path = os.path.join(csv_folder, name)

        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, name.replace(".csv", ".txt"))

        # Loading CSV file
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        time = df.shape[0]
        num_nodes = df.shape[1]
        nodes = [f"X{i+1}" for i in range(num_nodes)]

        experiment_list = []
        for i in range(1, time+1):
            experiment_list.append(f"s1:t{i}")

        # Writing to BNFinder TXT format
        with open(out_path, "w") as f:
            f.write("#default 0 1\n")
            f.write("conditions " + " ".join(experiment_list) + "\n")

            for i, node in enumerate(nodes):
                values = df.iloc[:, i].tolist() # states of i-th node
                line = node + " " + " ".join(values) + "\n"
                f.write(line)

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
                    python2_path, bnfinder_path, score: str, csv_filename)->None:
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





