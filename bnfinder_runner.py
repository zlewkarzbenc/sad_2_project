# bnfinder_runner.py
import os
import subprocess
import csv
import re


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
        if not name.endswith(".txt"):
            continue

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
