# Course project in "Statistical data analysis 2" (2025/2026)
The purpose of this project is to investigate how the type and amount of data describing
network dynamics influence the accuracy of inferring network structure within the
framework of Bayesian networks.
For more information view `assignment.pdf`.

## Project structure
```
sad_2_project/
├── __pycache__/
├── .gitignore
├── assignment.pdf
├── bn_exercise.py
├── bn.py
├── bnf_metrics_comparison.csv
├── bnfinder_manual.pdf
├── bnfinder_runner.py
├── comparison_utils.py
├── data_generation.py
├── graph_utils.py
├── input_for_bnfinder/
│   ├── bn_0_m_asynch_t_1_s_50_f_3.txt
│   ├── ...
│   └── bn_2_m_synch_t_8_s_50_f_3.txt
├── input_model/
│   └── model_data.txt
├── main.ipynb
├── metadata_model.csv
├── metadata_nets.csv
├── model.bnet
├── output_bnf_BDE/
│   ├── bn_0_m_asynch_t_1_s_50_f_3.sif
│   ├── ...
│   ├── bn_2_m_synch_t_8_s_50_f_3.sif
│   └── total_scores.csv
├── output_bnf_BDE_graphs/
│   ├── bn_0_m_asynch_t_1_s_50_f_3_comparison.png
│   ├── ...
│   └── bn_2_m_synch_t_8_s_50_f_3_comparison.png
├── output_bnf_MDL/
│   ├── bn_0_m_asynch_t_1_s_50_f_3.sif
│   ├── ...
│   └── total_scores.csv
├── output_bnf_MDL_graphs/
│   ├── bn_0_m_asynch_t_1_s_50_f_3_comparison.png
│   ├── ...
│   └── bn_2_m_synch_t_8_s_50_f_3_comparison.png
├── output_model/
│   ├── model_data.sif
│   └── total_scores.csv
├── output_model_graphs/
│   └── model_data_comparison.png
├── README.md
├── requirements_py2.txt
├── requirements_py3.txt
├── visualisations/
│   ├── attractor_histogram.png
│   ├── metric_comparison.png
│   └── parameter_effects.png
└── visualisations.py
```

## Repository overview
This repository contains a set of Python scripts and input/output files used for Boolean network reconstruction and evaluation using BNFinder. The codebase includes modular functions and classes, each documented with clear docstrings.</br>
Key components:
- bn.py – implements a class and a set of methods for Boolean Network representation, manipulation, and simulation.
- bnfinder_runner.py – manages BNFinder execution and handles input/output formatting
- data_generation.py – functions for generating synthetic datasets with configurable parameters
- graph_utils.py – helper functions for graph comparison and structure analysis
- comparison_utils.py – methods for evaluating reconstructed networks using Jaccard, precision, recall, F1
- visualisations.py – plotting functions for metric comparison and attractor analysis
- main.ipynb – example notebook demonstrating the full pipeline



## Setup

1. Download the [BNfinder source code](https://launchpad.net/bnfinder) and unpack it in the current directory.
```bash
tar -xzf BNfinder-2.0.*.tar.gz
rm -f BNfinder-2.0.*.tar.gz
```

2. Create a conda environment with `python2` and install dependencies.

```bash
conda create -n bnfinder python=2.7
conda activate bnfinder

cd BNfinder-2.0.*

conda install numpy=1.16 scipy=1.2
python2 setup.py install

cd ..
```

3. Use the `main.ipynb` jupyter notebook (or your own scripts) to use the provided software

## Input/Output Overview
The directory `input_for_bnfinder/` contains all synthetic time‑series datasets used as input for BNFinder.The folders `output_bnf_BDE/` and `output_bnf_MDL/` store the reconstructed network structures produced by BNFinder under the BDE and MDL scoring schemes, respectively. The corresponding `output_*_graphs/` directories include comparison plots visualizing differences between the true and reconstructed networks. These outputs illustrate reconstruction quality across different parameter settings and update modes.


