# Course project in "Statistical data analysis 2" (2025/2026)
For more information view `assignment.pdf`.

## opisać co w tym repo się znajduje i do czego służy (np. klasa)

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