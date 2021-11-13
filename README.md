# bug-free-winner

## Installation
Package requirements

```
pip install numpy cython POT sklearn scanpy umap-learn
```

## Running the code

```
python run_scot.py
```

## Running the code on CS grid
```bash
#!/bin/bash
export PYTHON=/data/bats/envs/nnayak2/anaconda3/envs/genomics/bin/python
export TRAIN=/data/bats/users/nnayak2/bug-free-winner/run_scot_large.py
export OUTPUT=/data/bats/users/nnayak2/logs/
qsub -cwd -b y -l vf=64G,vlong $PYTHON -u $TRAIN
```

## Reference
https://github.com/rsinghlab/SCOT
