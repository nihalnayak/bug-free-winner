# bug-free-winner

## Installation
Package requirements

```
pip install numpy cython POT sklearn scanpy umap-learn black torch tqdm
```

## Running the code

```
python run_scot.py
```

## Running grid search on the CS grid

```bash
export PYTHON=/data/bats/envs/nnayak2/anaconda3/envs/genomics/bin/python
export TRAIN=/data/bats/users/nnayak2/bug-free-winner/run_scot_large.py
export OUTPUT=/data/bats/users/nnayak2/logs/
for mod1_dim in 10 20 50
do
    for mod2_dim in 10 20 50
    do
        for k in 10 20 50 100
        do
            qsub -cwd -b y -l vf=64G,vlong $PYTHON -u $TRAIN --type pca --mod1_dim $mod1_dim --mod2_dim $mod2_dim --k $k
        done
    done
done
```

## Reference
https://github.com/rsinghlab/SCOT
