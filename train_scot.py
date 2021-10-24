import sys
import utils as ut
import evals as evals
import scot2 as sc
import numpy as np
import scanpy

X=np.load("./data/scatac_feat.npy")
y=np.load("./data/scrna_feat.npy")
print("Dimensions of input datasets are: ", "X= ", X.shape, " y= ", y.shape)

# initialize SCOT object
scot=sc.SCOT(X, y)
# call the alignment with l2 normalization
# X_new, y_new = scot.align(k=50, e=0.0005,  normalize=True)
x_new, y_new = scot.align(k=5, e=0.05,  normalize=True)

fracs=evals.calc_domainAveraged_FOSCTTM(x_new, y_new[0])
print("Average FOSCTTM score for this alignment with X onto Y is: ", np.mean(fracs))
