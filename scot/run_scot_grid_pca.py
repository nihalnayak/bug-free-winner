import sys
import os
import utils as ut
import evals as evals
import scot2 as sc
import numpy as np
import scanpy
import anndata as ad
import scipy
#import metric

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

#from down_sample.py import newsample, use to downsample the competition dataset matrix
from sklearn.decomposition import TruncatedSVD

## Load sample data for scot documentation
# x=np.load("./data/scatac_feat.npy")
# y=np.load("./data/scrna_feat.npy")

par = {
        "input_train_mod1": "Circular_Frustum/s3_mapped1.txt",
        "input_train_mod2": "Circular_Frustum/s3_mapped2.txt"
}

## Load cite-seq starter data
'''
par = {
     "input_train_mod1": "data/openproblems_bmmc_cite_starter/openproblems_bmmc_cite_starter.train_mod1.h5ad",
     "input_train_mod2": "data/openproblems_bmmc_cite_starter/openproblems_bmmc_cite_starter.train_mod2.h5ad",
     "train_sol": "data/openproblems_bmmc_cite_starter/openproblems_bmmc_cite_starter.train_sol.h5ad"
 }
'''
## Load cite-seq real data
'''
par = {
    "input_train_mod1": "data/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod1.h5ad",
    "input_train_mod2": "data/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_mod2.h5ad",
    "input_test_sol":   "data/openproblems_bmmc_cite_phase1_rna/openproblems_bmmc_cite_phase1_rna.censor_dataset.output_test_sol.h5ad",
}
'''
## Load multiome real data
'''
par = {
    "input_train_mod1": "data/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod1.h5ad",
    "input_train_mod2": "data/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod2.h5ad"
}
'''

## Used to get the competition data
'''
input_train_mod1 = ad.read_h5ad(os.path.join(DIR_PATH, par['input_train_mod1']))
input_train_mod2 = ad.read_h5ad(os.path.join(DIR_PATH, par['input_train_mod2']))
input_test_sol = ad.read_h5ad(os.path.join(DIR_PATH, par['input_test_sol']))

print("Dimensions of original datasets are: ", "GEX", input_train_mod1.shape, "ADT", input_train_mod2.shape)

new_mod1,new_mod2,new_sol=newsample(input_train_mod1,input_train_mod2,input_test_sol,50)

print("Dimensions of new downsampled datasets are: ", "GEX", new_mod1.shape, "ADT", new_mod2.shape)
'''

with open(par["input_train_mod1"], 'r') as f:
    input_train_mod1 = [[num for num in line.split()] for line in f]
input_train_mod1 = np.array(input_train_mod1).astype(np.float64)
print(input_train_mod1.shape)

with open(par["input_train_mod2"], 'r') as f:
    input_train_mod2 = [[num for num in line.split()] for line in f]
input_train_mod2 = np.array(input_train_mod2).astype(np.float64)
print(input_train_mod1.shape)


k1 = [20]
k2 = [20]
#k2 = [10,20,30,40,50,60,70,80,90,100]
#k2 = [50]
#k2 = [10 20 30 40 50 60 70 80 90 100]
#k3 = [50]
# k3 = [10,20,30,40,50,60,70,80,90,100]
#e1 = [0.005]
k3 = [50]
e1 = [0.0005,0.001,0.005,0.01,0.05,0.1,0.5]
'''
k1 = [10, 20]
k2 = [5,10]
k3 = [50]
e1 = [0.005]
'''


for k_1 in k1:
    for k_2 in k2:
        for k_3 in k3:
            for e_1 in e1:
                embedder_mod1 = TruncatedSVD(n_components=k_1)
                x = embedder_mod1.fit_transform(input_train_mod1)
                embedder_mod2 = TruncatedSVD(n_components=k_2)
                y = embedder_mod2.fit_transform(input_train_mod2)
                scot = sc.SCOT(x, y)
                x_new, y_new = scot.align(k=k_3, e=e_1,  normalize=True)
                fracs=evals.calc_domainAveraged_FOSCTTM(x_new, y_new[0])
                print("Average FOSCTTM score with k_1="+str(k_1)+" k_2="+str(k_2)+" k_3="+str(k_3)+" e="+str(e_1), np.mean(fracs))
                f=open("grid_result.txt", "a+")
                f.write('k_1: %d, k_2: %d, k_3: %d, e: %.5f, score: %.8f \n' % (k_1,k_2,k_3,e_1,np.mean(fracs)))
                f.close()
                #np.save(os.path.join(DIR_PATH, 'data/large_coupling.npy'), scot.coupling[0])
                #print("Matching score k_1="+str(k_1)+" k_2="+str(k_2)+" k_3="+str(k_3), metric.calc_match(scot.coupling[0],new_sol))
