import sys
import os
import utils as ut
import evals as evals
import scot2 as sc
import numpy as np
# import scanpy
import anndata as ad
import scipy


DIR_PATH = os.path.dirname(os.path.realpath(__file__))


from sklearn.decomposition import TruncatedSVD
#print("trying to read")
#ad.read_h5ad("/home/jovyan/data/explore/multiome/multiome_gex_processed_training.h5ad")
# x=np.load("./data/scatac_feat.npy")
# y=np.load("./data/scrna_feat.npy")
print("done reading")
par = {
    "input_train_mod1": "data/openproblems_bmmc_cite_starter/openproblems_bmmc_cite_starter.train_mod1.h5ad",
    "input_train_mod2": "data/openproblems_bmmc_cite_starter/openproblems_bmmc_cite_starter.train_mod2.h5ad",
    "train_sol": "data/openproblems_bmmc_cite_starter/openproblems_bmmc_cite_starter.train_sol.h5ad"
}

par = {
    "input_train_mod1": "F:\starter_kit-match_modality-python_v1.4.0\output\datasets_phase1v2\match_modality\openproblems_bmmc_cite_phase1v2_mod2\openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_test_mod1.h5ad",
    "input_train_mod2": "F:\starter_kit-match_modality-python_v1.4.0\output\datasets_phase1v2\match_modality\openproblems_bmmc_cite_phase1v2_mod2\openproblems_bmmc_cite_phase1v2_mod2.censor_dataset.output_test_mod2.h5ad"
}
# par = {
#     "input_train_mod1": "data/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod1.h5ad",
#     "input_train_mod2": "data/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod2.h5ad"
# }

#
input_train_mod1 = ad.read_h5ad(par['input_train_mod1'])
input_train_mod2 = ad.read_h5ad(par['input_train_mod2'])

print('reducing dimensionality: mod1')
# Do PCA on the input data
embedder_mod1 = TruncatedSVD(n_components=50)
x = embedder_mod1.fit_transform(input_train_mod1.X)

print('reducing dimensionality: mod2')
embedder_mod2 = TruncatedSVD(n_components=50)
y = embedder_mod2.fit_transform(input_train_mod2.X)

print("Dimensions of input datasets are: ", "x= ", x.shape, " y= ", y.shape)

# initialize SCOT object
scot = sc.SCOT(x, y)
# call the alignment with l2 normalization
x_new, y_new = scot.align(k=50, e=0.0005,  normalize=True)

fracs=evals.calc_domainAveraged_FOSCTTM(x_new, y_new[0])
print("Average FOSCTTM score for this alignment with X onto Y is: ", np.mean(fracs))

np.save(os.path.join(DIR_PATH, 'data/coupling.npy'), scot.coupling[0])
