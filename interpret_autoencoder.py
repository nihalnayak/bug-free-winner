
from re import L
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import pickle


from models import AutoEncoder
from pred_autoencoder import autoencoder_reduce_dimension
"""
    take in:
        - the numpy array of gene expression SNAREseq_DNA count (CELL_NUMBER * GENE_EXPRESSION)
        - the file with correspondence info between the cell_number and cell type 
        - the autoencoder
"""

"""
    X is the dataset we want to reduce 2D numpy array (NUM_CELLS * GENE_EXPRESSION)
    labels is 1D numpy array with labels (NUM_CELLS)
"""
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

def sanity_check(X, labels):
    # reduce the components of X using PCA
    PCA_red = PCA(n_components=2)
    X_red = PCA_red.fit_transform(X)
    x = X_red[:, 0]
    y = X_red[:, 1]
    c = [COLORS[i-1] for i in labels]

    # plot
    fig, ax = plt.subplots()

    ax.scatter(x, y, c=c)
    plt.show()


import csv
 
# open .tsv file
with open(r'C:\Users\PC\Desktop\GSE126074_CellLineMixture_SNAREseq_cDNA_counts.tsv') as file:
    print("opened file")
    # Passing the TSV file to 
    # reader() function
    # with tab delimiter
    # This function will
    # read data from file
    tsv_file = csv.reader(file, delimiter="\t")
     
    # printing data line by line
    for line in tsv_file:
        if 'NBN' in line[0] or "MFN1" in line[0]:
            print(line[0:10])
# # Simple Way to Read TSV Files in Python using pandas
# # importing pandas library
# import pandas as pd
 
# # Passing the TSV file to
# # read_csv() function
# # with tab separator
# # This function will
# # read data from file
# interviews_df = pd.read_csv(r'C:\Users\PC\Desktop\GSE126074_CellLineMixture_SNAREseq_cDNA_counts.tsv', sep='\t')
 
# # printing data
# print(interviews_df)       

X = np.load(r"C:\Users\PC\Desktop\rna_origin.npy")
print("X is loaded", X.shape)
labels = []
with open(r'C:\Users\PC\Desktop\SNAREseq_types.txt', 'r') as fh:
    fh.readline()
    for line in fh:
        labels.append(int(line.split("\t")[2][0]))

fh.close()
print("got all from file", len(labels))
sanity_check(X, labels)




def run_autoencoder_predictions(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = os.path.join(args.save_path, "model.pt")
    state_dict = torch.load(model_path, map_location="cpu")

    # load the params file
    with open(os.path.join(args.save_path, "params.pkl"), "rb") as fp:
        params = pickle.load(fp)

    input_dim = state_dict["encoder.1.weight"].size(1)

    model = AutoEncoder(
        input_dim, reduced_dim=params.reduced_dim, dropout=params.dropout
    )
    model.load_state_dict(state_dict)

    reduced_embs, reconstruct_embs = autoencoder_reduce_dimension(
        model, params.train_file, device, args.reconstruct
    )

    pred_path = os.path.join(args.save_path, "pred.pt")
    torch.save(reduced_embs, pred_path)