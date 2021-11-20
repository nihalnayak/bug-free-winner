import os
import utils as ut
import evals as evals
import scot2 as sc
import numpy as np
import anndata as ad
import argparse


DIR_PATH = os.path.dirname(os.path.realpath(__file__))

from dimensionality_reduction import run_decomposition
from sklearn.decomposition import TruncatedSVD, PCA, SparsePCA
from umap import UMAP

# TODO: autoencoders
METHOD_MAP = {
    "pca": PCA,
    "svd": TruncatedSVD,
    "sparse_pca": SparsePCA,
    "umap": UMAP,
}


def run_scot(x, y, args):

    print(
        "Dimensions of input datasets are: ", "x= ", x.shape, " y= ", y.shape
    )

    # initialize SCOT object
    scot = sc.SCOT(x, y)

    # call the alignment with l2 normalization
    x_new, y_new = scot.align(k=args.k, e=args.e, normalize=True)

    fracs = evals.calc_domainAveraged_FOSCTTM(x_new, y_new[0])
    print(
        "Average FOSCTTM score for this alignment with X onto Y is: ",
        np.mean(fracs),
    )

    np.save(
        os.path.join(
            DIR_PATH,
            f"data/large_coupling_type_{args.type}"
            f"_dim_{args.dim}_k_{args.k}_e_{args.e}_seed_{args.seed}.npy",
        ),
        scot.coupling[0],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type", help="type of the dimensionality reduction technique"
    )
    parser.add_argument(
        "--dim_mod1", help="reduced dimension of the first modality", type=int
    )
    parser.add_argument(
        "--dim_mod2", help="reduced dimension of the second modality", type=int
    )
    parser.add_argument(
        "--k",
        help="number of neighbours in the knn graph",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--e",
        help="epsilon value in the SCOT algorithm",
        type=int,
        default=0.0005,
    )
    parser.add_argument("--seed", help="seed value", default=0, type=int)

    args = parser.parse_args()

    # setting the seed value for the experiments
    ut.set_seed(args.seed)

    # TODO: work on making it easy to change the dataset
    par = {
        "input_train_mod1": "data/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod1.h5ad",
        "input_train_mod2": "data/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_test_mod2.h5ad",
    }

    #
    input_train_mod1 = ad.read_h5ad(
        os.path.join(DIR_PATH, par["input_train_mod1"])
    )
    input_train_mod2 = ad.read_h5ad(
        os.path.join(DIR_PATH, par["input_train_mod2"])
    )

    x, y = run_decomposition(
        METHOD_MAP[args.type],
        input_train_mod1.X,
        input_train_mod2.X,
        args.dim_mod1,
        args.dim_mod2,
    )

    run_scot(x, y, args)

    print("done!")
