import torch
import torch.nn as nn
import tqdm
import argparse
import os
import pickle
import anndata as ad
import re

from torch.utils.data import DataLoader, TensorDataset
from models import AutoEncoder

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def run_reduce_dimension(
    model: nn.Module, dataset: TensorDataset, device: str
):
    """Function uses the autoencoder encoder to
    reduce the dimensions of dataset.

    Args:
        model (nn.Module): autoencoder model.
        dataset (TensorDataset): tensor dataset containing the
            annData matrix.
        device (str): device.

    Returns:
        torch.Tensor: embeddings with reduced dimensions.
    """
    reduced_embs = torch.Tensor()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            x = batch[0]
            x = x.to(device)
            x_reduced = model.encoder(x)
            reduced_embs = torch.cat((reduced_embs, x_reduced), dim=0)
    return reduced_embs


def autoencoder_reduce_dimension(model, train_file, device):

    # read the train data
    input_train = ad.read_h5ad(train_file)
    train_data = input_train.X
    train_data = train_data.todense()

    train_dataset = TensorDataset((torch.Tensor(train_data)))
    reduced_train = run_reduce_dimension(model, train_dataset, device)

    test_file = re.sub("train", "test", train_file)
    input_test = ad.read_h5ad(test_file)
    test_data = input_test.X
    test_data = test_data.todense()

    test_dataset = TensorDataset((torch.Tensor(test_data)))
    reduced_test = run_reduce_dimension(model, test_dataset, device)

    reduced_embs = {"train": reduced_train, "test": reduced_test}

    return reduced_embs


def main(args):

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

    reduced_embs = autoencoder_reduce_dimension(
        model, params.train_file, device
    )

    pred_path = os.path.join(args.save_path, "pred.pt")
    torch.save(reduced_embs, pred_path)

    print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", help="save_path for the model")
    args = parser.parse_args()

    main(args)
