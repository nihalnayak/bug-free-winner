import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import anndata as ad
import os
import tqdm
from utils import set_seed

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# Reference: https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, reduced_dim, dropout=0.1) -> None:
        super().__init__()

        # adding a dropout makes it look like a denoising autoencoder.
        self.encoder = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, reduced_dim),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(reduced_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, input_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_prime = self.decoder(z)
        return x_prime


def train_epoch(model, optimizer, loss_fn, train_dataloader):
    return model, optimizer


def compute_val_loss(model, loss_fn, val_dataloader):
    return 0.0


def train_data(model, train_data, val_data, args):

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-05)
    loss_fn = nn.MSELoss()

    train_dataloader = None
    val_dataloader = None

    val_losses = []
    for epoch in tqdm.tqdm(range(args.num_epochs)):
        print(f"epoch {epoch + 1}")
        model, optimizer = train_epoch(
            model, optimizer, loss_fn, train_dataloader
        )

        val_loss = compute_val_loss(model, loss_fn, val_dataloader)

        # TODO: save the best model based on the val loss

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reduced_dim", help="reduced dimension", type=int)
    parser.add_argument(
        "--train_file",
        help="training file path",
        default="data/openproblems_bmmc_multiome_phase1_rna/openproblems_bmmc_multiome_phase1_rna.censor_dataset.output_train_mod1.h5ad",
    )
    parser.add_argument("--lr", default=1e-03, help="learning rate")
    parser.add_argument(
        "--dropout", help="dropout in the autoencoder", type=float, default=0.1
    )
    parser.add("--num_epochs", help="number of epochs", type=int, default=10)
    parser.add_argument("--save_path", help="save_path for the model")
    parser.add_argument("--seed", help="seed value", default=0, type=int)

    args = parser.parse_args()

    set_seed(args.seed)

    # read the train data
    input_train = ad.read_h5ad(args.train_file)
    data = input_train.X

    # split the train data into train and validation
    num_train = int(data.shape[0] * 0.9)
    train_data, val_data = data[:num_train, :], data[num_train:, :]

    autoencoder = AutoEncoder(
        train_data.shape[1], args.reduced_dim, dropout=args.dropout
    )

    # TODO: train the autoencoder (validation the performance with l2 loss
    # after every epoch)

    # save the model
    torch.save(autoencoder, args.save_path)
