from enum import auto
from sklearn.utils import shuffle
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.optim as optim
import argparse
import anndata as ad
import os
import tqdm
import copy
from utils import set_seed
from torch.utils.data import TensorDataset, DataLoader

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


def train_epoch(model, optimizer, loss_fn, train_dataloader, device):
    train_loss = 0.0
    for i, batch in enumerate(train_dataloader):
        x = batch[0]
        x = x.to(device)
        x_prime = model(x)
        loss = loss_fn(x_prime, x)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"train loss = {train_loss}")

    return model, optimizer


def compute_val_loss(model, loss_fn, val_dataloader, device):
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            x = batch[0]
            x = x.to(device)
            x_prime = model(x)
            loss = loss_fn(x_prime, x)
            val_loss += loss.item()

    print(f"val loss = {val_loss}")
    return val_loss


def train_model(model, train_dataset, val_dataset, args, device):

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-05)
    loss_fn = nn.MSELoss()

    # TODO: allow to change the batch size
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    val_losses = []
    best_model = None
    for epoch in tqdm.tqdm(range(args.num_epochs)):
        print(f"epoch {epoch + 1}")
        model, optimizer = train_epoch(
            model, optimizer, loss_fn, train_dataloader, device
        )

        val_loss = compute_val_loss(model, loss_fn, val_dataloader, device)

        # save the best model based on the val loss
        if len(val_losses) == 0:
            best_model = copy.deepcopy(model.state_dict())
        else:
            if val_loss < min(val_losses):
                best_model = copy.deepcopy(model.state_dict())

        val_losses.append(val_loss)

    model.load_state_dict(best_model)

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # read the train data
    input_train = ad.read_h5ad(args.train_file)
    data = input_train.X

    # split the train data into train and validation
    num_train = int(data.shape[0] * 0.9)
    train_data = data[:num_train, :]
    val_data = data[num_train:, :]

    autoencoder = AutoEncoder(
        train_data.shape[1], args.reduced_dim, dropout=args.dropout
    )
    autoencoder.to(device)

    # TODO: train the autoencoder (validation the performance with l2 loss
    # after every epoch)
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)

    autoencoder = train_model(
        autoencoder, train_dataset, val_dataset, args, device
    )

    # save the model
    torch.save(autoencoder.state_dict(), args.save_path)
