import torch
import torch.nn as nn

# Reference: https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/simple_autoencoder.py
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, dim, dropout=0.1) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, dim),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(True),
            nn.Linear(256, input_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_prime = self.decoder(z)
        return x_prime
