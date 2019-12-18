import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc41 = nn.Linear(128, z_dim)
        self.fc42 = nn.Linear(128, z_dim)
        self.fc5 = nn.Linear(z_dim, 128)
        self.fc6 = nn.Linear(128, 256)
        self.fc7 = nn.Linear(256, 400)
        self.fc8 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))
        return self.fc41(h1), self.fc42(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc7(F.relu(self.fc6(F.relu(self.fc5(z))))))
        return torch.sigmoid(self.fc8(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class Autoencoder(VAE):
    def __init__(self, z_dim):
        super().__init__(z_dim)

    # Override sampling step
    def reparametrize(self, mu, logvar):
        return mu
