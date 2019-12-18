import os
from pathlib import Path

import numpy as np
from scipy.stats import norm
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models import Autoencoder, VAE
from setup_args import parse_args


args = parse_args()
torch.manual_seed(args.seed)
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Load data
dataset = getattr(datasets, args.dataset)
if args.specific_class == -1:
    # Train on full data
    train_loader = torch.utils.data.DataLoader(
        dataset('data/', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset('data/', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
else:
    # Train on specific class
    train_loader = torch.utils.data.DataLoader(
        [example for example in dataset('data/', train=True, download=True, transform=transforms.ToTensor())
            if example[1] == args.specific_class],
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        [example for example in dataset('data/', train=False, transform=transforms.ToTensor())
            if example[1] == args.specific_class],
        batch_size=args.batch_size, shuffle=True, **kwargs)

# Select model
Model = Autoencoder if args.no_vae else VAE
model = Model(z_dim=args.z_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Create save directory
model_file = args.model_file
save_dir = Path(os.getcwd()).joinpath(
    'models', args.dataset, Model.__name__, str(args.noise))
# os.makedirs(save_dir)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    if args.no_vae:
        KLD = -0.5 * torch.sum(1 - mu.pow(2))
    else:
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):

        if args.noise != 0:
            # Corrupt data
            bs = data.shape[0]
            mask = np.random.binomial(1, args.noise, size=bs*28*28).reshape(data.shape)
            noise = np.abs(np.random.normal(size=bs*28*28).reshape(data.shape))
            data = torch.Tensor(data.numpy() + (mask*noise))

        data = data.to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                       recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           save_dir.joinpath('reconstruction_' + str(epoch) + '.png'), nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, args.z_dim).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       save_dir.joinpath('sample_' + str(epoch) + '.png'))

            # code for manifold construction
            if args.z_dim == 2:
                n = 20
                digit_size = 28
                figure = np.zeros((digit_size * n, digit_size * n))
                grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
                grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
                for i, yi in enumerate(grid_x):
                    for j, xi in enumerate(grid_y):
                        z = torch.Tensor([[xi, yi]]).to(device)
                        img = model.decode(z).view(1, 1, 28, 28).cpu()
                        figure[i * digit_size: (i+1) * digit_size,
                               j * digit_size: (j+1) * digit_size] = img[0, 0, :, :]
                save_image(torch.Tensor(figure), save_dir.joinpath('manifold_' + str(epoch) + '.png'))

    torch.save(model.state_dict(), save_dir.joinpath(args.model_file))
