# Script conating command line arguments
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument('--no-vae', action='store_true', default=False,
                        help='flag for training traditional instead of variational autoencoder')
    parser.add_argument('--z-dim', '-z', type=int, default=10, metavar='N',
                        help='z hidden dimension size')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # Datatset
    parser.add_argument('--dataset', '-d', type=str, default='MNIST',
                        choices=['MNIST', 'FashionMNIST'],
                        help='Choose dataset')
    parser.add_argument('--specific-class', type=int, default=-1,
                        help='only train on a specified class.\n'
                        'train on entire dataset if -1')
    parser.add_argument('--noise', type=float, default=0,
                        help='ratio of corrupted datapoints using Gaussian noise')
    # Logging
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model-file', '-mf', type=str, default='model.pt', metavar='N',
                        help='model save file name')
    return parser.parse_args()
