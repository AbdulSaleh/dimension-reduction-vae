# Variational Autoencoders: Generative Models for Dimension Reduction
This code accompanies [my Stat185 final project](www.google.com) on variational autoencoders. The project derives and explores variational autoencoder from a dimension reduction perspective closely following the analysis in [Kingma et al. (2013)](https://arxiv.org/abs/1312.6114) and [Dai et al. (2017)](https://arxiv.org/abs/1706.05148).

The VAE implementation is based on the official PyTorch example which can be found [here](https://github.com/pytorch/examples/tree/master/vae).

## Installation
This code only requires a few packages listed in `requirements.txt`. If you have Anaconda and PyTorch, you do not need to install additional packages.

## Usage
The possible options are in `setup_args.py`. You can use this code to train both traditional and variational autoencoders. You can vary the dimension of *z*, adjust the amount of noise, select between different datasets, etc.

Depending on your operating system you might need to manually create save directories such as `models/MNIST/VAE/0.2`

## Example
To run a variational autoencoder for 10 epochs, with a 3 dimension *z* vector, on FashionMNIST, with 0.2 of the entries corrupted with Gaussian noise,

```
python train.py --epochs 10 --dataset FashionMNIST --z-dim 3 --noise 0.2
```

## Analysis
This section briefly outlines some of our analysis and findings.

### Dimension Reduction
This diagram shows the

<p float="left">
  <img src="https://github.com/AbdulSaleh/dimension-reduction-vae/blob/master/images/manifold_30.png" width="20%" height="20%" />
  <img src="https://github.com/AbdulSaleh/dimension-reduction-vae/blob/master/images/manifold_30.png" width="20%" height="20%" />
</p>


<!--
<div class="row">
  <div class="column">
    <img src="https://github.com/AbdulSaleh/dimension-reduction-vae/blob/master/images/manifold_30.png" width="20%" height="20%">
  </div>
  <div class="column">
    <img src="https://github.com/AbdulSaleh/dimension-reduction-vae/blob/master/images/manifold_30.png" width="20%" height="20%">
  </div>
</div> -->

<!-- https://github.com/AbdulSaleh/dimension-reduction-vae/blob/master/images/manifold_30.png -->
<!--
<p align="center">
<img src="https://github.com/AbdulSaleh/TaoTeChing-NLP/blob/master/plots/dclau_mitchell_freq_comparison.png" width="40%">
</p> -->


### Generative Modelling

### Manifold Learning

### Manifold Dimension

### Disentanglement

<!-- ![Frequency Plot for Mitchell vs DC Lau 1963|20%](https://github.com/AbdulSaleh/TaoTeChing-NLP/blob/master/plots/dclau_mitchell_freq_comparison.png) -->

<!-- <p align="center">
<img src="https://github.com/AbdulSaleh/TaoTeChing-NLP/blob/master/plots/dclau_mitchell_freq_comparison.png" width="40%">
</p> -->

## Reference
If you find this code or analysis useful please cite:
```
@misc{saleh2019vae,
  title={Variational Autoencoders: Generative Models for Dimension reduction},
  author={Saleh, Abdul},
  url={www.google.com},
  year={2019}
}
```
