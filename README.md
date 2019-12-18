# Variational Autoencoders: Generative Models for Dimension Reduction
This code accompanies [my Stat185 final project](Dimension_Reduction_Variational_Autoencoders.pdf) on variational autoencoders. The project derives and explores variational autoencoder from a dimension reduction perspective closely following the analysis in [Kingma et al. (2013)](https://arxiv.org/abs/1312.6114) and [Dai et al. (2017)](https://arxiv.org/abs/1706.05148).

The VAE implementation is based on the official PyTorch example which can be found [here](https://github.com/pytorch/examples/tree/master/vae).

## Installation
This code only requires a few packages listed in `requirements.txt`. If you have Anaconda and PyTorch, you do not need to install additional packages.

## Usage
The possible options are in `setup_args.py`. You can use this code to train both traditional and variational autoencoders. You can vary the dimension of **z**, adjust the amount of noise, select between different datasets, etc.

Depending on your operating system you might need to manually create save directories such as `models/MNIST/VAE/0.2`

## Example
To run a variational autoencoder for 10 epochs, with a 3 dimension **z** vector, on FashionMNIST, with 0.2 of the entries corrupted with Gaussian noise,

```
python train.py --epochs 10 --dataset FashionMNIST --z-dim 3 --noise 0.2
```

## Analysis
This section briefly outlines some of our analysis and findings.

### Manifold Learning
These figures show the smooth manifolds learned for MNIST and FasionMNIST.  

<p float="left" align="middle">
  <img src="https://github.com/AbdulSaleh/dimension-reduction-vae/blob/master/images/manifold_30.png" width="35%" height="35%" />
  <img src="https://github.com/AbdulSaleh/dimension-reduction-vae/blob/master/images/fashion_manifold_30.png" width="35%" height="35%" />
</p>
</p>

### Dimension Reduction

VAEs are effective at encoding and reconstructing the data while smoothing over and correcting small details (such as the broken loop in the 6). The top images are the inputs and the bottom images are the reconstructions.

<p float="left" align="middle">
  <img src="https://github.com/AbdulSaleh/dimension-reduction-vae/blob/master/images/reconstruction_30.png" width="35%" height="35%" />
  <img src="https://github.com/AbdulSaleh/dimension-reduction-vae/blob/master/images/fashion_reconstruction_30.png" width="35%" height="35%" />
</p>
</p>

The scatter plot below shows latent vectors **z** map different MNIST digits to different clusters.

<p align="center">
<img src="https://github.com/AbdulSaleh/dimension-reduction-vae/blob/master/images/spacemap.png" width="35%" height="35%" />
</p>

### Disentanglement

VAEs can learn disentangled representations. Here we find that the x-axis corresponds to the height of the heel and the y-axis corresponds to the height of the boot (excluding the heel).

<p align="center">
<img src="https://github.com/AbdulSaleh/dimension-reduction-vae/blob/master/images/boot_manifold_20.png" width="35%" height="35%" />
</p>


### Manifold Dimension Discovery

VAEs can prune unnecessary dimensions of **z** and discover the true dimensionality of the data. Here there are the VAE identifies that it only requires 2 dimensions to encode the data.


<p align="center">
<img src="https://github.com/AbdulSaleh/dimension-reduction-vae/blob/master/images/manifolddim.png" width="35%" height="35%" />
</p>



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
