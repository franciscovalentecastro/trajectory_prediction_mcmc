# Applications of distribution modeling and MCMC methods to intention forecasting
## Francisco Valente Castro
## Dr. Jean-Bernard Hayet

An overview of the work done in this repository can be seen in the following presentation :
[presentation](doc/Applications_of_distribution_modeling_and_MCMC_methods_to_intention_forecasting.pdf)

The most important parts to understand this work and code are :

* Metropolis-Hastings algorithm - The metropolis-hastings implementation is in [metropolis_hastings_hybrid_kernels.py](mcmc/metropolis_hastings_hybrid_kernels.py)
* Inference on VAE - The inference done using an already trained VAE model is in [mcmc_on_vae.py](mcmc/mcmc_on_vae.py)
* VAE - The VAE model is defined in [network.py](vae_trajectories/network.py) using the PyTorch library.
* Trainning - The trainning was done using the script [train.py](vae_trajectories/train.py)