# Code to reproduce the noise plots in the supplementary information


There are several plots in the supplementary information that study the effects of hardware noise.
In particular, two figures capture the effects on the eigenvalue distribution of the Fisher Information, and a third looks at the effects on training in the presence of hardware noise. In this folder, there are 
2 subfolders pertaining to each of these analyses respectively. 

### Subfolder `eigenvalue_distribution`:

- `data`: this folder contains the raw data for the normalised Fisher information matrices generated from experiments involving hardware noise. We use a noise
model that simulates the effects of real hardware noise for the ibmq_montreal 27 qubit device. We examine both
  the easy quantum model and the quantum neural network and look at models of the size: 
  - input size = 4, output size = 2, d = 40
  - input size = 6, output size = 2, d = 60
  - input size = 8, output size = 2, d = 80 
  - input size = 10, output size = 2, d = 100
    


- `generate data`: in this folder, there are individual code files that generate the normalised
Fisher matrices for the models mentioned above. The index `Znoise` files pertain to the easy quantum model
  and `ZZnoise` pertain to the quantum neural network. There are also helper functions stored in the `functions`
  package which help compute the Fisher information under noisy hardware conditions.


- `plot_fig`: this folder contains a code file to load the raw data and generate the 
average eigenvalue distribution plot under noisy conditions, for all models. There is also an additional code file to produce a cumulative plot of the eigenvalues to demonstrate the degeneracy in the classical and easy quantum models.
  
  
### Subfolder `training`:

Since we are limited in resources, the training under hardware noise experiment is conducted for models
with input size = 4, d = 8 and output size = 2. We again use a subset of the Iris dataset and train the 
easy quantum model and quantum neural network 100 times.


- `data`: contains the raw data for the training experiment which is repeated 100 times in order to also 
include standard deviations in the plots.


- `generate data`: in this folder, there are 2 code files which train each quantum model respectively. They
both use the ADAM optimiser contained in the `adam` package. Each model is trained 100 times and the data is 
  stored for every trial.


- `plot_fig`: this folder contains a code file that simply plots the average loss for the easy quantum model and quantum 
neural network with and without noise, including +- 1 standard deviation.
