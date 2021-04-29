# Code to reproduce the effective dimension plot

In this folder, there are 3 subfolders each containing the following:

- `data`: raw data of the normalised Fisher information matrix for the classical neural network,
  the easy quantum model and the quantum neural network. All models have an input size of 
  4, output size of 2 and number of parameters d = 40.
  

- `generate data`: in this folder, there are 3 separate code files to reproduce the Fisher information
matrices for all 3 models discussed above. This is required for the calculation of 
  the effective dimension. These code files require functions from the `effective_dimension`
  package.


- `plot_fig`: this folder contains a code file to convert the Fisher information matrices to
the effective dimension over different values of data (n) and plot these results. There 
  is also a .txt file containing the raw source data of the effective dimension values.