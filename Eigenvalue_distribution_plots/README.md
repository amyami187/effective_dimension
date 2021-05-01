# Code to reproduce the eigenvalue distribution plots

In this folder, there are 2 subfolders, each pertaining to the data and graphs 
generated in the main text (Figure 2) and supplementary information respectively (Figure 5). All code files 
require functions generated in th `effective_dimension` package.

### Main text: `main_figure` folder

- `data`: raw data of the normalised Fisher information matrix for the classical neural network,
  the easy quantum model and the quantum neural network. All models have an input size of 
  4, output size of 2 and number of parameters d = 40.
  

- `generate data`: in this folder, there are 3 separate code files to reproduce the Fisher information
matrices for all 3 models discussed above. This is required for the calculation of 
  the average eigenvalues. 


- `plot_fig`: this folder contains a code file to load the Fisher information matrices for each model,
compute their average eigenvalues and plot the distribution of these average eigenvalues. In 
  addition, the code file also produces the "zoomed in" subplots which are also distribution
  plots of eigenvalues less than 1 to get a sense of how many eigenvalues are very close to zero.
  
### Supplementary information: `appendix_figure` folder

- `data`: raw data of the normalised Fisher information matrix for the classical neural network,
  the easy quantum model and the quantum neural network. In this case, there are 3 different model 
  sizes/architecture shared across all model types, and this subfolder contains the normalised Fisher data for all model 
  classes and all model sizes of: 
  - input size = 6, output size = 2, d = 60
  - input size = 8, output size = 2, d = 80
  - input size = 10, output size = 2, d = 100 
    

- `generate data`: in this folder, there are 3 separate code files to reproduce the Fisher information
matrices for all 3 model classes and their different architectures discussed above. This is required for the plots of their
  average eigenvalue distributions. These code files require functions from the `effective_dimension`
  package.
  

- `plot_fig`: this folder contains a code file to load the raw data and generate the 
average eigenvalue distribution plot for all models in a density and cumulative density plot.
