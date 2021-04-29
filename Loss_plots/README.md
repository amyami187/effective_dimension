# Code to reproduce the loss plot and Fisher-Rao norm calculations

In this folder, there are 5 subfolders each containing the following:

- `data`: raw data of the loss values over 100 training iterations and 100 training trials for the classical neural network,
  the easy quantum model and the quantum neural network. All models have an input size of 
  4, output size of 2 and number of parameters d = 8. These models are trained on a subset of the
  Iris dataset and serve as toy models to demo the intuition of the effective dimension's relationship
  to training.
  

- `generate data`: in this folder, there are 3 separate code files to train each model on a subset of the 
  Iris dataset. These files require the ADAM optimiser which is contained in `adam` packaged folder. Each
  code file will save the loss values for 100 training trails. 


- `hardware`: this folder contains a jupyter notebook with the relevant code to produce the training loss
results using actual hardware. The details of the hardware experiment can be found in the Methods section
  of the main manuscript.


- `plot_fig`: this folder contains a code file that simply plots the average loss for all models +-1 standard
deviation. The single loss values for the hardware experiment are hard coded and plotted as well. 
  

The `Fisher_rao_norm_calcs` folder has a similar structure and also uses information after training the models. 
In particular, the final parameter values after training each model are saved and used to compute the Fisher-Rao
norm which depends on the trained parameters and the Fisher information. In this folder, you will find:

- `data`: the raw data of each model's optimal parameters required to compute the Fisher-Rao norm.

- `generate_data`: 3 code files that are required to compute the average Fisher-Rao norm for the models
as well as a `functions_fr` package to provide helper functions to achieve this.