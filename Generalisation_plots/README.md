# Code to reproduce the generalisation experiment

Details of the generalisation experiment can be found in Section 5 of the supplementary
information. In this analysis, we fix the architecture of a classical feedforward neural network
to an input size = 6, 1 hidden layer and output size = 2, which contains d = 880 parameters overall. We use a dataset generated from scikit-learn's `make_blobs`
dataset and train the neural network to zero loss. We then calculate the effective dimension 
using the single trained parameter set and calculate the test error. We repeat this process 
multiple times training to zero test error, but each time randomising the true labels of the data more
and more.

In the end, we plot the effective dimension as a function of how much we randomise the labels
as well as plot the test error on the same axis.

In this folder, there are 3 subfolders each containing the following:

- `data`: raw data for the experiment which is repeated multiple times in order to also 
include standard deviations in the plots.


- `generate data`: in this folder, there are 5 code files, indexed by the level of randomisation
in the labels for the experiment. For example, `generalisation_random_03.py` conducts the 
  experiment with 30% label randomisation. There is also a package in this subfolder called `helper_functions`
  which contains the modified functions required for the experiment.


- `plot_fig`: this folder contains a code file to plot the average effective dimension after
training, with +- 1 standard deviation, as well as a plot containing the average test loss +-
  1 standard deviation. The deviations are a result of 100 repeated trials.