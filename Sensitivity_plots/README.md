# Code to reproduce the sensitivity analysis plot in the supplementary information

Upon estimating the effective dimension, there are two Monte Carlo estimates involved
to estimate the integral over data to estimate the Fisher information, and to estimate the integral 
over the full parameter space for roughly $\sqrt(\det(1 + n*Fisher(theta)))$ as defined in definition
of the effective dimension in the manuscript. Thus, choosing number of samples for each estimate is 
important and a sensitivity analysis to understand the behaviour of the effective dimension to each of
these is important. We thus, vary number of data samples used to estimate the Fisher as well as number
of thetas (parameter sets) used to estimate the integral over the full parameter space.

All these analyses are done using classical feedforward 
models. We found the quantum models far less sensitive to these estimates. In this folder, there are 3 subfolders each containing the following:

- `data`: raw data of the effective dimension for fixed n, but varying d, number of data samples used to 
estimate the Fisher (not be confused with n in the effective dimension) and number of theta
  samples. We separate the data into a higher depth and lower depth category depending on the dimension, d,
  of the model. d < 40 constitutes lower depth, and d >= 40 is considered a higher depth.
  

- `generate data`: in this folder, there are 2 separate code files to generate the sensitivity results
for higher depth and lower depth classical models respectively.


- `plot_fig`: in this folder, there are 2 separate code files to plot the sensitivity results
for higher depth and lower depth classical models respectively.