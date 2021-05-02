# The power of quantum neural networks 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4732830.svg)](https://doi.org/10.5281/zenodo.4732830)

In this repository, there are several folders containing data and code to reproduce the results/figures from 
the manuscript titled "The power of quantum neural networks" (arXiv: https://arxiv.org/abs/2011.00027 and new version to be linked soon). All code was generated using Python v3.7, 
PyTorch v1.3.1 and Qiskit v0.23.0 which can be pip installed. The hardware experiment was conducted on IBM's Montreal 27-qubit device. Below is an explanation of each folder's contents and installation.

## Installation 
This project requires Python version 3.7 and above, as well as Qiskit v0.23.0 and PyTorch v1.3.1. Installation of these packages and all dependencies, can be done using pip:

`$ python -m pip install qiskit==0.23.0` \
`$ python -m pip install torch==1.3.1`

##
### Effective dimension code
This folder contains the main package and framework for most calculations used in this research article. In order to calculate the effective dimension,
we create an abstract base class called `Model`, which can be either quantum or classical. We then have two classes for either regime,
the quantum models (`QuantumNeuralNetwork`) and classical models (`ClassicalNeuralNetwork`). And finally, we have an effective dimension class (`EffectiveDimension`)
to compute the effective dimension of any model class. 

- The classical model is assumed to be a fully connected feedforward neural 
network without biases, constructed using PyTorch. One can simply pass a list to this class which specifies the 
architecture of the model. For example, instantiating the class as follows: `ClassicalNeuralNetwork([4, 3, 4])` will create a feedforward 
neural network with input size = 4, 4 neurons in the first hidden layer and finally an output size of 4. The list can be arbitrarily long and thus, the 
model can have more hidden layers, neurons etc. There is an example file illustrating this setup (`example_cnn.py`).
- The quantum model is assumed to consist of a feature map, a variational form and a post processing function to extract the model's labels. 
The feature map and variational form circuits can be passed as arguments, as well as the post processing funciton, however, the default
post processing function has been hardcoded to compute the parity of output bitstrings as explained in more detail in the Methods section of the 
research article. There is an example file illustrating this setup (`example_qnn.py`).
- To instantiate the `QuantumNeuralNetwork` class, one must first create a feature map circuit and a variational form circuit. This can be done by importing circuits from 
Qiskit's circuit library. An example to create this is included in the folder.
- Both `QuantumNeuralNetwork` and `ClassicalNeuralNetwork` classes have a function to compute the probabilities of obtaining a certain label (the 
`forward` pass), as well as computation for the Fisher information required for the effective dimension. An example to create this is included in the folder.
- The `EffectiveDimension` class requires a model (classical or quantum) as well as a specification of the number of parameter
sets to use (`num_thetas` - default set to 100) and number of data to use (`num_data` - default set to 100 and data drawn from a standard normal 
distribution is used). This class calculates the normalised Fisher information (\hat{F}) and computes the effective dimension for a specified number of data, n, or a list of data 
values. 

#### Expected run time
- Depending on the model, number of parameter sets and data samples chosen, the run time of the effective dimension calculation can be time consuming on a regular laptop. In particular, quantum models take longer to run. As a benchmark, a 4-qubit model with 
40 trainable parameters, 100 parameter samples and 100 data samples can take up to 4 hours to compute the effective dimension. We execute circuits larger than 6 qubits on GPU devices. 
- The bottleneck is the computation of the Fisher information which requires Monte Carlo 
estimates. In other words, we first estimate the Fisher information using 100 data samples for a particular parameter set, then repeat this process for 100 parameters sets and 
finally average over the 100 estimated Fisher information matrices in order to compute the effective dimension. 

### Effective_dimension_plots
- The code file `ed_plot.py` generates the effective dimension plot for Figure 3 a). The raw data is produced by using 
the code provided in the Effective_dimension_code folder and specifying models
with input size = 4, output size = 2 and d = 40. 
- The classical network's architecture is [4,4,4,2]. The easy quantum model uses the ZFeatureMap from Qiskit and has feature map depth = 1 and variational form depth 
= 9 (variational form is the RealAmplitudes circuit from Qiskit). The quantum neural network has feature map depth = 2, variational form depth = 9 (ZZFeatureMap and same variational form).

### Eigenvalue_distribution_plots
- Similarly, the raw data for the Fisher information matrices of each model is generated 
from the code in the Effective_dimension_code folder. In particular, the `get_fhat` function in the `EffectiveDimension` class. 
We use 100 parameter sets and 100 data samples
to calculate the Fisher information (100 times for each parameter set) and plot the average 
distribution of eigenvalues for models of increasing input size = [4, 6, 8, 10] and fixed output size = 2.
- The classical architectures are chosen from a brute force search and specified in the script.

### Generalisation_plots
- In this experiment, we train a classical neural network 100 times over an increasing percentage of randomised labels. We then
calculate the effective dimension once the model has trained to zero loss each time. We also calculate the loss on the test data.
- The raw data is stored in the data folder, with code to generate it in the generate_data folder. The average effective dimension with +-1 standard
deviation over increasing randomised labels and the performance on the test data is plotted in the Supplementary Information.

### Loss_plots
- The raw data for the simulations are stored in the data folder, with code to generate it in the generate_data folder. This data reproduces the training loss for each model in Figure
3 b).
- We train each model for 100 training iterations on the Iris dataset. The input size = 4, output size = 2 and d = 8. We then repeat the training 100
times in order to gain a standard deviation over different random parameter initialisations.
- The code for the hardware experiment is contained in a jupyter notebook in the folder hardware. The raw data from this experiment is contained in the `loss_plot.py` file in the main Loss_plots folder. 
We run the hardware experiment for only 33 training iterations after seeing it performs better than the simulated results. 

### Fisher_rao_norm_calcs
- To calculate the Fisher-Rao norm, we train the models and use the parameter set produced after training (saved as `opt_params`). To ensure the results are robust, we train the models multiple times and compute the average Fisher-Rao norm.
- The raw data is stored in the data folder, with code to generate it in the generate_data folder. The average Fisher-Rao norm for each model is 
contained in the Supplementary Information in Table 1.

### Sensitivity_plots
- The raw data is stored in the data folder, with code to generate it in the generate_data folder. This data reproduces the sensitivity analysis in the 
Supplementary Information.
- We conduct a sensitivity analysis of the effective dimension to different numbers of 
parameter and data samples. We use a classical neural network to illustrate this and check the 
sensitivity in lower and higher depths/dimensions. 
  
### Noise_plots
We introduce the effects of hardware noise on both the training of the easy quantum model and the quantum neural network, as well as 
the eigenvalue distributions of each model. In order to generate the training and eigenvalue distribution plots, we have two folders that 
contain the data, code to generate the data and the code to produce the plots in the Supplementary Information.


________________________________________________________________________________________________________________________________________________________________
## License
**Apache 2.0** (see https://github.com/amyami187/effective_dimension/blob/master/LICENSE)
