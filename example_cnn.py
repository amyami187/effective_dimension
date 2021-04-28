from effective_dimension import Model, EffectiveDimension, ClassicalNeuralNetwork
import numpy as np
import matplotlib.pyplot as plt


# This is an example file to create a classical model and compute its effective dimension

# create ranges for the number of data, n
n = [5000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000]

# specify the size of your neural network
nnsize = [4, 4, 4, 2]

# specify number of data samples and parameter sets to estimate the effective dimension
num_inputs = 100
num_thetas = 100

# create the model
cnet = ClassicalNeuralNetwork(nnsize)

# compute the effective dimension
ed = EffectiveDimension(cnet, num_thetas=num_thetas, num_inputs=num_inputs)
f, trace = ed.get_fhat()
effdim = ed.eff_dim(f, n)

# true dimension of the model
d = cnet.d

# plot the normalised effective dimension
plt.plot(n, np.array(effdim)/d)
plt.xlabel('number of data')
plt.ylabel('normalised effective dimension')
plt.show()
