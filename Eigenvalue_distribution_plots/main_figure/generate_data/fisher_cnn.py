from effective_dimension import Model, EffectiveDimension, ClassicalNeuralNetwork
import numpy as np

# this code generates the data for the classical model's fisher information eigenvalue distribution plot \
# in the main figure

nnsize = [4, 4, 4, 2]
cnet = ClassicalNeuralNetwork(nnsize)
num_inputs = 100
num_thetas = 100
ed = EffectiveDimension(cnet, num_thetas=num_thetas, num_inputs=num_inputs)
f, trace = ed.get_fhat()

np.save("fhat4_[4 4 4 2]_fisher.npy", f)

