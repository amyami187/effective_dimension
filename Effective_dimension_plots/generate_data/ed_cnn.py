from effective_dimension import Model, EffectiveDimension, ClassicalNeuralNetwork
import numpy as np

# this code generates the data for the classical model's effective dimension in the main text figure

n = [1000, 2000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000]
nnsize = [4, 4, 4, 2]
cnet = ClassicalNeuralNetwork(nnsize)
num_inputs = 100
num_thetas = 100
ed = EffectiveDimension(cnet, num_thetas=num_thetas, num_inputs=num_inputs)
f, trace = ed.get_fhat()
effdim = ed.eff_dim(f, n)
np.save("fhat4_[4 4 4 2]_fisher.npy", f)
np.save("fhat4_[4 4 4 2]_ed.npy", effdim)