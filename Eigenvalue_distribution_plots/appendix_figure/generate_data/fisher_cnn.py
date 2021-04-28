from effective_dimension import Model, EffectiveDimension, ClassicalNeuralNetwork
import numpy as np


# this code generates the data for the classical models' fisher information eigenvalue distribution plot \
# in the Supplementary Information figure

num_inputs = 100
num_thetas = 100

nnsize = [6, 7, 2, 2]
cnet = ClassicalNeuralNetwork(nnsize)
ed = EffectiveDimension(cnet, num_thetas=num_thetas, num_inputs=num_inputs)
f, trace = ed.get_fhat()
np.save("fhat6_[6 7 2 2].npy", f)

nnsize = [8, 8, 2]
cnet = ClassicalNeuralNetwork(nnsize)
ed = EffectiveDimension(cnet, num_thetas=num_thetas, num_inputs=num_inputs)
f, trace = ed.get_fhat()
np.save("fhat8_[8 8 2].npy", f)

nnsize = [10, 8, 1, 4, 2]
cnet = ClassicalNeuralNetwork(nnsize)
ed = EffectiveDimension(cnet, num_thetas=num_thetas, num_inputs=num_inputs)
f, trace = ed.get_fhat()
np.save("fhat10_[10  8  1  4  2].npy", f)

