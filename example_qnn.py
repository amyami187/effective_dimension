from effective_dimension import QuantumNeuralNetwork, EffectiveDimension
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
import matplotlib.pyplot as plt
import numpy as np

# This is an example file to create a quantum model and compute its effective dimension

# create ranges for the number of data, n
n = [5000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000]

# number of times to repeat the variational circuit
blocks = 1

# number of qubits, data samples and parameter sets to estimate the effective dimension
num_qubits = 3
num_inputs = 10
num_thetas = 10

# create a feature map
fm = ZFeatureMap(feature_dimension=num_qubits, reps=1)

# create a variational circuit
circ = RealAmplitudes(num_qubits, reps=blocks)

# set up the combined quantum model
qnet = QuantumNeuralNetwork(var_form=circ, feature_map=fm)

# number of model parameters is d
d = qnet.d
# set up the effective dimension and compute
ed = EffectiveDimension(qnet, num_thetas=num_thetas, num_inputs=num_inputs)
f, trace = ed.get_fhat()

# compute the effective dimension
effdim = ed.eff_dim(f, n)
###############################

# plot the normalised effective dimension for the model
plt.plot(n, np.array(effdim)/d)
plt.xlabel('number of data')
plt.ylabel('normalised effective dimension')
plt.show()