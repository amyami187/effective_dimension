from effective_dimension import Model, EffectiveDimension, QuantumNeuralNetwork
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
import numpy as np


# this code generates the data for the quantum neural network model's fisher information eigenvalue distribution plot \
# in the main figure

# Global variables
blocks = 9
###################################################################################
num_qubits = 4
num_inputs = 100
num_thetas = 100
fm = ZZFeatureMap(feature_dimension=num_qubits, reps=2)
circ = RealAmplitudes(num_qubits, reps=blocks)
qnet = QuantumNeuralNetwork(var_form=circ, feature_map=fm)
ed = EffectiveDimension(qnet, num_thetas=num_thetas, num_inputs=num_inputs)
f, trace = ed.get_fhat()
np.save("4qubits_9layer_f_hats_dep2.npy", f)
####################################################################################