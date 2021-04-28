from effective_dimension import Model, EffectiveDimension, QuantumNeuralNetwork
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
import numpy as np

# this code generates the data for the quantum neural network models' fisher information eigenvalue distribution plot \
# in the Supplementary Information figure

# Global variables
n = [1000, 2000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000, 10000000, 10000000000, 10000000000000]
blocks = 9
num_inputs = 100
num_thetas = 100

###################################################################################
num_qubits = 6
fm = ZZFeatureMap(feature_dimension=num_qubits, reps=2)
circ = RealAmplitudes(num_qubits, reps=blocks)
qnet = QuantumNeuralNetwork(var_form=circ, feature_map=fm)
ed = EffectiveDimension(qnet, num_thetas=num_thetas, num_inputs=num_inputs)
f, trace = ed.get_fhat()
effdim = ed.eff_dim(f, n)
np.save("6qubits_9layer_f_hats_dep2.npy", f)
np.save("6qubits_9layer_effective_dimension_dep2.npy", effdim)
####################################################################################

num_qubits = 8
fm = ZZFeatureMap(feature_dimension=num_qubits, reps=2)
circ = RealAmplitudes(num_qubits, reps=blocks)
qnet = QuantumNeuralNetwork(var_form=circ, feature_map=fm)
ed = EffectiveDimension(qnet, num_thetas=num_thetas, num_inputs=num_inputs)
f, trace = ed.get_fhat()
effdim = ed.eff_dim(f, n)
np.save("8qubits_9layer_f_hats_dep2.npy", f)
np.save("8qubits_9layer_effective_dimension_dep2.npy", effdim)
####################################################################################

num_qubits = 10
fm = ZZFeatureMap(feature_dimension=num_qubits, reps=2)
circ = RealAmplitudes(num_qubits, reps=blocks)
qnet = QuantumNeuralNetwork(var_form=circ, feature_map=fm)
ed = EffectiveDimension(qnet, num_thetas=num_thetas, num_inputs=num_inputs)
f, trace = ed.get_fhat()
effdim = ed.eff_dim(f, n)
np.save("10qubits_9layer_f_hats_dep2.npy", f)
np.save("10qubits_9layer_effective_dimension_dep2.npy", effdim)
####################################################################################