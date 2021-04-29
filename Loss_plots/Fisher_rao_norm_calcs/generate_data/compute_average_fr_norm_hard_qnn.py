from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from sklearn import datasets
from sklearn import preprocessing
import numpy as np
from qiskit.quantum_info import Statevector
from functions_fr import *

# This code trains a fixed quantum neural network model 100 times and computes the average Fisher-Rao norm which is \
# equal to W^T*F*W where trained parameters = W and F = Fisher information

iris = datasets.load_iris()
x = preprocessing.normalize(iris.data)
iris_x = x[:100]
n = 4
blocks = 1
sv = Statevector.from_label('0' * n)
feature_map = ZZFeatureMap(n, reps=2)
var_form = RealAmplitudes(n, reps=blocks)
circuit = feature_map.combine(var_form)
d = circuit.num_parameters
effdims = []
frnorms = []
nrange = [100000000]
# randomly initialize the parameters
for i in range(100):
    file = 'opt_params_hard_dep2_%d.npy' %i
    opt_params_one = np.load(file)
    opt_params = np.tile(opt_params_one, (100,1))
    cnet = QuantumNeuralNetwork(feature_map=feature_map, var_form=var_form)
    ed = EffectiveDimension(cnet, iris_x)
    f = ed.get_fhat()
    effdim = ed.eff_dim(f, nrange)[0]
    effdims.append(effdim)
    o1 = np.reshape(opt_params_one, (8,1))
    inner = np.dot(f, o1)  # 8 x 1
    o2 = np.reshape(opt_params_one, (1, 8))
    fisher_rao = np.dot(o2, inner)
    frnorms.append(fisher_rao)
    print('progress:', i/100)
np.save('average_fr_norm.npy', frnorms)