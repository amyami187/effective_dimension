from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, TwoLocal
import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector
from sklearn import datasets
from sklearn import preprocessing
from math import pi
from scipy.special import logsumexp
from sklearn.preprocessing import normalize
import numpy as np
from qiskit import QuantumCircuit, transpile
from typing import List, Union
from collections import OrderedDict
import itertools
from qiskit.quantum_info import Statevector
import multiprocessing as mp
from abc import ABC, abstractmethod


# This code trains a fixed quantum neural network model 100 times and computes the average Fisher-Rao norm which is equal to \
# W^T*F*W where
# trained parameters = W and F = Fisher information


class Model(ABC):
    """
    Abstract base class for classical/quantum models.
    """
    def __init__(self):
        """
        :param thetamin: int, minimum used in uniform sampling of the parameters
        :param thetamax: int,  minimum used in uniform sampling of the parameters
        """
        # Stack data together and combine parameter sets to make calcs more efficient
        self.thetamin = 0
        self.thetamax = 1

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_gradient(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_fisher(self, *args, **kwargs):
        raise NotImplementedError()


class QuantumNeuralNetwork(Model):
    """Creates a quantum neural network with a specified feature map and variational form."""
    def __init__(self,  feature_map, var_form, post_processing=None):
        """
        :param feature_map: quantum circuit, feature map for the data
        :param var_form: quantum circuit, parameterised variational circuit
        :param post_processing: function, returns dictionary of lists containing indices which determine post processing
        """
        super(QuantumNeuralNetwork, self).__init__()
        self.d = len(var_form.parameters)
        self.var_form = var_form
        self.feature_map = feature_map
        self.circuit = QuantumCircuit(feature_map.num_qubits)
        self.circuit = self.circuit.combine(feature_map)
        self.circuit = self.circuit.combine(var_form)
        self.circuit = transpile(self.circuit)
        self.inputsize = self.circuit.num_qubits
        self.sv = Statevector.from_label('0' * self.circuit.num_qubits)
        self.post_processing = OrderedDict(self._parity())  # get the dictionary
        self.outputsize = len(self._parity())

    def _get_params_dict(self, params, x):
        """Get the parameters dict for the circuit"""
        parameters = {}
        for i, p in enumerate(self.feature_map.ordered_parameters):
            parameters[p] = x[i]
        for i, p in enumerate(self.var_form.ordered_parameters):
            parameters[p] = params[i]
        return parameters

    def forward(self, params: Union[List, np.ndarray], x: Union[List, np.ndarray]):
        """
        Computes the model output (p_theta) for given data input and parameter set
        :param params: ndarray, parameters for the model (can be one or multiple sets)
        :param x: ndarray, data inputs for the model (can be one or multiple)
        :return: ndarray, p_theta for every possible basis state
        """

        # specify function to be run in parallel by each process
        def get_probs(inds, thetas, datas, circuit, results):
            for i, theta, data in zip(inds, thetas, datas):
                circuit_ = circuit.assign_parameters(self._get_params_dict(theta, data))
                result = self.sv.evolve(circuit_)
                start = i * 2**circuit.num_qubits
                end = (i+1) * 2**circuit.num_qubits
                results[start:end] = result.probabilities()

        # map input to arrays
        params = np.array(params)
        x = np.array(x)
        # specify number of parallel processes
        num_processes = 5
        # construct index set per process
        indices = []
        start = 0
        size = len(x) // num_processes
        for i in range(num_processes-1):
            end = start + size
            indices += [list(range(start, end))]
            start = end
        indices += [list(range(end, len(x)))]

        # initialize shared array to store results (only supports 1D-array, needs reshaping later)
        results = mp.Array('d', (len(x) * 2**self.circuit.num_qubits))

        # construct processes to be run in parallel
        processes = [mp.Process(target=get_probs, args=(inds, params[inds], x[inds], self.circuit, results))
                     for inds in indices]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

        aggregated_results = np.zeros((len(x), self.outputsize))
        num_probs = 2**self.circuit.num_qubits
        for i in range(len(x)):
            start = i * num_probs
            end = (i+1) * num_probs
            probabilities = results[start:end]
            temp_ = []
            for y in self.post_processing.keys():
                index = self.post_processing[y]  # index for each label
                temp_.append([sum(probabilities[u] for u in index)])
            temp_ = np.reshape(temp_, (1, self.outputsize))
            aggregated_results[i] = temp_
        return aggregated_results

    def get_gradient(self, params, x):
        """
        Computes the gradients wrt parameter for every x using the forward passes.
        :param params: ndarray, parameters for the model (can be one or multiple sets)
        :param x: ndarray, data inputs for the model (can be one or multiple)
        :return: numpy array, gradients
        """
        grads = []
        qc_plus = []
        qc_minus = []
        zeros = np.zeros(np.shape(params))
        for i in range(self.d):
            zeros[:, i] = np.pi / 2.
            qc_plus += [self._get_probabilities(params+zeros, x)]
            qc_minus += [self._get_probabilities(params-zeros, x)]
            zeros[:, i] = 0
            grads.append((np.array(qc_plus[i]) - np.array(qc_minus[i])) * 0.5)
        grads = np.array(grads)

        # reshape the dp_thetas
        full = np.zeros((len(x), self.d, 2**self.circuit.num_qubits))
        for j in range(len(x)):
            row = np.zeros((self.d, 2**self.circuit.num_qubits))
            for i in range(self.d):
                tensor = grads[i]
                row[i] += tensor[j]
            full[j] += row
        return full

    def get_fisher(self, gradients, model_output):
        """
        Computes the jacobian as we defined it and then returns the average jacobian:
        1/K(sum_k(sum_i dp_theta_i/sum_i p_theta_i)) for i in index for label k
        :param gradients: ndarray, dp_theta
        :param model_output: ndarray, p_theta
        :return: ndarray, average jacobian for every set of gradients and model output given
        """
        gradvectors = []
        for k in range(len(gradients)):
            jacobian = []
            m_output = model_output[k]  # p_theta size: (1, outputsize)
            jacobians_ = gradients[k, :, :]  # dp_theta size: (d, 2**num_qubits)
            for idx, y in enumerate(self.post_processing.keys()):
                index = self.post_processing[y]  # index for each label
                denominator = m_output[idx]  # get correct model output sum(p_theta) for indices
                for j in range(self.d):
                    row = jacobians_[j, :]
                    # for each row of a particular dp_theta, do sum(dp_theta)/sum(p_theta) for indices
                    # multiply by sqrt(sum(p_theta)) so that the outer product cross term is correct
                    jacobian.append(np.sqrt(denominator)*(sum(row[u] for u in index) / denominator))
            # append gradient vectors for every output for all data points
            gradvectors.append(np.reshape(jacobian, (self.outputsize, self.d)))
        # full gradient vector
        gradients = np.reshape(gradvectors, (len(gradients), self.outputsize, self.d))

        fishers = np.zeros((len(gradients), self.d, self.d))
        for i in range(len(gradients)):
            grads = gradients[i]  # size = (outputsize, d)
            temp_sum = np.zeros((self.outputsize, self.d, self.d))
            for j in range(self.outputsize):
                temp_sum[j] += np.array(np.outer(grads[j], np.transpose(grads[j])))
            fishers[i] += np.sum(temp_sum, axis=0)  # sum the two matrices to get fisher estimate
        return fishers

    def _get_probabilities(self, params: Union[List, np.ndarray], x: Union[List, np.ndarray]):
        """
        Computes the model output (p_theta) for given data input and parameter set
        :param params: ndarray, parameters for the model (can be one or multiple sets)
        :param x: ndarray, data inputs for the model (can be one or multiple)
        :return: ndarray, p_theta for every possible basis state
        """

        # specify function to be run in parallel by each process
        def get_probs(inds, thetas, datas, circuit, results):
            for i, theta, data in zip(inds, thetas, datas):
                circuit_ = circuit.assign_parameters(self._get_params_dict(theta, data))
                result = self.sv.evolve(circuit_)
                start = i * 2**circuit.num_qubits
                end = (i+1) * 2**circuit.num_qubits
                results[start:end] = result.probabilities()

        # map input to arrays
        params = np.array(params)
        x = np.array(x)

        # specify number of parallel processes
        num_processes = 5

        # construct index set per process
        indices = []
        start = 0
        size = len(x) // num_processes
        for i in range(num_processes-1):
            end = start + size
            indices += [list(range(start, end))]
            start = end
        indices += [list(range(end, len(x)))]

        # initialize shared array to store results (only supports 1D-array, needs reshaping later)
        results = mp.Array('d', (len(x) * 2**self.circuit.num_qubits))

        # construct processes to be run in parallel
        processes = [mp.Process(target=get_probs, args=(inds, params[inds], x[inds], self.circuit, results))
                     for inds in indices]

        for p in processes:
            p.start()
        for p in processes:
            p.join()
        probabilities = []
        num_probs = 2**self.circuit.num_qubits
        for i in range(len(x)):
            start = i * num_probs
            end = (i+1) * num_probs
            probabilities += [results[start:end]]

        # return results
        return probabilities

    # an example of post_processing function
    def _parity(self):
        y1 = []
        y2 = []
        basis_states = [list(i) for i in itertools.product([0, 1], repeat=self.circuit.num_qubits)]
        for idx, k in enumerate(basis_states):
            parity = sum(int(k[i]) for i in range(len(k)))
            if parity % 2 == 0:
                y1.append(idx)
            else:
                y2.append(idx)
        return {'y1': y1, 'y2': y2}


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
# randomly initialize the parameters
for i in range(100):
    file = 'opt_params_hard_dep2_%d.npy' %i
    opt_params_one = np.load(file)
    opt_params = np.tile(opt_params_one, (100,1))


    class EffectiveDimension:
        def __init__(self, model):
            np.random.seed(0)
            self.num_thetas = 1
            self.model = model
            self.d = model.d
            # x, _ = make_blobs(1000, n_features=2, centers=2, random_state=137)  # changed xs to 100 from num_data
            self.x = normalize(iris_x)
            self.num_inputs = len(self.x)

        def get_fhat(self):
            grads = self.model.get_gradient(x=self.x, params=opt_params)  # get gradients, dp_theta
            output = self.model.forward(x=self.x, params=opt_params)  # get model output
            fishers = self.model.get_fisher(gradients=grads, model_output=output)
            fisher_trace = np.trace(np.average(fishers, axis=0))  # compute the trace with all fishers
            # average the fishers over the num_inputs to get the empirical fishers
            fisher = np.average(np.reshape(fishers, (self.num_thetas, self.num_inputs, self.d, self.d)), axis=1)
            f_hat = self.d * fisher / fisher_trace  # calculate f_hats for all the empirical fishers
            return f_hat

        def eff_dim(self, f_hat, n):
            effective_dim = []
            for ns in n:
                Fhat = f_hat * ns / (2 * pi * np.log(n))
                one_plus_F = np.eye(self.d) + Fhat
                det = np.linalg.slogdet(one_plus_F)[1]  # log det because of overflow
                r = det / 2  # divide by 2 because of sqrt
                effective_dim.append(2 * (logsumexp(r) - np.log(self.num_thetas)) / np.log(ns / (2 * pi * np.log(n))))
            return effective_dim

    nrange = [100000000]

    cnet = QuantumNeuralNetwork(feature_map=feature_map, var_form=var_form)
    ed = EffectiveDimension(cnet)
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
print(np.average(frnorms))
