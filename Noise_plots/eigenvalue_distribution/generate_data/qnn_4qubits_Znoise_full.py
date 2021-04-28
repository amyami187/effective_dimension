from qiskit import QuantumCircuit, transpile
from typing import List, Union
from collections import OrderedDict
import itertools
from qiskit.quantum_info import Statevector
import multiprocessing as mp
from qiskit.aqua import QuantumInstance
from qiskit import IBMQ, Aer
from collections import Counter
from qiskit.providers.aer.noise.noise_model import NoiseModel
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from math import pi
from scipy.special import logsumexp
import numpy as np
from abc import ABC, abstractmethod
from qiskit import compiler


# Here we store the normalised Fisher for a 4 qubit model with hardware noise
# If you want to scale up, change the dict that is hardcoded in line 165 and possibly elsewhere
# Please insert your own token in line 110


def basis_dict(n):
    basis_states = [list(i) for i in itertools.product([0, 1], repeat=n)]
    d = {}
    for state in basis_states:
        st = ''
        for i in range(n):
            st += str(state[i])
        d.update({st:0})
    return d


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


class EffectiveDimension:
    def __init__(self, model, num_thetas, num_inputs):
        """
        Computes the effective dimension for a parameterised model.
        :param model: class instance (i think that's how you call it?)
        :param num_thetas: int, number of parameter sets to include
        :param num_inputs: int, number of input samples to include
        """
        np.random.seed(0)
        self.model = model
        self.d = model.d
        self.num_thetas = num_thetas
        self.num_inputs = num_inputs
        # Stack data together and combine parameter sets to make calcs more efficient
        rep_range = np.tile(np.array([num_inputs]), num_thetas)
        params = np.random.uniform(self.model.thetamin, self.model.thetamax, size=(self.num_thetas, model.d))
        self.params = np.repeat(params, repeats=rep_range, axis=0)
        x = np.random.normal(0, 1, size=(self.num_inputs, self.model.inputsize))
        self.x = np.tile(x, (self.num_thetas, 1))

    def get_fhat(self):
        """
        :return: ndarray, f_hat values of size (num_inputs, d, d)
        """
        grads = self.model.get_gradient(params=self.params, x=self.x)  # get gradients, dp_theta
        output = self.model.forward(params=self.params, x=self.x)  # get model output
        fishers = self.model.get_fisher(gradients=grads, model_output=output)
        fisher_trace = np.trace(np.average(fishers, axis=0))  # compute the trace with all fishers
        # average the fishers over the num_inputs to get the empirical fishers
        fisher = np.average(np.reshape(fishers, (self.num_thetas, self.num_inputs, self.d, self.d)), axis=1)
        f_hat = self.d * fisher / fisher_trace  # calculate f_hats for all the empirical fishers
        return f_hat, fisher_trace

    def eff_dim(self, f_hat, n):
        """
        Compute the effective dimension.
        :param f_hat: ndarray
        :param n: list, used to represent number of data samples available as per the effective dimension calc
        :return: list, effective dimension for each n
        """
        effective_dim = []
        for ns in n:
            Fhat = f_hat * ns / (2 * pi * np.log(ns))
            one_plus_F = np.eye(self.d) + Fhat
            det = np.linalg.slogdet(one_plus_F)[1]  # log det because of overflow
            r = det / 2  # divide by 2 because of sqrt
            effective_dim.append(2 * (logsumexp(r) - np.log(self.num_thetas)) / np.log(ns / (2 * pi * np.log(ns))))
        return effective_dim

TOKEN = 'INSERT YOUR TOKEN HERE'

IBMQ.save_account(TOKEN, overwrite=True)
provider = IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')

backend_name = 'ibmq_montreal'
backend_ibmq = provider.get_backend(backend_name)
properties = backend_ibmq.properties()
coupling_map = backend_ibmq.configuration().coupling_map
noise_model = NoiseModel.from_backend(properties)
# layout = [0, 1, 2, 3]
layout = [2, 3, 5, 8]

shots = 8000
qi_qasm = QuantumInstance(backend=Aer.get_backend('qasm_simulator'), shots=shots, optimization_level=2)
qi_sv = QuantumInstance(backend=Aer.get_backend('statevector_simulator'))
qi_ibmq_noise_model = QuantumInstance(backend=Aer.get_backend('qasm_simulator'),
                                       noise_model=noise_model, optimization_level=0, shots=shots,
                                       seed_transpiler=2, initial_layout=layout)
qi = qi_ibmq_noise_model
compile_config = {'initial_layout': layout,
                  'seed_transpiler': 2,
                  'optimization_level': 3
                  }
# qi_ibmq = QuantumInstance(backend=backend_ibmq, optimization_level=3, shots=shots,
#                           skip_qobj_validation=False,
#                           seed_transpiler=2, measurement_error_mitigation_cls=CompleteMeasFitter,
#                           seed_simulator=2,
#                           measurement_error_mitigation_shots=8000, initial_layout=layout)


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
        self.circuit = compiler.transpile(self.circuit)
        self.inputsize = self.circuit.num_qubits
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
                circuit_.measure_all()
                result = qi.execute(circuit_)
                createdict = basis_dict(self.circuit.num_qubits)
                jj = result.get_counts(circuit_)
                counter_empty = Counter(createdict)
                counter_results = Counter(jj)
                counter_empty.update(counter_results)
                kk = dict(counter_empty)
                start = i * 2**circuit.num_qubits
                end = (i+1) * 2**circuit.num_qubits
                results[start:end] = np.array(list(kk.values()))/shots

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
                print(i)
                circuit_ = circuit.assign_parameters(self._get_params_dict(theta, data))
                circuit_.measure_all()
                result = qi.execute(circuit_)
                createdict = basis_dict(self.circuit.num_qubits)
                jj = result.get_counts(circuit_)
                counter_empty = Counter(createdict)
                counter_results = Counter(jj)
                counter_empty.update(counter_results)
                kk = dict(counter_empty)
                start = i * 2**circuit.num_qubits
                end = (i+1) * 2**circuit.num_qubits
                gg = np.array(list(kk.values()))
                results[start:end] = gg/shots

        # map input to arrays
        params = np.array(params)
        x = np.array(x)

        # specify number of parallel processes
        num_processes = 8

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


from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
n = [1000, 2000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000, 10000000, 10000000000, 10000000000000]
qubits = 4
fm = ZFeatureMap(qubits, reps=1)
varform = RealAmplitudes(qubits, reps=9, entanglement='full')
qnet = QuantumNeuralNetwork(fm, varform)
ed = EffectiveDimension(qnet, 100, 100)
fhat, _ = ed.get_fhat()
effdim = ed.eff_dim(fhat, n)
np.save('4qubits_fhats_noise_linearZ_full.npy', fhat)
np.save('4qubits_effective_dimension_noise_linearZ_full.npy', effdim)
