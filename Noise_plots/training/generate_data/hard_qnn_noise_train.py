from adam import ADAM
from copy import deepcopy
import logging
import os
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit import IBMQ, Aer
from qiskit.aqua import QuantumInstance
from qiskit.providers.aer.noise.noise_model import NoiseModel
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.aqua.operators import ListOp, VectorStateFn, StateFn, Gradient
from math import log

# This code file generates the loss data for the quantum neural network with hardware noise. We train the model
# 100 times and save the loss after 100 training iterations for each each trial using the ADAM optimiser.
# we use two classes from the Iris dataset

TOKEN = 'insert token here'
IBMQ.save_account(TOKEN, overwrite=True)
provider = IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
backend_name = 'ibmq_montreal'
backend_ibmq = provider.get_backend(backend_name)
properties = backend_ibmq.properties()
coupling_map = backend_ibmq.configuration().coupling_map
noise_model = NoiseModel.from_backend(properties)
layout = [2, 3, 5, 8] # qubit layout for coupling map
shots = 8000
qi_ibmq_noise_model = QuantumInstance(backend=Aer.get_backend('qasm_simulator'),
                                       noise_model=noise_model, optimization_level=0, shots=shots,
                                       seed_transpiler=2, initial_layout=layout)
qi = qi_ibmq_noise_model
logger = logging.getLogger(__name__)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sv_sim = False

# size of training data set
training_size = 100
# dimension of data sets
n = 4

from sklearn import datasets
from sklearn import preprocessing
iris = datasets.load_iris()

# load iris and normalise
x = preprocessing.normalize(iris.data)
x1_train = x[0:49, :] # class A
x2_train = x[50:99, :] # class B
training_input = {'A':x1_train, 'B':x2_train}
class_labels = ['A', 'B']
blocks = 1
sv = Statevector.from_label('0' * n)
# circuit = QuantumCircuit(n)
feature_map = ZZFeatureMap(n, reps=2, entanglement='full')
var_form = RealAmplitudes(n, reps=blocks, entanglement='full')
circuit = feature_map.combine(var_form)


def get_data_dict(params, x):
    """Get the parameters dict for the circuit"""
    parameters = {}
    for i, p in enumerate(feature_map.ordered_parameters):
        parameters[p] = x[i]
    for i, p in enumerate(var_form.ordered_parameters):
        parameters[p] = params[i]
    return parameters


def assign_label(bit_string, class_labels):
    hamming_weight = sum([int(k) for k in list(bit_string)])
    is_odd_parity = hamming_weight & 1
    if is_odd_parity:
        return class_labels[1]
    else:
        return class_labels[0]


def return_probabilities(counts, class_labels):
    result = {class_labels[0]: 0,
              class_labels[1]: 0}
    for key, item in counts.items():
        label = assign_label(key, class_labels)
        result[label] += counts[key]
    return result


def classify(x_list, params, class_labels):
    qc = deepcopy(circuit)
    if not sv_sim:
        qc.measure_all()
    qc_list = []
    for i, x in enumerate(x_list):
        circ_ = qc.assign_parameters(get_data_dict(params, x))
        circ_.name = 'circ' + str(i)
        if sv_sim:
            circ_ = sv.evolve(circ_)
        qc_list += [circ_]
    if not sv_sim:
        results = qi.execute(qc_list)
    probs = []
    # TODO execute qc_list
    for i in range(len(qc_list)):
        if sv_sim:
            counts = qc.to_counts()
        else:
            counts = results.get_counts(qc_list[i])
            # print('counts ', counts)
        counts = {k: v / sum(counts.values()) for k, v in counts.items()}
        prob = return_probabilities(counts, class_labels)
        probs += [prob]
    return probs


def grad_classify(x_list, params, class_labels):
    qc = deepcopy(circuit)
    qc_list = []
    for x in x_list:
        parameters = {}
        for i, p in enumerate(feature_map.ordered_parameters):
            parameters[p] = x[i]
        circ_ = qc.assign_parameters(parameters)
        if sv_sim:
            raise TypeError('For now the gradient implementation only allows for Aer backends.')
        qc_list += [StateFn(circ_)]
    if not sv_sim:
        qc_list = ListOp(qc_list)
        grad_fn = Gradient(method='lin_comb').gradient_wrapper(qc_list, var_form.ordered_parameters,
                                                               backend=qi)
        grad = grad_fn(params)
    probs = []
    for grad_vec in grad:
        prob = []
        for i, qc in enumerate(qc_list):
            counts = VectorStateFn(grad_vec[i]).to_dict_fn().primitive
            prob += [return_probabilities(counts, class_labels)]
        probs += [prob]
    return probs


def CrossEntropy(yHat, y):
    if y == 'A':
        return -log(yHat['A'])
    else:
        return -log(1-yHat['A'])


def grad_CrossEntropy(yHat, y, yHat_grad):
    if y == 'A':
        return -yHat_grad['A']/(yHat['A'])
    else:
        return yHat_grad['A']/((1-yHat['A']))


def cost_function(training_input, class_labels, params, shots=100, print_value=False):
    # map training input to list of labels and list of samples
    cost = 0
    training_labels = []
    training_samples = []
    for label, samples in training_input.items():
        for sample in samples:
            training_labels += [label]
            training_samples += [sample]

    # classify all samples
    probs = classify(training_samples, params, class_labels)

    # evaluate costs for all classified samples
    for i, prob in enumerate(probs):
        # cost += cost_estimate_sigmoid(prob, training_labels[i])
        cost += CrossEntropy(yHat=prob, y=training_labels[i])
    cost /= len(training_samples)
    # return objective value
    print('Cost %.4f' % cost)
    return cost


def grad_cost_function(training_input, class_labels, params, shots=100, print_value=False):
    # map training input to list of labels and list of samples
    grad_cost = np.zeros(len(params))
    training_labels = []
    training_samples = []
    for label, samples in training_input.items():
        for sample in samples:
            training_labels += [label]
            training_samples += [sample]

    # classify all samples
    probs = classify(training_samples, params, class_labels)
    grad_probs = grad_classify(training_samples, params, class_labels)
    # grad_probs = list(map(list, zip(*grad_probs))) #transpose
    # evaluate costs for all classified samples
    for j, grad_prob in enumerate(grad_probs):
        for i, prob in enumerate(probs):
            grad_cost[j] += grad_CrossEntropy(yHat=prob, y=training_labels[i],
                                              yHat_grad=grad_prob[i]) / len(training_samples)
    # return objective value
    print('Gradient', grad_cost)
    return grad_cost


# setup the optimizer
optimizer = ADAM(maxiter=100, lr=0.1)


# define objective function for training
objective_function = lambda params: cost_function(training_input, class_labels, params, print_value=True)


# define function for training
grad_function = lambda params: grad_cost_function(training_input, class_labels, params, print_value=True)


# run simulations
for i in range(41,51):
    print('index: %s' % i)
    np.random.seed(i)
    d = 8  # num of trainable params
    # train classifier
    init_params = 2 * np.pi * np.random.rand(n * (1) * 2)
    opt_params, value, _, loss = optimizer.optimize(len(init_params), objective_function,
                                                    gradient_function=grad_function,
                                                    initial_point=init_params)
    f1 = 'quantum_loss_hard_noise_%d.npy' %i
    np.save(f1, loss)
    f2 = 'opt_params_hard_noise_%d.npy'%i
    np.save(f2, opt_params)