from typing import Optional, Callable, Tuple, List
from copy import deepcopy
import logging
import os
import numpy as np
from qiskit.quantum_info import Statevector
import csv
from qiskit import IBMQ, Aer
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.components.optimizers import Optimizer, OptimizerSupportLevel
from qiskit.providers.aer.noise.noise_model import NoiseModel
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit.aqua.operators import ListOp, VectorStateFn, StateFn, Gradient

# This code file generates the loss data for the easy quantum model with hardware noise. We train the model 100 times \
# and save the loss after 100 training iterations for each each trial using the ADAM optimiser.
# we use two classes from the Iris dataset


## Please insert your own token here
TOKEN = ''

IBMQ.save_account(TOKEN, overwrite=True)
provider = IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')

backend_name = 'ibmq_montreal'
backend_ibmq = provider.get_backend(backend_name)
properties = backend_ibmq.properties()
coupling_map = backend_ibmq.configuration().coupling_map
noise_model = NoiseModel.from_backend(properties)
layout = [2, 3, 5, 8]

shots = 8000

qi_ibmq_noise_model = QuantumInstance(backend=Aer.get_backend('qasm_simulator'),
                                       noise_model=noise_model, optimization_level=0, shots=shots,
                                       seed_transpiler=2, initial_layout=layout)
qi = qi_ibmq_noise_model

# qi = qi_ibmq_noise_model
logger = logging.getLogger(__name__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class ADAM(Optimizer):
    """Adam and AMSGRAD optimizers.

        Adam [1] is a gradient-based optimization algorithm that is relies on adaptive estimates of
        lower-order moments. The algorithm requires little memory and is invariant to diagonal
        rescaling of the gradients. Furthermore, it is able to cope with non-stationary objective
        functions and noisy and/or sparse gradients.

        AMSGRAD [2] (a variant of Adam) uses a 'long-term memory' of past gradients and, thereby,
        improves convergence properties.

        References:

            [1]: Kingma, Diederik & Ba, Jimmy (2014), Adam: A Method for Stochastic Optimization.
                 `arXiv:1412.6980 <https://arxiv.org/abs/1412.6980>`_

            [2]: Sashank J. Reddi and Satyen Kale and Sanjiv Kumar (2018),
                 On the Convergence of Adam and Beyond.
                 `arXiv:1904.09237 <https://arxiv.org/abs/1904.09237>`_

        """

    _OPTIONS = ['maxiter', 'tol', 'lr', 'beta_1', 'beta_2',
                'noise_factor', 'eps', 'amsgrad', 'snapshot_dir']

    def __init__(self,
                 maxiter: int = 10000,
                 tol: float = 1e-6,
                 lr: float = 1e-3,
                 beta_1: float = 0.9,
                 beta_2: float = 0.99,
                 noise_factor: float = 1e-8,
                 eps: float = 1e-10,
                 amsgrad: bool = False,
                 snapshot_dir: Optional[str] = None) -> None:
        """
        Args:
            maxiter: Maximum number of iterations
            tol: Tolerance for termination
            lr: Value >= 0, Learning rate.
            beta_1: Value in range 0 to 1, Generally close to 1.
            beta_2: Value in range 0 to 1, Generally close to 1.
            noise_factor: Value >= 0, Noise factor
            eps : Value >=0, Epsilon to be used for finite differences if no analytic
                gradient method is given.
            amsgrad: True to use AMSGRAD, False if not
            snapshot_dir: If not None save the optimizer's parameter
                after every step to the given directory
        """
        super().__init__()
        for k, v in locals().items():
            if k in self._OPTIONS:
                self._options[k] = v
        self._maxiter = maxiter
        self._snapshot_dir = snapshot_dir
        self._tol = tol
        self._lr = lr
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._noise_factor = noise_factor
        self._eps = eps
        self._amsgrad = amsgrad
        self.loss_list = []

        # runtime variables
        self._t = 0  # time steps
        self._m = np.zeros(1)
        self._v = np.zeros(1)
        if self._amsgrad:
            self._v_eff = np.zeros(1)

        if self._snapshot_dir:

            with open(os.path.join(self._snapshot_dir, 'adam_params.csv'), mode='w') as csv_file:
                if self._amsgrad:
                    fieldnames = ['v', 'v_eff', 'm', 't']
                else:
                    fieldnames = ['v', 'm', 't']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

    def get_support_level(self):
        """ Return support level dictionary """
        return {
            'gradient': OptimizerSupportLevel.supported,
            'bounds': OptimizerSupportLevel.ignored,
            'initial_point': OptimizerSupportLevel.supported
        }

    def save_params(self, snapshot_dir: str) -> None:
        """Save the current iteration parameters to a file called ``adam_params.csv``.

        Note:

            The current parameters are appended to the file, if it exists already.
            The file is not overwritten.

        Args:
            snapshot_dir: The directory to store the file in.
        """
        if self._amsgrad:
            with open(os.path.join(snapshot_dir, 'adam_params.csv'), mode='a') as csv_file:
                fieldnames = ['v', 'v_eff', 'm', 't']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({'v': self._v, 'v_eff': self._v_eff,
                                 'm': self._m, 't': self._t})
        else:
            with open(os.path.join(snapshot_dir, 'adam_params.csv'), mode='a') as csv_file:
                fieldnames = ['v', 'm', 't']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow({'v': self._v, 'm': self._m, 't': self._t})

    def load_params(self, load_dir: str) -> None:
        """Load iteration parameters for a file called ``adam_params.csv``.

        Args:
            load_dir: The directory containing ``adam_params.csv``.
        """
        with open(os.path.join(load_dir, 'adam_params.csv'), mode='r') as csv_file:
            if self._amsgrad:
                fieldnames = ['v', 'v_eff', 'm', 't']
            else:
                fieldnames = ['v', 'm', 't']
            reader = csv.DictReader(csv_file, fieldnames=fieldnames)
            for line in reader:
                v = line['v']
                if self._amsgrad:
                    v_eff = line['v_eff']
                m = line['m']
                t = line['t']

        v = v[1:-1]
        self._v = np.fromstring(v, dtype=float, sep=' ')
        if self._amsgrad:
            v_eff = v_eff[1:-1]
            self._v_eff = np.fromstring(v_eff, dtype=float, sep=' ')
        m = m[1:-1]
        self._m = np.fromstring(m, dtype=float, sep=' ')
        t = t[1:-1]
        self._t = np.fromstring(t, dtype=int, sep=' ')

    def minimize(self, objective_function: Callable[[np.ndarray], float], initial_point: np.ndarray,
                 gradient_function: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float, int]:
        """Run the minimization.

        Args:
            objective_function: A function handle to the objective function.
            initial_point: The initial iteration point.
            gradient_function: A function handle to the gradient of the objective function.

        Returns:
            A tuple of (optimal parameters, optimal value, number of iterations).
        """
        derivative = gradient_function(initial_point)
        self._t = 0
        self._m = np.zeros(np.shape(derivative))
        self._v = np.zeros(np.shape(derivative))
        if self._amsgrad:
            self._v_eff = np.zeros(np.shape(derivative))
        self.loss_list = []
        params = params_new = initial_point
        while self._t < self._maxiter:
            derivative = gradient_function(params)
            self._t += 1
            self._m = self._beta_1 * self._m + (1 - self._beta_1) * derivative
            self._v = self._beta_2 * self._v + (1 - self._beta_2) * derivative * derivative
            lr_eff = self._lr * np.sqrt(1 - self._beta_2 ** self._t) / (1 - self._beta_1 ** self._t)
            if not self._amsgrad:
                params_new = (params - lr_eff * self._m.flatten()
                              / (np.sqrt(self._v.flatten()) + self._noise_factor))
                self.loss_list.append(objective_function(params_new))
            else:
                self._v_eff = np.maximum(self._v_eff, self._v)
                params_new = (params - lr_eff * self._m.flatten()
                              / (np.sqrt(self._v_eff.flatten()) + self._noise_factor))
                self.loss_list.append(objective_function(params_new))
            if self._snapshot_dir:
                self.save_params(self._snapshot_dir)
            if np.linalg.norm(params - params_new) < self._tol:
                return params_new, objective_function(params_new), self._t
            else:
                params = params_new
        return params_new, objective_function(params_new), self._t

    def optimize(self, num_vars: int, objective_function: Callable[[np.ndarray], float],
                 gradient_function: Optional[Callable[[np.ndarray], float]] = None,
                 variable_bounds: Optional[List[Tuple[float, float]]] = None,
                 initial_point: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, float, int]:
        """Perform optimization.

        Args:
            num_vars: Number of parameters to be optimized.
            objective_function: Handle to a function that computes the objective function.
            gradient_function: Handle to a function that computes the gradient of the objective
                function.
            variable_bounds: deprecated
            initial_point: The initial point for the optimization.

        Returns:
            A tuple (point, value, nfev) where\n
                point: is a 1D numpy.ndarray[float] containing the solution\n
                value: is a float with the objective function value\n
                nfev: is the number of objective function calls
        """
        super().optimize(num_vars, objective_function, gradient_function,
                         variable_bounds, initial_point)
        if initial_point is None:
            initial_point = aqua_globals.random.random(num_vars)
        if gradient_function is None:
            gradient_function = Optimizer.wrap_function(Optimizer.gradient_num_diff,
                                                        (objective_function, self._eps))

        point, value, nfev = self.minimize(objective_function, initial_point, gradient_function)
        return point, value, nfev, self.loss_list


sv_sim = False

# size of training data set
training_size = 100

# dimension of data sets (not to be confused with number of data)
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
feature_map = ZFeatureMap(n, reps=1)
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


from math import log


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
for i in range(100):
    print('index: %s' % i)
    np.random.seed(i)
    d = 8  # num of trainable params

    # train classifier
    init_params = 2 * np.pi * np.random.rand(n * (1) * 2)
    opt_params, value, _, loss = optimizer.optimize(len(init_params), objective_function,
                                                    gradient_function=grad_function,
                                                    initial_point=init_params)

    f1 = 'quantum_loss_easy_noise_%d.npy' %i
    np.save(f1, loss)
    f2 = 'opt_params_easy_noise_%d.npy'%i
    np.save(f2, opt_params)


