from typing import Optional, Callable, Tuple, List
import logging
import os
import csv
import numpy as np
from qiskit.aqua import aqua_globals
from qiskit.aqua.components.optimizers import Optimizer
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
import numpy as np
from qiskit.quantum_info import Statevector


# Train the easy quantum model with feature map = ZFeatureMap, variational form = RealAmplitudes

logger = logging.getLogger(__name__)


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
            'gradient': Optimizer.SupportLevel.supported,
            'bounds': Optimizer.SupportLevel.ignored,
            'initial_point': Optimizer.SupportLevel.supported
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
circuit = QuantumCircuit(n)
feature_map = ZFeatureMap(n, reps=1)
var_form = RealAmplitudes(n, reps=blocks)
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
    shots = sum(counts.values())
    result = {class_labels[0]: 0,
              class_labels[1]: 0}
    for key, item in counts.items():
        label = assign_label(key, class_labels)
        result[label] += counts[key]/shots
    return result


def classify(x_list, params, class_labels):
    qc_list = []
    for x in x_list:
        circ_ = circuit.assign_parameters(get_data_dict(params, x))
        qc = sv.evolve(circ_)
        qc_list += [qc]
    probs = []
    for qc in qc_list:
        counts = qc.to_counts()
        prob = return_probabilities(counts, class_labels)
        probs += [prob]
    return probs


from math import log


def CrossEntropy(yHat, y):
    if y == 'A':
      return -log(yHat['A'])
    else:
      return -log(1-yHat['A'])


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
    return cost


# setup the optimizer
optimizer = ADAM(maxiter=100, lr=0.1)

# define objective function for training
objective_function = lambda params: cost_function(training_input, class_labels, params, print_value=True)


d = 8  # num of trainable params

for i in range(100):
    np.random.seed(i)
    # randomly initialize the parameters
    init_params = 2 * np.pi * np.random.rand(n * (1) * 2)

    # train classifier
    init_params = 2 * np.pi * np.random.rand(n * (1) * 2)
    opt_params, value, _, loss = optimizer.optimize(len(init_params), objective_function, initial_point=init_params)

    # print results

    f1 = 'quantum_loss_easy_%d.npy' %i
    np.save(f1, loss)
