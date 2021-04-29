from typing import Optional, Callable, Tuple, List
import os
import numpy as np
import csv
from qiskit.aqua import aqua_globals
from qiskit.aqua.components.optimizers import Optimizer, OptimizerSupportLevel


class ADAM(Optimizer):
    """Adam and AMSGRAD optimizers.
        This implementation is taken directly from Qiskit. Please see Qiskit optimiser documentation for details.
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

        super().optimize(num_vars, objective_function, gradient_function,
                         variable_bounds, initial_point)
        if initial_point is None:
            initial_point = aqua_globals.random.random(num_vars)
        if gradient_function is None:
            gradient_function = Optimizer.wrap_function(Optimizer.gradient_num_diff,
                                                        (objective_function, self._eps))

        point, value, nfev = self.minimize(objective_function, initial_point, gradient_function)
        return point, value, nfev, self.loss_list