from qiskit.aqua import QuantumInstance
from qiskit import IBMQ, Aer
from qiskit.providers.aer.noise.noise_model import NoiseModel
import numpy as np
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes
from functions import EffectiveDimension, QuantumNeuralNetwork

TOKEN = 'INSERT YOUR TOKEN HERE'
IBMQ.save_account(TOKEN, overwrite=True)
provider = IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-internal', group='deployed', project='default')
backend_name = 'ibmq_montreal'
backend_ibmq = provider.get_backend(backend_name)
properties = backend_ibmq.properties()
coupling_map = backend_ibmq.configuration().coupling_map
noise_model = NoiseModel.from_backend(properties)
layout = [2, 3, 5, 8]
qi_ibmq_noise_model = QuantumInstance(backend=Aer.get_backend('qasm_simulator'),
                                       noise_model=noise_model, optimization_level=0, shots=8000,
                                       seed_transpiler=2, initial_layout=layout)
qi = qi_ibmq_noise_model
compile_config = {'initial_layout': layout,
                  'seed_transpiler': 2,
                  'optimization_level': 3
                  }

n = [1000, 2000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000, 10000000, 10000000000, 10000000000000]
qubits = 4
fm = ZFeatureMap(qubits, reps=1)
varform = RealAmplitudes(qubits, reps=9, entanglement='full')
qnet = QuantumNeuralNetwork(fm, varform, qi)
ed = EffectiveDimension(qnet, 100, 100)
fhat, _ = ed.get_fhat()
effdim = ed.eff_dim(fhat, n)
np.save('4qubits_fhats_noise_linearZ_full.npy', fhat)
np.save('4qubits_effective_dimension_noise_linearZ_full.npy', effdim)