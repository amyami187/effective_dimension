import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

# This code produces the effective dimension plot for Figure 3 a)

# load data
path = 'insert_path_to_data_here'

fhat_classical = np.load(path + "data/fhat4_[4 4 4 2]_ed.npy")
fhat_easy_qnn = np.load(path + "data/4qubits_9layer_f_hats_pauli.npy")
fhat_qnn = np.load(path + "data/4qubits_9layer_f_hats_dep2.npy")

# colors:
rooi = np.array([255, 29, 0])/255
blou = np.array([0, 150, 236])/255
groen = np.array([0,208,0])/255


# get the effective dimension using the normalised fisher matrices
def eff_dim(f_hat, n):
    d = len(f_hat[0])
    effective_dim = []
    for ns in n:
        Fhat = f_hat * ns / (2 * np.pi * np.log(ns))
        one_plus_F = np.eye(d) + Fhat
        det = np.linalg.slogdet(one_plus_F)[1]  # log det because of overflow
        r = det / 2  # divide by 2 because of sqrt
        effective_dim.append(2 * (logsumexp(r) - np.log(100)) / np.log(ns / (2 * np.pi * np.log(ns))))
    return np.array(effective_dim)/d


# specify a range for number of data samples, n
n = [1000, 2000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000]

plt.figure(figsize=(8,6))

plt.plot(n, eff_dim(fhat_qnn, n), label='quantum NN', color=groen)
plt.plot(n, eff_dim(fhat_classical, n), label='classical NN', color=rooi)
plt.plot(n, eff_dim(fhat_easy_qnn, n), label='easy quantum', color=blou)

plt.ylabel("normalised effective dimension")
plt.ylim(ymax=1)
plt.xlabel("number of data")
plt.legend()
plt.savefig('ed_3comp_in4_dep2.eps', format='eps', dpi=1000)
plt.show()

# save raw data in text file
np.savetxt('effective_dimension.txt',[eff_dim(fhat_qnn, n), eff_dim(fhat_classical, n), eff_dim(fhat_easy_qnn, n)])