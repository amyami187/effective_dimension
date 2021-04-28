import numpy as np
from effective_dimension import Model, EffectiveDimension, ClassicalNeuralNetwork

# This code file generates data to test the sensitivity of the effective dimension to different number of samples used \
# for the Monte Carlo estimates for the integrals in the effective dimension formula. m = number of theta samples  \
# and we fix n here to a sufficiently large value. In particular, this file looks at models with higher d (higher \
#  number of parameters).
# We fix the model architecture for different input sizes to the classical models that produce the highest effective \
# dimension on average with 100 theta samples and 100 data samples.


n = [1000, 2000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000, 10000000, 10000000000, 10000000000000]

size4 = [4, 4, 4, 2]

size6 = [6, 7, 2, 2]

size8 = [8, 8, 2]

size10 = [10, 8, 1, 4, 2]

seeds = [1, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

for j, k in enumerate(seeds):
    for i in range(10, 110, 10):
        cnet = ClassicalNeuralNetwork(size=size4, samples=i)
        ed = EffectiveDimension(cnet, num_thetas=i, num_inputs=i, seed=k)
        f, trace, fishers = ed.get_fhat()
        effdims = ed.eff_dim(f, n)
        np.save("4in_depth9_effective_dimension_samples_%i_seed_%i.npy" %(i, k), effdims)
        np.save("4in_depth9_f_hats_samples_%i_seed_%i.npy" %(i, k), f)
        np.save("4in_depth9_trace_samples_%i_seed_%i.npy" %(i, k), trace)
    print("status: model 1 of 4, seed ", j+1, " of 10 completed")

for j, k in enumerate(seeds):
    for i in range(10, 110, 10):
        cnet = ClassicalNeuralNetwork(size=size6, samples=i)
        ed = EffectiveDimension(cnet, num_thetas=i, num_inputs=i, seed=k)
        f, trace, fishers = ed.get_fhat()
        effdims = ed.eff_dim(f, n)
        np.save("6in_depth9_effective_dimension_samples_%i_seed_%i.npy" %(i, k), effdims)
        np.save("6in_depth9_f_hats_samples_%i_seed_%i.npy" %(i, k), f)
        np.save("6in_depth9_trace_samples_%i_seed_%i.npy" %(i, k), trace)
    print("status: model 2 of 4, seed ", j + 1, " of 10 completed")

for j, k in enumerate(seeds):
    for i in range(10, 110, 10):
        cnet = ClassicalNeuralNetwork(size=size8, samples=i)
        ed = EffectiveDimension(cnet, num_thetas=i, num_inputs=i, seed=k)
        f, trace, fishers = ed.get_fhat()
        effdims = ed.eff_dim(f, n)
        np.save("8in_depth9_effective_dimension_samples_%i_seed_%i.npy" %(i, k), effdims)
        np.save("8in_depth9_f_hats_samples_%i_seed_%i.npy" %(i, k), f)
        np.save("8in_depth9_trace_samples_%i_seed_%i.npy" %(i, k), trace)
    print("status: model 3 of 4, seed ", j + 1, " of 10 completed")

for j, k in enumerate(seeds):
    for i in range(10, 110, 10):
        cnet = ClassicalNeuralNetwork(size=size10, samples=i)
        ed = EffectiveDimension(cnet, num_thetas=i, num_inputs=i, seed=k)
        f, trace, fishers = ed.get_fhat()
        effdims = ed.eff_dim(f, n)
        np.save("10in_depth9_effective_dimension_samples_%i_seed_%i.npy" %(i, k), effdims)
        np.save("10in_depth9_f_hats_samples_%i_seed_%i.npy" %(i, k), f)
        np.save("10in_depth9_trace_samples_%i_seed_%i.npy" %(i, k), trace)
    print("status: model 4 of 4, seed ", j + 1, " of 10 completed")

