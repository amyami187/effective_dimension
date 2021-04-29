import numpy as np
import matplotlib.pyplot as plt

# This code file plots the sensitivity of the effective dimension to different number of samples used for the \
# Monte Carlo estimates for the integrals in the effective dimension formula. m = number of theta samples and we fix \
# n here to a sufficiently large value. In particular, this file looks at models with lower d (lower number of \
# parameters).

path = 'insert_path_to_data_here'
path = path+"data/lower_depth/"
# number of theta samples used
m = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# random seeds
seeds = [1,200,300,400,500,600,700,800,900,1000]

# number of data samples
n = [1000, 2000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000, 10000000, 10000000000, 10000000000000]

# load files
avg_ed4 = []
stdev_4 = []
for ms in m:
    ed_m = []
    for seed in seeds:
        ed = np.load(path+"4in_1layer_effective_dimension_samples_%i_seed_%i.npy" %(ms, seed))
        ed_m += [ed/8]
    ed_m = np.array(ed_m)
    avg_ed4 += [np.mean(ed_m, axis=0)]
    stdev_4 += [np.std(ed_m, axis=0)]

avg_ed6 = []
stdev_6 = []
for ms in m:
    ed_m = []
    for seed in seeds:
        ed = np.load(path+"6in_1layer_effective_dimension_samples_%i_seed_%i.npy" %(ms, seed))
        ed_m += [ed/12]
    ed_m = np.array(ed_m)
    avg_ed6 += [np.mean(ed_m, axis=0)]
    stdev_6 += [np.std(ed_m, axis=0)]

avg_ed8 = []
stdev_8 = []
for ms in m:
    ed_m = []
    for seed in seeds:
        ed = np.load(path+"8in_1layer_effective_dimension_samples_%i_seed_%i.npy" %(ms, seed))
        ed_m += [ed/16]
    ed_m = np.array(ed_m)
    avg_ed8 += [np.mean(ed_m, axis=0)]
    stdev_8 += [np.std(ed_m, axis=0)]

avg_ed10 = []
stdev_10 = []
for ms in m:
    ed_m = []
    for seed in seeds:
        ed = np.load(path+"10in_1layer_effective_dimension_samples_%i_seed_%i.npy" %(ms, seed))
        ed_m += [ed/20]
    ed_m = np.array(ed_m)
    avg_ed10 += [np.mean(ed_m, axis=0)]
    stdev_10 += [np.std(ed_m, axis=0)]

plt.figure(figsize=(12, 4))
plt.subplot(1,3,1)
plt.ylim(ymax=1)
plt.ylabel("Normalised effective dimension")
# pick a fixed n

avg_ed4_n1 = np.transpose(np.array(avg_ed4)[:,3])
stdev_4_n1 = np.transpose(np.array(stdev_4)[:,3])
plt.fill_between(m, avg_ed4_n1-stdev_4_n1, avg_ed4_n1+stdev_4_n1, alpha=0.1)
plt.plot(m, avg_ed4_n1, label="4, d = 8")

avg_ed6_n1 = np.transpose(np.array(avg_ed6)[:,3])
stdev_6_n1 = np.transpose(np.array(stdev_6)[:,3])
plt.fill_between(m, avg_ed6_n1-stdev_6_n1, avg_ed6_n1+stdev_6_n1, alpha=0.1)
plt.plot(m, avg_ed6_n1, label="6, d = 12")

avg_ed8_n1 = np.transpose(np.array(avg_ed8)[:,3])
stdev_8_n1 = np.transpose(np.array(stdev_8)[:,3])
plt.fill_between(m, avg_ed8_n1-stdev_8_n1, avg_ed8_n1+stdev_8_n1, alpha=0.1)
plt.plot(m, avg_ed8_n1, label="8, d = 16")

avg_ed10_n1 = np.transpose(np.array(avg_ed10)[:,3])
stdev_10_n1 = np.transpose(np.array(stdev_10)[:,3])
plt.fill_between(m, avg_ed10_n1-stdev_10_n1, avg_ed10_n1+stdev_10_n1, alpha=0.1)
plt.plot(m, avg_ed10_n1, label="10, d = 20")

plt.legend(title="Input size")
plt.title("n = 1e4")

plt.subplot(1,3,2)
# pick a fixed n
plt.ylim(ymax=1)

avg_ed4_n2 = np.transpose(np.array(avg_ed4)[:,-3])
stdev_4_n2 = np.transpose(np.array(stdev_4)[:,-3])
plt.fill_between(m, avg_ed4_n2-stdev_4_n2, avg_ed4_n2+stdev_4_n2, alpha=0.1)
plt.plot(m, avg_ed4_n2, label="4, d = 8")

avg_ed6_n2 = np.transpose(np.array(avg_ed6)[:,-3])
stdev_6_n2 = np.transpose(np.array(stdev_6)[:,-3])
plt.fill_between(m, avg_ed6_n2-stdev_6_n2, avg_ed6_n2+stdev_6_n2, alpha=0.1)
plt.plot(m, avg_ed6_n2, label="6, d = 12")

avg_ed8_n2 = np.transpose(np.array(avg_ed8)[:,-3])
stdev_8_n2 = np.transpose(np.array(stdev_8)[:,-3])
plt.plot(m, avg_ed8_n2, label="8, d = 16")
plt.fill_between(m, avg_ed8_n2-stdev_8_n2, avg_ed8_n2+stdev_8_n2, alpha=0.1)

avg_ed10_n2 = np.transpose(np.array(avg_ed10)[:,-3])
stdev_10_n2 = np.transpose(np.array(stdev_10)[:,-3])
plt.fill_between(m, avg_ed10_n2-stdev_10_n2, avg_ed10_n2+stdev_10_n2, alpha=0.1)
plt.plot(m, avg_ed10_n2, label="10, d = 20")

plt.legend(title="Input size")
plt.title("n = 1e7")
plt.xlabel("# MC samples")


plt.subplot(1,3,3)
# pick a fixed n
plt.ylim(ymax=1)

avg_ed4_n3 = np.transpose(np.array(avg_ed4)[:,-1])
stdev_4_n3 = np.transpose(np.array(stdev_4)[:,-1])
plt.fill_between(m, avg_ed4_n3-stdev_4_n3, avg_ed4_n3+stdev_4_n3, alpha=0.1)
plt.plot(m, avg_ed4_n3, label="4, d = 8")

avg_ed6_n3 = np.transpose(np.array(avg_ed6)[:,-1])
stdev_6_n3 = np.transpose(np.array(stdev_6)[:,-1])
plt.fill_between(m, avg_ed6_n3-stdev_6_n3, avg_ed6_n3+stdev_6_n3, alpha=0.1)
plt.plot(m, avg_ed6_n3, label="6, d = 12")

avg_ed8_n3 = np.transpose(np.array(avg_ed8)[:,-1])
stdev_8_n3 = np.transpose(np.array(stdev_8)[:,-1])
plt.plot(m, avg_ed8_n3, label="8, d = 16")
plt.fill_between(m, avg_ed8_n3-stdev_8_n3, avg_ed8_n3+stdev_8_n3, alpha=0.1)

avg_ed10_n3 = np.transpose(np.array(avg_ed10)[:,-1])
stdev_10_n3 = np.transpose(np.array(stdev_10)[:,-1])
plt.fill_between(m, avg_ed10_n3-stdev_10_n3, avg_ed10_n3+stdev_10_n3, alpha=0.1)
plt.plot(m, avg_ed10_n3, label="10, d = 20")

plt.legend(title="Input size")
plt.title("n = 1e13")

plt.savefig("sensitivity_lower_depth.pdf", format='pdf', dpi=1000)
plt.show()