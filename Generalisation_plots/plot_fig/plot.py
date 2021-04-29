import numpy as np
import matplotlib.pyplot as plt

# This code creats the plots in the Supplementary information where we compute the effective \
# dimension as well as the generalisation error and see how they correlate

path ='insert_path_to_data_here'
effdims = np.zeros((10, 5))
losses = np.zeros((10, 5))

# color
groen = np.array([0,106,0])/255

# load data
for i in range(10):
    for j in range(0,5):
        filename = path+"data/data_s%s_e%s.npy" % (i,j)
        l = np.load(filename, allow_pickle=True)
        effdims[i, j] += l[1]
        losses[i, j] += l[0]

effdims = effdims/880  # normalise by d = 880
avs = np.average(effdims, axis=0)
std = np.std(effdims, axis=0)
avsl = np.average(losses, axis=0)
stdl = np.std(losses, axis=0)

errors = [0, 0.1, 0.2, 0.3, 0.4]

nq = []
sd = []
edim = []
for i in range(0,200):
    edim.append(i/200)
    sd.append(np.interp(i/200, errors, std))
    nq.append(np.interp(i/200, errors, avs))

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False, figsize=(12, 4))
plt.subplot(1,2,1)
plt.ylabel('normalized effective dimension')
plt.xlim(xmin=0, xmax=0.5)
plt.plot(edim, nq, color=groen)
plt.fill_between(edim, np.array(nq)+np.array(sd), np.array(nq)-np.array(sd), alpha=0.2, color=groen)

nq = []
sd = []
loss = []
for i in range(0,200):
    loss.append(i/200)
    sd.append(np.interp(i/200, errors, stdl))
    nq.append(np.interp(i/200, errors, avsl))

plt.subplot(1,2,2)
plt.ylabel('test error')
plt.plot(edim, nq, color=groen)
plt.fill_between(loss, np.array(nq)+np.array(sd), np.array(nq)-np.array(sd), alpha=0.2, color=groen)
plt.xlim(xmin=0, xmax=0.5)
# create shared axis labels
fig.text(0.5, 0.01, '% of randomized labels', ha='center')
plt.savefig('generalisation.pdf', format='pdf', dpi=1000)
plt.show()
