import numpy as np
import matplotlib.pyplot as plt

# This code generates the distribution of eigenvalues plot in the Supplementary Information

# colors:
rooi = np.array([255, 29, 0])/255
blou = np.array([0, 150, 236])/255
groen = np.array([0,208,0])/255

path = 'insert path here'
# load data
fhat_classical = np.load(path +'/data/fhat6_[6 7 2 2].npy')
fhat_easy_qnn = np.load(path +"/data/6qubits_9layer_f_hats_pauli_Z.npy")
fhat_qnn = np.load(path+"/data/6qubits_9layer_f_hats_dep2.npy")

plt.figure(figsize=(11,9))

e1=[]
e2=[]
e3=[]

# get eigenvalues for each sample

for i in range(100):
    e1.append(np.linalg.eigh(fhat_classical[i])[0])
    e2.append(np.linalg.eigh(fhat_easy_qnn[i])[0])
    e3.append(np.linalg.eigh(fhat_qnn[i])[0])

# take the average of the eigenvalues
e1 = np.average(e1, axis=0)
e2 = np.average(e2, axis=0)
e3 = np.average(e3, axis=0)

plt.subplot(3,3,1)
plt.title('Classical neural network')
counts, bins = np.histogram(e1, bins=np.linspace(np.min(e1), np.max(e1), 40))
cumulative = np.cumsum(counts)/np.sum(counts)
# plot the cumulative function
plt.plot(bins[:-1], cumulative, c=rooi, label='A1')
plt.ylim((0,1))
plt.legend()

###########
plt.subplot(3,3,2)

plt.title('Easy quantum model')
counts, bins = np.histogram(e2, bins=np.linspace(np.min(e2), np.max(e2), 40))
cumulative = np.cumsum(counts)/np.sum(counts)
# plot the cumulative function
plt.plot(bins[:-1], cumulative, c=blou, label='A2')
plt.legend()
plt.ylim((0,1))
#########
counts, bins = np.histogram(e3, bins=np.linspace(np.min(e3), np.max(e3), 40))
plt.subplot(3,3,3)
plt.title('Quantum neural network')
cumulative = np.cumsum(counts)/np.sum(counts)
# plot the cumulative function
plt.plot(bins[:-1], cumulative, c=groen, label='A3')
plt.legend()
plt.ylim((0,1))

# repeat for input size = 8

fhat_classical = np.load(path +"/data/fhat8_[8 8 2].npy")
fhat_easy_qnn = np.load(path+"/data/8qubits_9layer_f_hats_easy.npy")
fhat_qnn = np.load(path+"/data/8qubits_9layer_f_hats_dep2.npy")

e1=[]
e2=[]
e3=[]

for i in range(100):
    e1.append(np.linalg.eigh(fhat_classical[i])[0])
    e2.append(np.linalg.eigh(fhat_easy_qnn[i])[0])
    e3.append(np.linalg.eigh(fhat_qnn[i])[0])


e1 = np.average(e1, axis=0)
e2 = np.average(e2, axis=0)
e3 = np.average(e3, axis=0)

plt.subplot(3,3,4)
counts, bins = np.histogram(e1, bins=np.linspace(np.min(e1), np.max(e1), 40))
cumulative = np.cumsum(counts)/np.sum(counts)
# plot the cumulative function
plt.plot(bins[:-1], cumulative, c=rooi, label='B1')
plt.legend()
plt.ylabel('density')
plt.legend()
plt.ylim((0,1))
###########
counts, bins = np.histogram(e2, bins=np.linspace(np.min(e2), np.max(e2), 40))
plt.subplot(3,3,5)
cumulative = np.cumsum(counts)/np.sum(counts)
# plot the cumulative function
plt.plot(bins[:-1], cumulative, c=blou, label='B2')
plt.legend()
plt.ylim((0,1))
#########
counts, bins = np.histogram(e3, bins=np.linspace(np.min(e3), np.max(e3), 40))
plt.subplot(3,3,6)
cumulative = np.cumsum(counts)/np.sum(counts)
# plot the cumulative function
plt.plot(bins[:-1], cumulative, c=groen, label='B3')
plt.legend()
plt.ylim((0,1))
# repeat for input size = 10
fhat_classical = np.load(path+"/data/fhat10_[10  8  1  4  2].npy")
fhat_easy_qnn = np.load(path+"/data/10qubits_9layer_f_hats_easy.npy")
fhat_qnn = np.load(path+"/data/10qubits_9layer_f_hats_dep2.npy")

e1=[]
e2=[]
e3=[]

for i in range(100):
    e1.append(np.linalg.eigh(fhat_classical[i])[0])
    e2.append(np.linalg.eigh(fhat_easy_qnn[i])[0])
    e3.append(np.linalg.eigh(fhat_qnn[i])[0])


e1 = np.average(e1, axis=0)
e2 = np.average(e2, axis=0)
e3 = np.average(e3, axis=0)

plt.subplot(3,3,7)
counts, bins = np.histogram(e1, bins=np.linspace(np.min(e1), np.max(e1), 40))
cumulative = np.cumsum(counts)/np.sum(counts)
# plot the cumulative function
plt.plot(bins[:-1], cumulative, c=rooi, label='C1')
plt.legend()
plt.ylim((0,1))

###########
counts, bins = np.histogram(e2, bins=np.linspace(np.min(e2), np.max(e2), 40))
plt.subplot(3,3,8)
cumulative = np.cumsum(counts)/np.sum(counts)
# plot the cumulative function
plt.plot(bins[:-1], cumulative, c=blou, label='C2')
plt.legend()
plt.xlabel('value of the eigenvalues')
plt.ylim((0,1))
#########
counts, bins = np.histogram(e3, bins=np.linspace(np.min(e3), np.max(e3), 40))
plt.subplot(3,3,9)
cumulative = np.cumsum(counts)/np.sum(counts)
# plot the cumulative function
plt.plot(bins[:-1], cumulative, c=groen, label='C3')
plt.legend()
plt.ylim((0,1))
plt.savefig('cum_plot.eps', format='eps', dpi=1000)

plt.show()
