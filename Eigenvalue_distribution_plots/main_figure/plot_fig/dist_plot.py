import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# This code generates the distribution of eigenvalues plot in the main manuscript

# colors:
rooi = np.array([255, 29, 0])/255
blou = np.array([0, 150, 236])/255
groen = np.array([0,208,0])/255

path = 'specify_path_to_data_here'

# load data
fhat_classical = np.load(path+"/data/fhat4_[4 4 4 2]_fisher.npy")
fhat_easy_qnn = np.load(path+"/data/4qubits_9layer_f_hats_pauli.npy")
fhat_qnn = np.load(path+"/data/4qubits_9layer_f_hats_dep2.npy")

e1=[]
e2=[]
e3=[]

# get the eigenvalues for each sample
for i in range(100):
    e1.append(np.linalg.eigh(fhat_classical[i])[0])
    e2.append(np.linalg.eigh(fhat_easy_qnn[i])[0])
    e3.append(np.linalg.eigh(fhat_qnn[i])[0])


# take the average of the eigenvalues
e1 = np.average(e1, axis=0)
e2 = np.average(e2, axis=0)
e3 = np.average(e3, axis=0)

plt.figure(figsize=(14,4))
plt.subplot(1,3,1)
plt.title('Classical neural network')
counts, bins = np.histogram(e1, bins=np.linspace(np.min(e1), np.max(e1), 40))
mid_points = (bins[1:] + bins[:-1])/2
x1, y1 = mid_points, counts/np.sum(counts)
plt.ylim(0)

# smooth the plot

x_new = np.linspace(np.min(x1), np.max(x1), 10)
a_BSpline = interpolate.make_interp_spline(x1, y1)
y_new = a_BSpline(x_new)
plt.plot(x_new, y_new, color=rooi)
plt.ylabel('density')

###########
counts, bins = np.histogram(e2, bins=np.linspace(np.min(e2), np.max(e2), 40))
mid_points = (bins[1:] + bins[:-1])/2
x2, y2 = mid_points, counts/np.sum(counts)

plt.subplot(1,3,2)
plt.title('Easy quantum model')
plt.ylim((0,1))
x_new = np.linspace(np.min(x2), np.max(x2), 10)
a_BSpline = interpolate.make_interp_spline(x2, y2)
y_new = a_BSpline(x_new)
plt.plot(x_new, y_new, color = blou)
plt.xlabel('value of the eigenvalues')
#########
counts, bins = np.histogram(e3, bins=np.linspace(np.min(e3), np.max(e3), 40))
mid_points = (bins[1:] + bins[:-1])/2
x3, y3 = mid_points, counts/np.sum(counts)
plt.subplot(1,3,3)
plt.title('Quantum neural network')
plt.ylim(0)
x_new = np.linspace(np.min(x3), np.max(x3), 10)
a_BSpline = interpolate.make_interp_spline(x3, y3)
y_new = a_BSpline(x_new)
plt.plot(x_new, y_new, color = groen)
plt.savefig('fisher_3comp_in4_zoomed_out_only.eps', format='eps', dpi=1000)
plt.show()

# repeat the process for just the eigenvalues < 1

###############################################################
e11 = []
for i in range(len(e1)):
    if e1[i] < 1:
        e11.append(e1[i])

e22 = []
for i in range(len(e2)):
    if e2[i] >0 and e2[i] <1:
        e22.append(e2[i])

e33 = []
for i in range(len(e3)):
    if e3[i] >0 and e3[i] <1:
        e33.append(e3[i])

plt.figure(figsize=(14,4))
plt.subplot(1,3,1)
plt.title('Classical neural network')
counts, bins = np.histogram(e11, bins=np.linspace(np.min(e11), np.max(e11), 40))
mid_points = (bins[1:] + bins[:-1])/2
x1, y1 = mid_points, counts/np.sum(counts)
plt.ylim((0,1))
plt.xlim((0,1))
x_new = np.linspace(np.min(x1), np.max(x1), 10)
a_BSpline = interpolate.make_interp_spline(x1, y1)
y_new = a_BSpline(x_new)
plt.ylabel('density')
plt.plot(x_new, y_new, color=rooi)

###########
counts, bins = np.histogram(e22, bins=np.linspace(np.min(e22), np.max(e22), 40))
mid_points = (bins[1:] + bins[:-1])/2
x2, y2 = mid_points, counts/np.sum(counts)

plt.subplot(1,3,2)
plt.ylim((0,1))
plt.xlim((0,1))
x_new = np.linspace(np.min(x2), np.max(x2), 10)
a_BSpline = interpolate.make_interp_spline(x2, y2)
y_new = a_BSpline(x_new)
plt.title('Easy quantum model')
plt.xlabel('value of the eigenvalues')
plt.plot(x_new, y_new, color=blou)

#########
counts, bins = np.histogram(e33, bins=np.linspace(np.min(e33), np.max(e33), 40))
mid_points = (bins[1:] + bins[:-1])/2
x, y = mid_points, counts/np.sum(counts)

plt.subplot(1,3,3)
x_new = np.linspace(np.min(x), np.max(x), 10)
a_BSpline = interpolate.make_interp_spline(x, y)
y_new = a_BSpline(x_new)
plt.ylim((0,1))
plt.xlim((0,1))
plt.plot(x_new, y_new, color=groen)
plt.title('Quantum neural network')
plt.savefig('fisher_3comp_in4_zoomed_in_only.eps', format='eps', dpi=1000)
plt.show()