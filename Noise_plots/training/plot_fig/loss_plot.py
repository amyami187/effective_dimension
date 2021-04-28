import numpy as np
import matplotlib.pyplot as plt

# This code produces the loss/training plot in the Supplementary Information that includes noise

path = 'insert_path_to_data_here'

# colors:
blou = np.array([0, 150, 236])/255
groen = np.array([0,208,0])/255
lig_blou = np.array([32, 26, 124])/255
lig_groen = np.array([0, 108, 22])/255

# Easy qnn data
loss_eqnn_d1 = np.load(path+'data/easy_qnn/quantum_loss_easy_99.npy')
loss_eqnn_d1 = np.reshape(np.array(loss_eqnn_d1), (100, 100))
sd = np.std(loss_eqnn_d1, axis=0)
av = np.average(loss_eqnn_d1, axis=0)
plt.fill_between(range(100), av+np.array(sd), av-np.array(sd), alpha=0.1, color=blou)
plt.plot(range(100), av, label='easy quantum model', color=blou)

# Easy qnn with noise
loss_nh = np.zeros((100,100))
for i in range(100):
    file = path+'data/train_easy_noisy/quantum_loss_easy_noise_%d.npy'%i
    loss_nh[i] += np.load(file, allow_pickle=True)
sd = np.std(loss_nh, axis=0)
av = np.average(loss_nh, axis=0)

plt.plot(range(100), av, '--', label='easy qnn with noise', color=lig_blou)
plt.fill_between(range(100), av+np.array(sd), av-np.array(sd), alpha=0.1, color=lig_blou)


# hard qnn data
loss = np.zeros((100,100))
for i in range(100):
    file = path+'data/hard_qnn/quantum_loss_hard_dep2_%d.npy'%i
    loss[i] += np.load(file, allow_pickle=True)

sd = np.std(loss, axis=0)
av = np.average(loss, axis=0)
plt.plot(range(100), av, label='quantum neural network', color=groen)
plt.fill_between(range(100), av+np.array(sd), av-np.array(sd), alpha=0.1, color=groen)

# hard qnn with noise
loss_nh = np.zeros((100,100))
for i in range(100):
    file = path+'data/train_hard_noisy/quantum_loss_hard_noise_%d.npy'%i
    loss_nh[i] += np.load(file, allow_pickle=True)
sd = np.std(loss_nh, axis=0)
av = np.average(loss_nh, axis=0)

plt.plot(range(100), av, '--', label='quantum nn with noise', color = lig_groen)
plt.fill_between(range(100), av+np.array(sd), av-np.array(sd), alpha=0.1, color=lig_groen)

plt.ylabel('loss value')
plt.xlabel('number of training iterations')
plt.legend()
plt.savefig('loss_with_noise_std_dev.pdf', format='pdf', dpi=1000)
plt.show()