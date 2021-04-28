import numpy as np
import matplotlib.pyplot as plt

# This code produces the loss/training plot for Figure 3 b).


path = 'insert_path_to_data_folder_here'

# colors:
rooi = np.array([255, 29, 0])/255
blou = np.array([0, 150, 236])/255
groen = np.array([0,208,0])/255

# Load classical data
loss = np.zeros((100,100))
for i in range(100):
    file = path + 'data/classical/classical_loss_%d.npy'%i
    loss[i] += np.load(file, allow_pickle=True)

sd = np.std(loss, axis=0)
av = np.average(loss, axis=0)
plt.plot(range(100), av, label='classical neural network', color=rooi)
plt.fill_between(range(100), av+np.array(sd), av-np.array(sd), alpha=0.1, color=rooi)

# Load easy qnn data
loss_eqnn_d1 = np.load(path+'data/easy_qnn/quantum_loss_easy_99.npy')
loss_eqnn_d1 = np.reshape(np.array(loss_eqnn_d1), (100, 100))
sd = np.std(loss_eqnn_d1, axis=0)
av = np.average(loss_eqnn_d1, axis=0)
plt.fill_between(range(100), av+np.array(sd), av-np.array(sd), alpha=0.1, color=blou)
plt.plot(range(100), av, label='easy quantum model', color=blou)

# Load hard qnn data
loss = np.zeros((100,100))
for i in range(100):
    file = path+'data/hard_qnn/quantum_loss_hard_dep2_%d.npy'%i
    loss[i] += np.load(file, allow_pickle=True)

sd = np.std(loss, axis=0)
av = np.average(loss, axis=0)
print(av)
plt.plot(range(100), av, label='quantum neural network', color=groen)
plt.fill_between(range(100), av+np.array(sd), av-np.array(sd), alpha=0.1, color=groen)


# IBMQ Montreal raw data
loss_ibmq_montreal = [
    0.5864, 0.5115, 0.4597, 0.4062, 0.3654, 0.3390, 0.3330, 0.3339, 0.3241, 0.3276, # 10
    0.3234, 0.3038, 0.2978, 0.2728, 0.2598, 0.2575, 0.2486, 0.2564, 0.2653, 0.2712, # 20
    0.2668, 0.2809, 0.2638, 0.2652, 0.2551, 0.2453, 0.2386, 0.2543, 0.2440, 0.2404, # 30
    0.2417, 0.2278, 0.2235]

loss_ibmq_montreal_with_stable = [
    0.5864, 0.5115, 0.4597, 0.4062, 0.3654, 0.3390, 0.3330, 0.3339, 0.3241, 0.3276, # 10
    0.3234, 0.3038, 0.2978, 0.2728, 0.2598, 0.2575, 0.2486, 0.2564, 0.2653, 0.2712, # 20
    0.2668, 0.2809, 0.2638, 0.2652, 0.2551, 0.2453, 0.2386, 0.2543, 0.2440, 0.2404, # 30
    0.2417, 0.2278, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235,
    0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235,
    0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235,
    0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235,
    0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235,
    0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235,
    0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235, 0.2235]


plt.plot(loss_ibmq_montreal, label='ibmq_montreal backend', color='black')
plt.plot(loss_ibmq_montreal_with_stable, '--', color='black')
plt.ylabel('loss value')
plt.xlabel('number of training iterations')
plt.legend()
plt.savefig('loss_with_std_dev.pdf', format='pdf', dpi=1000)
plt.show()