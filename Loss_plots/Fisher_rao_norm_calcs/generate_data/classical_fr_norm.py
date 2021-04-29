from functions_fr import *
import torch
import torch.nn as nn
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.utils import shuffle

# This code trains a fixed classical model 100 times and computes the average Fisher-Rao norm which is equal to \
# W^T*F*W where
# trained parameters = W and F = Fisher information


def create_rand_params(h):
    if type(h) == nn.Linear:
        h.weight.data.uniform_(0, 1)


nnsize = [4,1,1,1,2]
from sklearn import datasets
iris = datasets.load_iris()
x, Y = iris.data, iris.target
x = x[:100]
Y = Y[:100]
num_data = 100
x = normalize(x)
x_train, y_train = x, Y
x_train, y_train = shuffle(x_train, y_train)
x_train = torch.Tensor(x_train).double()
y_train = torch.Tensor(y_train).long()

nrange = [100000000]
for j in range(100):
    np.random.seed(j)
    torch.manual_seed(j)
    model = ANN(size=nnsize)
    model.apply(create_rand_params)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    epochs = 100
    loss_arr = []
    for i in range(epochs):
        y_hat = model.forward(x_train)
        loss = criterion(y_hat, y_train)
        loss_ = loss
        loss_arr.append(loss_.detach().numpy())
        # if i % 1000 == 0:
        print(f'Epoch: {i} Loss: {loss}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    PATH = "model.pt"
    # Save
    torch.save(model, PATH)
    # Load
    model1 = torch.load(PATH)
    ######################################################################################################################
    # NOTE here, the effective dimension is calculated with a fixed value for n and using the trained set of parameters \
    # i.e. 1 parameter set
    cnet = ClassicalNeuralNetwork(nnsize, model1)
    ed = EffectiveDimension(cnet, x_train)
    f = ed.get_fhat()
    effdim = ed.eff_dim(f, nrange)[0]
    filename = 'classical_ed_%d.npy' % j
    np.save(filename, effdim)
    edl = ed.eff_dim(f, nrange)[0]
    opt_params = torch.nn.utils.parameters_to_vector(list(model1.parameters()))
    opt_params = opt_params.detach().numpy()
    o1 = np.reshape(opt_params, (8,1))
    inner = np.dot(f, o1)  # 8 x 1
    o2 = np.reshape(opt_params, (1, 8))
    fisher_rao = np.dot(o2, inner)
    filename = 'classical_frnorm_%d.npy' % j
    np.save(filename, fisher_rao)