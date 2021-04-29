import torch
import torch.nn as nn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import numpy as np
from helper_functions import ANN, ClassicalNeuralNetwork, EffectiveDimension
from sklearn.utils import shuffle
np.random.seed(42)

# We train a classical neural network to zero training error 10 times and randomise the labels \
# This script has 0% label randomisation
# This data goes into the generalisation error plots in the Supplementary information where we compute the effective \
# dimension as well as the generalisation error and see how they correlate

# NOTE here, the effective dimension is calculated with a fixed value for n and using the trained set of parameters \
# i.e. 1 parameter set

nrange = [100000000]
nnsize=[6, 110, 2]
num_data = 1000

for l in range(100):
    torch.manual_seed(l)

    def create_rand_params(h):
        if type(h) == nn.Linear:
            h.weight.data.uniform_(0, 1)
    model = ANN(size=nnsize)
    model.apply(create_rand_params)
    x, Y = make_blobs(num_data, n_features=6, centers=2, random_state=42)
    x = normalize(x)
    x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.4, random_state=0)
    x_train, y_train = shuffle(x_train, y_train)
    x_train = torch.Tensor(x_train).double()
    y_train = torch.Tensor(y_train).long()
    x_test= torch.Tensor(x_test).double()
    y_test = torch.Tensor(y_test).long()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 100000
    loss_arr = []
    for i in range(epochs):
        y_hat = model.forward(x_train)
        loss = criterion(y_hat, y_train)
        loss_arr.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    preds = model.forward(x_test)
    test_loss = criterion(preds, y_test)
    print(test_loss)
    PATH = "model_20.pt"
    # Save
    torch.save(model, PATH)
    # Load
    model1 = torch.load(PATH)
    ######################################################################################################################
    cnet = ClassicalNeuralNetwork(nnsize, model1)
    print("true dimension:", cnet.d)
    ed = EffectiveDimension(cnet, x_train)
    f = ed.get_fhat()
    print("effective dimension:", ed.eff_dim(f, nrange)[0]/880)
    error = np.array([0])
    test_l = test_loss.detach().numpy()
    edl = ed.eff_dim(f, nrange)[0]
    data = np.array([test_l, edl, error], dtype=object)
    filename = "data_s%s_e0.npy" % l
    np.save(filename, data)