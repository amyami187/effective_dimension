import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.utils import shuffle
from abc import ABC, abstractmethod

# Train a classical neural network 100 times with size = [4,1,1,1,2]


class ANN(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.layers = nn.ModuleList(
            [nn.Linear(self.size[i - 1], self.size[i], bias=False) for i in range(1, len(self.size))]).double()
        self.d = sum(size[i] * size[i + 1] for i in range(len(size) - 1))

    def forward(self, x):
        for i in range(len(self.size) - 2):
            x = F.leaky_relu(self.layers[i](x))
        x = self.layers[-1](x)
        return x


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


class IrisDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
    
iris_ds = IrisDataset(x_train, y_train)
# batch_size = 100; lrn= 0.1  # to get paper results; loss often gets stuck in a plateau at 0.69 or 0.34;
batch_size = 1; lrn= 0.5  # only occasionally gets stuck in plateau; often reaches near 0 loss.

train_dl = torch.utils.data.DataLoader(iris_ds, batch_size=batch_size, shuffle=False)

for j in range(100):
    np.random.seed(j)
    torch.manual_seed(j)
    model = ANN(size=nnsize)
    model.apply(create_rand_params)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrn)
    epochs = 100
    loss_arr = []
    for i in range(epochs):
        model.train()
        loss_ = 0
        for k, (x_train, y_train ) in enumerate(train_dl):
            y_hat = model.forward(x_train)
            loss = criterion(y_hat, y_train)
            loss_ += loss / len(train_dl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if loss < 0.01:
            #    break
        loss_arr.append(loss_.detach().numpy())
    
        
            
    print(f'Run: {j} Loss(train): {loss_arr[-1]}')
    
    filename = 'classical_loss_%d.npy' % j
    np.save(filename, loss_arr)
    PATH = "../model_20.pt"
    # Save
    torch.save(model, PATH)
    # Load
    model1 = torch.load(PATH)
