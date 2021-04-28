from math import pi
from scipy.special import logsumexp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.utils import shuffle
from abc import ABC, abstractmethod

# This code trains a fixed classical model 100 times and computes the average Fisher-Rao norm which is equal to \
# W^T*F*W where
# trained parameters = W and F = Fisher information


class Model(ABC):
    """
    Abstract base class for classical/quantum models.
    """
    def __init__(self):
        """
        :param thetamin: int, minimum used in uniform sampling of the parameters
        :param thetamax: int,  minimum used in uniform sampling of the parameters
        """
        # Stack data together and combine parameter sets to make calcs more efficient
        self.thetamin = 0
        self.thetamax = 1

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_gradient(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_fisher(self, *args, **kwargs):
        raise NotImplementedError()


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

    class EffectiveDimension:
        def __init__(self, model):
            np.random.seed(0)
            self.num_thetas = 1
            self.model = model
            self.d = model.d
            self.x = normalize(x_train)
            self.num_inputs = len(self.x)

        def get_fhat(self):
            grads = self.model.get_gradient(x=self.x)  # get gradients, dp_theta
            output = self.model.forward(x=self.x)  # get model output
            fishers = self.model.get_fisher(gradients=grads, model_output=output)
            fisher_trace = np.trace(np.average(fishers, axis=0))  # compute the trace with all fishers
            # average the fishers over the num_inputs to get the empirical fishers
            fisher = np.average(np.reshape(fishers, (self.num_thetas, self.num_inputs, self.d, self.d)), axis=1)
            f_hat = self.d * fisher / fisher_trace  # calculate f_hats for all the empirical fishers
            return f_hat

        def eff_dim(self, f_hat, n):
            effective_dim = []
            for ns in n:
                Fhat = f_hat * ns / (2 * pi * np.log(n))
                one_plus_F = np.eye(self.d) + Fhat
                det = np.linalg.slogdet(one_plus_F)[1]  # log det because of overflow
                r = det / 2  # divide by 2 because of sqrt
                effective_dim.append(2 * (logsumexp(r) - np.log(self.num_thetas)) / np.log(ns / (2 * pi * np.log(n))))
            return effective_dim


    class ClassicalNeuralNetwork(Model, nn.Module):
        def __init__(self, size):
            Model.__init__(self)
            nn.Module.__init__(self)
            self.size = size
            self.inputsize = size[0]
            self.outputsize = size[-1]
            self.d = sum(size[i] * size[i + 1] for i in range(len(size) - 1))

        def forward(self, x):
            if not torch.is_tensor(x):
                x = torch.from_numpy(x)
            net = model1
            return F.softmax(net(x), dim=-1)

        def get_gradient(self, x):
            if not torch.is_tensor(x):
                x = torch.from_numpy(x)
                x.requires_grad_(False)
            gradvectors = []
            for m in range(len(x)):
                net = model1
                output = F.softmax(net(x[m]), dim=-1)
                output = torch.max(output, torch.tensor(1e-20).double())
                logoutput = torch.log(output)  # get the output values to calculate the jacobian
                grad = []
                for i in range(self.outputsize):
                    net.zero_grad()
                    logoutput[i].backward(retain_graph=True)
                    grads = []
                    for param in net.parameters():
                        grads.append(param.grad.view(-1))
                    gr = torch.cat(grads)
                    grad.append(gr * torch.sqrt(output[i]))
                jacobian = torch.cat(grad)
                jacobian = torch.reshape(jacobian, (self.outputsize, self.d))
                gradvectors.append(jacobian.detach().numpy())
            return gradvectors

        def get_fisher(self, gradients, model_output):
            fishers = np.zeros((len(gradients), self.d, self.d))
            for i in range(len(gradients)):
                grads = gradients[i]
                temp_sum = np.zeros((self.outputsize, self.d, self.d))
                for j in range(self.outputsize):
                    temp_sum[j] += np.array(np.outer(grads[j], np.transpose(grads[j])))
                fishers[i] += np.sum(temp_sum, axis=0)
            return fishers


    nrange = [100000000]

    cnet = ClassicalNeuralNetwork(nnsize)
    ed = EffectiveDimension(cnet)
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


