from math import pi
from scipy.special import logsumexp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize
import numpy as np


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


class EffectiveDimension:
    def __init__(self, model, x_train):
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


class ClassicalNeuralNetwork(nn.Module):
    def __init__(self, size, model1):
        nn.Module.__init__(self)
        self.size = size
        self.inputsize = size[0]
        self.outputsize = size[-1]
        self.d = sum(size[i] * size[i + 1] for i in range(len(size) - 1))
        self.model1 = model1

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        net = self.model1
        return F.softmax(net(x), dim=-1)

    def get_gradient(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
            x.requires_grad_(False)
        gradvectors = []
        for m in range(len(x)):
            net = self.model1
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