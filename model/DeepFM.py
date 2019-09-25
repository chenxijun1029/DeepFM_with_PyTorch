# -*- coding: utf-8 -*-

"""
A pytorch implementation of DeepFM for rates prediction problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from time import time


class DeepFM(nn.Module):
    """
    A DeepFM network with RMSE loss for rates prediction problem.

    There are two parts in the architecture of this network: fm part for low
    order interactions of features and deep part for higher order. In this 
    network, we use bachnorm and dropout technology for all hidden layers,
    and "Adam" method for optimazation.

    You may find more details in this paper:
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
    """

    def __init__(self, feature_sizes, embedding_size=4,
                 hidden_dims=[200,200,200],
                 dropout=[0.5, 0.5, 0.5], 
                 
                 use_cuda=True, verbose=False):
        """
        Initialize a new network

        Inputs:
        - feature_size: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - batch_size: An integer giving size of instances used in each interation.
        - use_cuda: Bool, Using cuda or not
        - verbose: Bool
        """
        super().__init__()
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.hidden_dims = hidden_dims
        self.dtype = torch.long
        self.bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        """
            check if use cuda
        """
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        """
            init fm part
        """
        self.fm_first_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
        self.fm_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
        """
            init deep part
        """
        all_dims = [self.field_size * self.embedding_size] + self.hidden_dims
        for i in range(1, len(hidden_dims) + 1):
            setattr(self, 'linear_'+str(i), nn.Linear(all_dims[i-1], all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_'+str(i), nn.Dropout(dropout[i-1]))
        self.avg_acc = None

    def forward(self, Xi, Xv):
        """
        Forward process of network. 

        Inputs:
        - Xi: A tensor of input's index, shape of (N, field_size, 1)
        - Xv: A tensor of input's value, shape of (N, field_size, 1)
        """
        """
            fm part
        """

        # average term
        fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.fm_first_order_embeddings)]
        f1 = torch.cat(fm_first_order_emb_arr, 1)

        # use 2xy = (x+y)^2 - (x^2 + y^2) reduce calculation
        xv = torch.stack([(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.fm_second_order_embeddings)], dim=1)
        xv = F.normalize(xv,dim=1)
        s1 = torch.sum(xv,dim=1).pow(2.0)
        s2 = torch.sum(xv.pow(2.0), dim=1)
        f2 = 0.5 * (s1 - s2)
        self.xv = xv
        self.s1 = s1
        self.s2 = s2

        """
            deep part
        """
        deep_emb = torch.flatten(xv, start_dim=1)
        deep_out = deep_emb
        for i in range(1,len(self.hidden_dims) + 1):
            deep_out = getattr(self, 'linear_' + str(i))(deep_out)
            deep_out = F.relu(getattr(self, 'batchNorm_' + str(i))(deep_out))
            deep_out = getattr(self, 'dropout_' + str(i))(deep_out)

        """
            sum
        """
        self.f1 = f1
        self.f2 = f2
        self.deep_out = deep_out
        total_sum = torch.sum(f1, 1) + torch.sum(f2, 1) + torch.sum(deep_out, 1) + self.bias
        return total_sum

    def l2_reg(self):
        reg_loss = 0
        for param in self.parameters():
            reg_loss += torch.norm(param)
        return reg_loss

    def l1_reg(self):
        reg_loss = 0
        for varname, param in self.named_parameters():
            reg_loss += torch.abs(param).sum()
        return reg_loss


    def fit(self, loader_train, loader_val, optimizer, epochs=1, verbose=False, print_every=100):
        """
        Training a model and valid accuracy.

        Inputs:
        - loader_train: I
        - loader_val: .
        - optimizer: Abstraction of optimizer used in training process, e.g., "torch.optim.Adam()""torch.optim.SGD()".
        - epochs: Integer, number of epochs.
        - verbose: Bool, if print.
        - print_every: Integer, print after every number of iterations. 
        """
        """
            load input data
        """
        model = self.train().to(device=self.device)
        criterion = torch.nn.BCEWithLogitsLoss()
        #criterion = nn.MSELoss()
        self.iter_val = iter(loader_val)
        #l2_loss = self.l2_reg()

        for epoch in range(epochs):
            for t, (xi, xv, y) in enumerate(loader_train):
                xi = xi.to(device=self.device, dtype=self.dtype)
                xv = xv.to(device=self.device, dtype=torch.float32)
                y = y.to(device=self.device, dtype=torch.float32)
                
                total = model(xi, xv)
                reg = self.l1_reg()
                err = criterion(total, y)
                loss = err + 1e-3*reg
                if not torch.isnan(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if verbose and t % print_every == 0:
                    print('Epoch: %d, Iteration %d, loss = %.4f,%.4f,%.4f' % (epoch, t, loss.item(), reg.item(), err.item()))
                    try:
                        self.check_accuracy(self.iter_val, model)
                    except StopIteration:
                        self.iter_val = iter(loader_val)
                    model.train()
                    print()
    
    def check_accuracy(self, loader, model):
        if loader.dataset.train:
            print('Checking accuracy on validation set')
        else:
            print('Checking accuracy on test set')   
        num_correct = 0
        num_samples = 0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            xi, xv, y = next(loader)
            xi = xi.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
            xv = xv.to(device=self.device, dtype=torch.float32)
            y = y.to(device=self.device, dtype=torch.float32)
            total = model(xi, xv)
            preds = (torch.sigmoid(total) > 0.5).type(torch.float32)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

            try:
                acc = float(num_correct) / num_samples
                if self.avg_acc is None:
                    self.avg_acc = acc
                else:
                    self.avg_acc = 0.9 * self.avg_acc + 0.1 * acc
                print('Got %d / %d correct (%.2f%%), avg_acc=%.2f%%' % (num_correct, num_samples, 100 * acc, 100 * self.avg_acc))
            except ZeroDivisionError as e:
                print(e)
                return




                        
