import os
import sys
import random

import numpy as np
import torch
from torchvision import datasets
from PIL import Image

from torch.utils.data import ConcatDataset, Dataset

# for openml datasets
import openml
from sklearn.preprocessing import LabelEncoder

def get_dataset(name):
    if 'CIFAR10' in name and 'CIFAR100' not in name:
        if name[-2:] == '04':
            return get_CIFAR10(0.4)
        elif name[-2:] == '06':
            return get_CIFAR10(0.6)
        else:
            raise NotImplementedError
    elif 'CIFAR100' in name:
        if name[-2:] == '04':
            return get_CIFAR100(0.4)
        elif name[-2:] == '06':
            return get_CIFAR100(0.6)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

def get_CIFAR10(ratio):
    data_tr = datasets.CIFAR10('./../data/CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10('./../data/CIFAR10', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets)).long()
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets)).long()

    label_set = [0,1,2,3,4,5,6,7,8,9]
    ID_label_num = int(10 * (1.0 - ratio))
    OOD_label_num = int(10 * ratio)
    ID_labels = label_set[0:ID_label_num]
    OOD_labels = label_set[ID_label_num:]
    X_tr_id = []
    X_te_id = []
    Y_tr_id = []
    Y_te_id = []

    X_tr_ood = []
    Y_tr_ood = []

    random.seed(4666)
    nan_num = torch.tensor(-1)
    for i in range(Y_tr.shape[0]):
        if Y_tr[i] in ID_labels:
            X_tr_id.append(X_tr[i])
            Y_tr_id.append(Y_tr[i])
        else:
            X_tr_ood.append(X_tr[i])
            Y_tr_ood.append(Y_tr[i])
    for i in range(Y_te.shape[0]):
        if Y_te[i] in ID_labels:
            X_te_id.append(X_te[i])
            Y_te_id.append(Y_te[i])
    X_tr_fin = X_tr_id + X_tr_ood
    Y_tr_id = torch.tensor(np.array(Y_tr_id)).type_as(Y_tr)
    Y_tr_ood = nan_num * torch.ones(len(Y_tr_ood)).type_as(Y_tr)
    
    

    Y_tr_fin = torch.cat((Y_tr_id, Y_tr_ood),0).type_as(Y_tr)

    X_tr_fin = np.array(X_tr_fin).astype(X_tr.dtype)

    X_te_id = np.array(X_te_id).astype(X_te.dtype)
    Y_te_id = torch.tensor(np.array(Y_te_id))
    return X_tr_fin, Y_tr_fin, X_te_id, Y_te_id

def get_CIFAR100(ratio):
    data_tr = datasets.CIFAR100('./../data/CIFAR100', train=True, download=True)
    data_te = datasets.CIFAR100('./../data/CIFAR100', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets)).long()
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets)).long()

    label_set = list(range(0,100))
    ID_label_num = int(100 * (1.0 - ratio))
    OOD_label_num = int(100 * ratio)
    ID_labels = label_set[0:ID_label_num]
    OOD_labels = label_set[ID_label_num:]
    X_tr_id = []
    X_te_id = []
    Y_tr_id = []
    Y_te_id = []

    X_tr_ood = []
    Y_tr_ood = []

    random.seed(4666)
    nan_num = torch.tensor(-1)
    for i in range(Y_tr.shape[0]):
        if Y_tr[i] in ID_labels:
            X_tr_id.append(X_tr[i])
            Y_tr_id.append(Y_tr[i])
        else:
            X_tr_ood.append(X_tr[i])
            Y_tr_ood.append(Y_tr[i])
    for i in range(Y_te.shape[0]):
        if Y_te[i] in ID_labels:
            X_te_id.append(X_te[i])
            Y_te_id.append(Y_te[i])
    X_tr_fin = X_tr_id + X_tr_ood
    Y_tr_id = torch.tensor(np.array(Y_tr_id)).type_as(Y_tr)
    Y_tr_ood = nan_num * torch.ones(len(Y_tr_ood)).type_as(Y_tr)
    X_tr_fin = np.array(X_tr_fin).astype(X_tr.dtype)


    Y_tr_fin = torch.cat((Y_tr_id, Y_tr_ood),0)

    X_te_id = np.array(X_te_id).astype(X_te.dtype)
    Y_te_id = torch.tensor(np.array(Y_te_id)).type_as(Y_te)
    return X_tr_fin, Y_tr_fin, X_te_id, Y_te_id



def get_handler(name):
    if 'CIFAR' in name:
        return DataHandler3
    else:
        raise NotImplementedError


class DataHandler3(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler4(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)
