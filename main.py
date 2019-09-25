import numpy as np
import math
import torch
import torch.optim as optim
import radam
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import random

from model.DeepFM import DeepFM
from data.dataset import CriteoDataset

def split_train_and_valid(train_data):
    total_set = len(train_data)
    samples_idx = np.arange(0,total_set)
    np.random.shuffle(samples_idx)
    num_train = math.floor(len(samples_idx)*0.8)
    num_valid = len(samples_idx) - num_train
    train_idx = samples_idx[:num_train]
    valid_idx = samples_idx[num_train:]
    return train_idx, valid_idx

seed = 20170705
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
train_file = "train_large.txt"
test_file = "test_large.txt"
#train_file = "train.txt"
#test_file = "test.txt"

# load data
train_data = CriteoDataset('./data', train=True, train_file=train_file)

# split trani and valid set
train_idx, valid_idx = split_train_and_valid(train_data)

# loader
loader_train = DataLoader(train_data, batch_size=128, sampler=sampler.SubsetRandomSampler(train_idx), num_workers=7)
loader_val = DataLoader(train_data, batch_size=1000, sampler=sampler.SubsetRandomSampler(valid_idx), num_workers=7)

feature_sizes = np.loadtxt('./data/feature_sizes.txt', delimiter=',')
feature_sizes = [int(x) for x in feature_sizes]
print(feature_sizes)

model = DeepFM(feature_sizes, use_cuda=True)
#optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
optimizer = radam.RAdam(model.parameters(), lr=1e-3, weight_decay=0.0)
model.fit(loader_train, loader_val, optimizer, epochs=10, verbose=True, print_every=1000)
