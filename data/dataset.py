import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

continous_features = 13

class CriteoDataset(Dataset):
    """
    Custom dataset class for Criteo dataset in order to use efficient 
    dataloader tool provided by PyTorch.
    """ 
    def __init__(self, root, train=True):
        """
        Initialize file path and train/test mode.

        Inputs:
        - root: Path where the processed data file stored.
        - train: Train or test. Required.
        """
        self.root = root
        self.train = train

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.train:
            data = pd.read_csv(os.path.join(root, 'train.txt'))
            self.train_data = data.iloc[:, :-1].values
            self.target = data.iloc[:, -1].values
        else:
            data = pd.read_csv(os.path.join(root, 'test.txt'))
            self.test_data = data.iloc[:, :-1].values
    
    def __getitem__(self, idx):
        if self.train:
            dataI, targetI = self.train_data[idx, :], self.target[idx]
            # index of continous features are zero
            Xi_coutinous = np.zeros_like(dataI[:continous_features])
            Xi_categorial = dataI[continous_features:]
            Xi = torch.from_numpy(np.concatenate((Xi_coutinous, Xi_categorial)).astype(np.int32)).unsqueeze(-1)
            
            # value of categorial features are one (one hot features)
            Xv_categorial = np.ones_like(dataI[continous_features:])
            Xv_coutinous = dataI[:continous_features]
            Xv = torch.from_numpy(np.concatenate((Xv_coutinous, Xv_categorial)).astype(np.int32))
            return Xi, Xv, targetI
        else:
            dataI = self.test_data.iloc[idx, :]
            # index of continous features are one
            Xi_coutinous = np.ones_like(dataI[:continous_features])
            Xi_categorial = dataI[continous_features:]
            Xi = torch.from_numpy(np.concatenate((Xi_coutinous, Xi_categorial)).astype(np.int32)).unsqueeze(-1)
            
            # value of categorial features are one (one hot features)
            Xv_categorial = np.ones_like(dataI[continous_features:])
            Xv_coutinous = dataI[:continous_features]
            Xv = torch.from_numpy(np.concatenate((Xv_coutinous, Xv_categorial)).astype(np.int32))
            return Xi, Xv

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(self.root)
