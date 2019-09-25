import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class CriteoDataset(Dataset):
    """
    Custom dataset class for Criteo dataset in order to use efficient 
    dataloader tool provided by PyTorch.
    """ 
    def __init__(self, root, train=True, train_file="train.txt", test_file="test.txt"):
        """
        Initialize file path and train/test mode.

        Inputs:
        - root: Path where the processed data file stored.
        - train: Train or test. Required.
        """
        self.continous_features = range(0, 13)
        self.categorial_features = range(13, 39)
        self.root = root
        self.train = train

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.train:
            data = pd.read_csv(os.path.join(root, train_file))
            self.train_data = data.iloc[:, :-1].values
            self.target = data.iloc[:, -1].values
        else:
            data = pd.read_csv(os.path.join(root, test_file))
            self.test_data = data.iloc[:, :-1].values
    
    def __getitem__(self, idx):
        if self.train:
            dataI, targetI = self.train_data[idx, :], self.target[idx]
            Xi = torch.cat([torch.zeros(len(self.continous_features),dtype=torch.int64), torch.tensor(dataI[self.categorial_features])], dim=0).unsqueeze(-1)
            Xv = torch.cat([torch.tensor(dataI[self.continous_features],dtype=torch.float32), torch.ones(len(self.categorial_features),dtype=torch.float32)], dim=0).type(torch.float32)
            #Xi = torch.from_numpy(dataI.astype(np.int32)).unsqueeze(-1)
            #Xv = torch.from_numpy(np.ones_like(dataI))
            return Xi, Xv, targetI
        else:
            dataI = self.test_data[idx, :]
            breakpoint()
            #Xi = torch.from_numpy(dataI.astype(np.int32)).unsqueeze(-1)
            #Xv = torch.from_numpy(np.ones_like(dataI))
            Xi = torch.cat([torch.zeros(len(self.continous_features),dtype=torch.int64), torch.tensor(dataI[self.categorial_features])], dim=0).unsqueeze(-1)
            Xv = torch.cat([torch.tensor(dataI[self.continous_features],dtype=torch.float32), torch.ones(len(self.categorial_features),dtype=torch.float32)], dim=0)
            return Xi, Xv

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(self.root)
