from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

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

        if not self._check_exists:
            raise RuntimeError('Dataset not found.')

        if self.train:
            data = pd.read_csv(os.path.join(root, 'train.txt'))
            self.train_data = data.iloc[:, :-1]
            self.target = data.iloc[:, -1]
        else:
            data = pd.read_csv(os.path.join(root, 'test.txt'))
            self.test_data = data.iloc[:, :-1]
    
    def __getitem__(self, idx):
        if self.train:
            dataI, targetI = self.train_data.iloc[idx, :], self.target.iloc[idx, :]
            Xi = dataI.astype(np.int32)
            Xv = np.ones_like(dataI)
            return Xi, Xv, targetI
        else:
            dataI = self.test_data.iloc[idx, :]
            Xi = dataI.astype(np.int32)
            Xv = np.ones_like(dataI)
            return Xi, Xv

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(self.root)