"""data utils
"""
import torch

import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import Counter, defaultdict
from torch.utils.data import Dataset, DataLoader


class TextLoader(object):
    """
    Textloader

    modified PyTorch DataLoader

    Properties
    ----------

    """

    def __init__(self):

        pass

    def read_text(self, fpath):
        """
        read text from input file
        """

        return tokens, data

    def split_set(self, data):
        """
        perform training, testing, validation split
        """

        return train_set, val_set, test_set


class TextDataset(Dataset):
    """
    TextDataset

    modified PyTorch Dataset

    Properties
    ----------

    """

    def __init__(self):

        pass

    def __getitem__(self, index):

        return 0

    def __len__(self):

        return self.data.size(0)
