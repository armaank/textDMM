"""data utils
"""
import os
import random

import torch

import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import Counter, defaultdict
from torch.utils.data import Dataset


class TextLoader(object):
    """
    Textloader

    

    Properties
    ----------

    """

    def __init__(self, fpath):

        self.token2id = defaultdict(int)

        self.tokens, dataset = self.read_text(fpath)

        self.train_set, self.val_set, self.test_set = self.split_set(dataset)

        self.token2id = self.set2id(self.tokens, "PAD", "UNK")

        self.tag2id = self.set2id(set(dataset.keys()))

        pass

    def read_text(self, fpath):
        """
        read text from input file
        """

        fnames = os.listdir(fpath)
        tokens = set()
        data = defaultdict(list)

        for f in fnames:
            if not f.endswith("txt"):
                continue

            cat = f.replace(".txt", "")

            with open(os.path.join(fpath, f)) as f:
                for line in f:
                    line = line.strip().lower()
                    data[cat].append(line)
                    for token in line:
                        tokens.add(token)

        return tokens, dataset

    def split_set(self, dataset):
        """
        perform training, testing, validation split
        """

        train_dataset = []
        val_dataset = []
        test_dataset = []

        all_data = []
        for cat in data:
            cat_data = data[cat]
            # print(cat, len(data[cat]))
            all_data += [(dat, cat) for dat in cat_data]

        all_data = random.sample(all_data, len(all_data))

        train_ratio = int(len(all_data) * 0.8)
        val_ratio = int(len(all_data) * 0.9)

        train_dataset = all_data[:train_ratio]
        val_dataset = all_data[train_ratio:dev_ratio]
        test_dataset = all_data[dev_ratio:]

        return train_dataset, val_dataset, test_datset

    def set2id(self, item_set, pad=None, unk=None):

        item2id = defaultdict(int)

        if pad is not None:
            item2id[pad] = 0

        if unk is not None:
            item2id[unk] = 1

        for item in item_set:
            item2id[item] = len(item2id)

        return item2id


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


if __name__ == "__main__":
    """
    testing
    """

    pass
