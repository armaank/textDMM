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
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextLoader(object):
    """
    Textloader

    todo: add comments
          do I really need this class/functionality?

    

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

    def __init__(self, data_tensor, target_tensor, length_tensor, raw_data):
        assert data_tensor.size(0) == target_tensor.size(0) == length_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.length_tensor = length_tensor
        self.raw_data = raw_data

    def __getitem__(self, index):
        return (
            self.data_tensor[index],
            self.target_tensor[index],
            self.length_tensor[index],
            self.raw_data[index],
        )

    def __len__(self):
        return self.data_tensor.size(0)


def vectorized_data(data, item2id):
    """

    """

    return [
        [item2id[token] if token in item2id else item2id["UNK"] for token in seq]
        for seq, _ in data
    ]


def pad_sequences(vectorized_seqs, seq_lengths):
    """

    """
    # create a zero matrix
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()

    # fill the index
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

    return seq_tensor


def create_dataset(data, input2id, target2id, batch_size=4):
    """

    """

    vectorized_seqs = vectorized_data(data, input2id)

    seq_lengths = torch.LongTensor([len(s) for s in vectorized_seqs])
    seq_tensor = pad_sequences(vectorized_seqs, seq_lengths)
    target_tensor = torch.LongTensor([target2id[y] for _, y in data])
    raw_data = [x for x, _ in data]

    return DataLoader(
        TextDataset(seq_tensor, target_tensor, seq_lengths, raw_data),
        batch_size=batch_size,
    )


def sort_batch(batch, targets, lengths):
    """

    """

    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    target_tensor = targets[perm_idx]

    return seq_tensor.transpose(0, 1), target_tensor, seq_lengths


if __name__ == "__main__":
    """
    testing
    """

    pass
