"""datahandler

TODO: allow option to use a subset of train/test/val
      make better comments
"""

import os

import numpy as np
import torch
import torchtext

from torch import nn
from torchtext import data

unk_token = "<unk>"
MIN_LEN = 2


class lmDataset(data.Dataset):
    """
    a simple dataset class for a character-level language model
    basically a wrapper for a torchtext.data.Dataset object
    """

    def __init__(self, path, text_field, encoding="utf-8", eos=True, **kwargs):

        fields = [("text", text_field)]
        examples = []

        with open(path, encoding=encoding) as f:
            for line in f:
                text = text_field.preprocess(line)
                if eos:
                    text += [u"<eos"]
                examples.append(data.Example.fromlist([text], fields))

        super().__init__(examples, fields, **kwargs)


def load_data(dataset_fpath, max_len):
    """
    load_data
        - loads data from a dataset as a lmDataset object
        -
    """
    text = torchtext.data.Field(
        include_lengths=True, unk_token=unk_token, tokenize=(lambda s: list(s.strip()))
    )

    train, val, test = lmDataset.splits(
        path=dataset_fpath,
        train="ptb.train.txt",
        validation="ptb.valid.txt",
        test="ptb.test.txt",
        text_field=text,
        eos=True,
        filter_pred=lambda x: len(vars(x)["text"]) <= max_len
        and len(vars(x)["text"]) >= MIN_LEN,
    )

    text.build_vocab(train)
    pad_val = text.vocab.stoi["<pad>"]

    return train, val, test, text.vocab
