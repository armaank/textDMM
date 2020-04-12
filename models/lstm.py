"""lstm
"""

import numpy as np
import torch

import torch.autograd as autograd
import torch.functional as F
import torch.optim as optim
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMCAT(nn.Module):
    """
    LSTMCAT

    A simple LSTM network for text categorization
    todo: add choice of embedding layer (normal or vq)

    Properties
    ----------
    dict_size: size of dictionary for embedding layer
                [int]
    embed_dim: size of each embedding vector
                [int]
    hidden_dim: the number of features in each hidden state
                [int]
    output_size: output shape of final softmax layer
                [int]
    
    """

    def __init__(self, dict_size, embed_dim, hidden_dim, output_size):

        super(LSTMCAT, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dict_size = dict_size

        # init layers
        self.embedding = nn.Embedding(dict_size, embed_dim)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1)

        self.hidden = nn.Linear(hidden_dim, output_size)

        self.softmax = nn.LogSoftmax()

        self.dropout_layer = nn.Dropout(p=0.2)

        pass

    def init_hidden(self, batch_size):
        """
        initilize hidden layer
        """
        h1 = autograd.Variable(torch.randn(1, batch_size, self.hidden_dim))
        h2 = autograd.Variable(torch.randn(1, batch_size, self.hidden_dim))

        return (h1, h2)

    def forward(self, batch, lengths):
        """
        forward pass through lastm network
        """

        self.hidden = self.init_hidden(batch.size(-1))

        embeds = self.embedding(batch)
        packed_input = pack_padded_sequence(embeds, lengths)
        outputs, (ht, ct) = self.lstm(packed_input, self.hidden)

        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        output = self.dropout_layer(ht[-1])
        output = self.hidden(output)
        output = self.softmax(output)

        return output
