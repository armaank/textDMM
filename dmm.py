"""dmm
"""
import argparse
import os

import numpy as np
import torch
import torchtext
import pyro

import torch.nn as nn
import pyro.distributions as dist
import pyro.poutine as poutine

from torch.autograd import Variable
from pyro.distributions import TransformedDistribution


class Emitter(nn.Module):
    """
    parameterizes the categorical observation likelihood p(x_t|z_t)
    """

    def __init__(self, input_dim, z_dim, emission_dim):
        super().__init__()
        """
        initilize the fcns used in the network
        """
        self.lin_z_to_hidden = nn.Linear(z_dim, emission_dim)
        self.lin_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_hidden_to_input = nn.Linear(emission_dim, input_dim)
        self.relu = nn.ReLU()

        pass

    def forward(self, z_t):
        """
        given z_t, compute the probabilities that parameterizes the categorical distribution p(x_t|z_t)
        """
        h1 = self.relu(self.lin_z_to_hidden(z_t))
        h2 = self.relu(self.lin_hidden_to_hidden(h1))
        probs = torch.sigmoid(
            self.lin_hidden_to_input(h2)
        )  # might need to change to argmax, max?, softmax?

        return probs


class GatedTransition(nn.Module):
    """
    parameterizes the gaussian latent transition probability p(z_t | z_{t-1})
    """

    def __init__(self, z_dim, transition_dim):
        super().__init__()
        """
        initilize the fcns used in the network
        """
        self.lin_gate_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_proposed_mean_z_to_hidden = nn.Linear(z_dim, transition_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transition_dim, z_dim)
        self.lin_sig = nn.Linear(z_dim, z_dim)
        self.lin_z_to_loc = nn.Linear(z_dim, z_dim)

        self.lin_z_to_loc.weight.data = torch.eye(z_dim)
        self.lin_z_to_loc.bias.data = torch.zeros(z_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

        pass

    def forward(self, z_t_1):
        """
        Given the latent z_{t-1} we return the mean and scale vectors that parameterize the
        (diagonal) gaussian distribution p(z_t | z_{t-1})`
        """
        # compute the gating function
        _gate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = torch.sigmoid(self.lin_gate_hidden_to_z(_gate))
        # compute the 'proposed mean'
        _proposed_mean = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(_proposed_mean)
        # assemble the actual mean used to sample z_t, which mixes a linear transformation
        # of z_{t-1} with the proposed mean modulated by the gating function
        loc = (1 - gate) * self.lin_z_to_loc(z_t_1) + gate * proposed_mean
        # compute the scale used to sample z_t, using the proposed mean from
        # above as input the softplus ensures that scale is positive
        scale = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        # return loc, scale which can be fed into Normal
        return loc, scale


class Combiner(nn.Module):
    """
    parameterizes q(z_t | z_{t-1}, x_{t:T}), which is the basic building block
    of the guide (i.e. the variational distribution). The dependence on x_{t:T} is
    through the hidden state of the RNN
    """

    def __init__(self, z_dim, rnn_dim):
        super().__init__()
        """
        initilize the fcns used in the network
        """
        self.lin_z_to_hidden = nn.Linear(z_dim, rnn_dim)
        self.lin_hidden_to_loc = nn.Linear(rnn_dim, z_dim)
        self.lin_hidden_to_scale = nn.Linear(rnn_dim, z_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

        pass

    def forward(self, z_t_1, h_rnn):
        """
        Given the latent z_{t-1} at at a particular time as well as the hidden
        state of the RNN h(x_{t:T}) we return the mean and scale vectors that
        parameterize the (diagonal) gaussian distribution q(z_t | z_{t-1}, x_{t:T})
        """
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_rnn)
        # use the combined hidden state to compute the mean used to sample z_t
        loc = self.lin_hidden_to_loc(h_combined)
        # use the combined hidden state to compute the scale used to sample z_t
        scale = self.softplus(self.lin_hidden_to_scale(h_combined))
        # return loc, scale which can be fed into Normal
        return loc, scale


class DMM(nn.Module):
    """
    module for the model and the guide (variational distribution) for the DMM
    """

    def __init__(
        self,
        input_dim=52,
        z_dim=100,
        emissions_dim=100,
        transition_dim=200,
        rnn_dim=600,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()
        """
        instantiate modules used in the model and guide
        """
        self.emitter = Emitter(intput_dim, z_dim, emission_dim)
        self.transition = GatedTransition(z_dim, transition_dim)
        self.combiner = Combiner(z_dim, rnn_dim)

        # TODO: alter dropout scheme
        if num_layers == 1:
            rnn_dropout = 0.0
        else:
            rnn_dropout = dropout

        # TODO: add option for bidirectional rnn?
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=rnn_dim,
            nonlinearity="relu",
            batch_first=True,
            bidirectional=False,
            num_layers=num_layers,
            dropout=rnn_drouput,
        )
        """
        define learned parameters that define the probability distributions P(z_1) and q(z_1) and hidden state of rnn
        """
        self.z_0 = nn.Parameter(torch.zeros(z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

        pass

    def model(self, batch, reversed_batch, batch_mask, batch_seqlens, kl_anneal=1.0):
        """
        the model defines p(x_{1:T}|z_{1:T}) and p(z_{1:T})
        """
        # maximum duration of batch
        Tmax = batch.size(1)

        # register torch submodules w/ pyro
        pyro.module("dmm", self)

        # setup recursive conditioning for p(z_t|z_{t-1})
        z_prev = self.z_0.expand(batch.size(0), self.z_0.size(0))

        # sample conditionally indepdent text across the batch
        with pyro.plate("z_batch", len(batch)):
            # sample latent vars z and observed x w/ multiple samples from the guide for each z
            for t in pyro.markov(range(1, Tmax + 1)):

                # compute params of diagonal gaussian p(z_t|z_{t-1})
                z_loc, z_scale = self.trans(z_prev)

                # sample latent variable
                with poutine.scale(scale=kl_anneal):
                    z_t = pyro.sample(
                        "z_%d" % t,
                        dist.Normal(z_loc, z_scale)
                        .mask(batch_mask[:, t - 1 : t])
                        .to_event(1),
                    )

                # compute emission probability from latent variable
                emission_prob = self.emitter(z_t)

                # observe x_t according to the Categorical distribution defined by the emitter probability
                pyro.sample(
                    "obs_x_%d" % t,
                    dist.OneHotCategorical(emission_prob)
                    .mask(batch_mask[:, t - 1 : t])
                    .to_event(1),
                    obs=batch[:, t - 1, :],
                )

                # set conditional var for next time step
                z_prev = z_t
        pass

    def guide(self, batch, reversed_batch, batch_mask, batch_seqlens, kl_anneal=1.0):
        """
        the guide defines the variational distribution q(z_{1:T}|x_x{1:T})
        """
        # maximum duration of batch
        Tmax = batch.size(1)

        # register torch submodules w/ pyro
        pyro.module("dmm", self)

        # to parallelize, we broadcast rnn into continguous gpu memory
        h_0_contig = self.h_0.expand(
            1, batch_size(0), self.rnn.hidden_size
        ).contiguous()

        # push observed sequence through rnn
        rnn_output, _ = self.rnn(batch_reversed, h_0_contig)

        # reverse and unpack rnn output
        rnn_output = pad_reverse(rnn_output, batch_seqlens)

        # setup recursive conditioning
        z_prev = self.z_q_0.expand(batch.size(0), self.z_q_0.size(0))

        with pyro.plate("z_batch", len(mini_batch)):

            for t in pyro.markov(range(1, Tmax + 1)):

                z_loc, z_scale = self.combiner(z_prev, rnn_output[:, t - 1, :])

                z_dist = dist.Normal(z_loc, z_scale)

                assert z_dist.event_shape == ()
                assert z_dist.batch_shape[-2:] == len(batch) == self.z_q_0.size(0)

                # sample z_t from distribution z_dist
                with pyro.poutine.scale(scale=kl_anneal):
                    z_t = pyro.samle(
                        "z_%d" % t, z_dist.mask(batch[:, t - 1 : t]).to_event(1)
                    )

                # set conditional var for next time step
                z_prev = z_t

        pass
