"""train
"""
import os
import random

import numpy
import torch
import pyro

from pyro.infer import SVI, Trace_ELBO

import util
import dataloader
from dmm import DMM


class Trainer(object):
    """
    trainer class used to instantiate, train and validate a network
    """

    def __init__(self, args):

        # argument gathering
        self.seed = args.rand_seed
        self.dev_num = args.dev_num
        self.cuda = args.cuda
        self.n_epoch = args.n_epoch
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.wd = args.wd
        self.cn = args.cn
        self.lr_decay = args.lr_decay
        self.ae = args.kl_ae
        self.maf = args.maf
        self.dropout = args.dropout
        self.ckpt_f = args.ckpt_f
        self.load_opt = args.load_opt
        self.load_model = args.load_model
        self.save_opt = args.save_opt
        self.save_model = args.save_model
        self.log = args.log

    def validate(self):
        """
        freezes training and validates on the network with a validation set
        """

        return loss

    def train(self):
        """
        trains a network with a given training set
        """
        # TODO: add device
        np.random.seed(self.rand_seed)
        torch.manual_seed(self.rand_seed)

        # setup logging
        log = get_logger(self.log)

        # load dataset

        # instantiate the dmm
        dmm = DMM(input_dim=V, dropout=self.dropout)

        # setup optimizer
        opt_params = {
            "lr": self.lr,
            "betas": (self.beta1, self.beta2),
            "clip_norm": self.cn,
            "lrd": self.lr_decay,
            "weight_decay": self.wd,
        }
        adam = ClippedAdam(opt_params)
        # set up inference algorithm
        elbo = Trace_ELBO()
        svi = SVI(dmm.model, dmm.guide, loss=elbo)

        pass

    def save_ckpt(self):
        """
        saves the state of the network and optimizer for later
        """

        pass

    def load_ckpt(self):
        """
        loads a saved checkpoint
        """

        pass
