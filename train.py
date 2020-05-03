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

    def validate(self, val_iter):
        """
        freezes training and validates on the network with a validation set
        """
        # freeze training
        self.dmm.rnn.eval()
        val_nll = 0
        for ii, batch in enumerate(iter(val_iter)):

            batch_data = Variable(batch.text[0].to(self.device))
            seqlens = Variable(batch.text[1].to(self.device))

            # transpose to [B, seqlen, vocab_size] shape
            batch_data = torch.t(batch_data)
            # compute one hot character embedding
            batch = nn.functional.one_hot(batch_data, self.vocab_size).float()
            # flip sequence for rnn
            batch_reversed = utils.reverse_seq(batch, seqlens)
            batch_reversed = nn.utils.rnn.pack_padded_sequence(
                batch_reversed, seqlens, batch_first=True
            )
            # compute temporal mask
            batch_mask = utils.get_batch_mask(batch, seqlens).cuda()
            # perform evaluation
            val_nll += self.svi.evaluate_loss(
                batch, batch_reversed, batch_mask, seqlens
            )

        # resume training
        self.dmm.rnn.train()

        # report loss TODO: normalize
        loss = val_loss

        return loss

    def train_batch(self, train_iter, epoch):
        """
        process a batch (single epoch)
        """
        batch_loss = 0
        loss = 0
        for ii, batch in enumerate(iter(train_iter)):

            batch_data = Variable(batch.text[0].to(self.device))
            seqlens = Variable(batch.text[1].to(self.device))

            # transpose to [B, seqlen, vocab_size] shape
            batch_data = torch.t(batch_data)
            # compute one hot character embedding
            batch = nn.functional.one_hot(batch_data, self.vocab_size).float()
            # flip sequence for rnn
            batch_reversed = utils.reverse_seq(batch, seqlens)
            batch_reversed = nn.utils.rnn.pack_padded_sequence(
                batch_reversed, seqlens, batch_first=True
            )
            # compute temporal mask
            batch_mask = utils.get_batch_mask(batch, seqlens).cuda()

            # compute kl-div annealing factor
            if self.kl_ae > 0 and epoch < self.kl_ae:
                min_af = self.maf
                kl_anneal = min_af + (1 - min_af) * (
                    float(ii + epoch * self.N_batches + 1)
                    / float(args.kl_ae * self.N_batches)
                )
            else:
                # default kl-div annealing factor is unity
                kl_anneal = 1.0

            # take gradient step
            batch_loss = self.svi.step(
                batch_data, batch_reversed, batch_mask, seqlens, kl_anneal
            )
            loss += batch_loss / (torch.sum(seqlens).float())

        return loss

    def train(self):
        """
        trains a network with a given training set
        """
        # TODO: add device cli arg
        self.device = torch.device("cuda")
        np.random.seed(self.rand_seed)
        torch.manual_seed(self.rand_seed)

        # setup logging
        log = get_logger(self.log)

        # load dataset TODO: make data fpath cli arg
        (train, val, test), vocab = datahandler.load_data("./data/ptb")

        self.vocab_size = len(vocab)

        # make iterable dataset object
        train_iter, val_iter, test_iter = torchtext.data.BucketIterator.splits(
            (train, val, test),
            batch_sizes=[args.batch_Size, 1, 1],
            device=device,
            repeat=False,
            sort_key=lambda x: len(x.text),
            sort_within_batch=True,
        )

        # instantiate the dmm
        self.dmm = DMM(input_dim=self.vocab_size, dropout=self.dropout)

        # setup optimizer
        opt_params = {
            "lr": self.lr,
            "betas": (self.beta1, self.beta2),
            "clip_norm": self.cn,
            "lrd": self.lr_decay,
            "weight_decay": self.wd,
        }
        self.adam = ClippedAdam(opt_params)
        # set up inference algorithm
        self.elbo = Trace_ELBO()
        self.svi = SVI(dmm.model, dmm.guide, loss=elbo)

        # TODO: compute number of minibatches

        val_f = 10

        print("training dmm")
        times = [time.time()]
        for epoch in range(self.n_epochs):

            if self.ckpt_f > 0 and epoch > 0 and epoch % self.ckpt_f == 0:
                save_ckpt()

            # train and report metrics
            train_nll = train_batch(train_iter, epoch,)

            times.append(time.time())
            t_elps = times[-1] - times[-2]
            log(
                "epoch %04d -> train nll: %.4f \t t_elps=%.3f sec"
                % (epoch, train_nll, t_elps)
            )

            if epoch % val_f == 0:
                val_nll = validate(val_iter)
        pass

    def save_ckpt(self):
        """
        saves the state of the network and optimizer for later
        """
        log("saving model to %s" % self.save_model)
        torch.save(self.dmm.state_dict(), args.save_model)
        log("saving optimizer states to %s" % args.save_opt)
        self.adam.save(self.save_opt)

        pass

    def load_ckpt(self):
        """
        loads a saved checkpoint
        """
        assert exists(args.load_opt) and exists(
            args.load_model
        ), "--load-model and/or --load-opt misspecified"
        log("loading model from %s..." % self.load_model)
        self.dmm.load_state_dict(torch.load(self.load_model))
        log("loading optimizer states from %s..." % self.load_opt)
        self.adam.load(self.load_opt)

        pass
