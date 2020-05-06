"""main
"""
import argparse
import os

import numpy as np
import torch

from train import Trainer

# TODO: make set_device a cli
torch.cuda.set_device(0)


def main(args):
    """main
    calls Trainer to train and validate a character-level language model

    inputs are the passed in cli args
    """

    dmm = Trainer(args)
    dmm.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="lm")

    parser.add_argument("--rand_seed", default=1, type=int, help="random seed")
    parser.add_argument("--dev_num", default=2, type=int, help="gpu device number")
    parser.add_argument("--cuda", default=True, type=bool, help="enable cuda")
    parser.add_argument("--n_epoch", default=5000, type=float, help="num of epochs")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size")
    parser.add_argument("--lr", default=1e-4, type=float, help="ADAM learning rate")
    parser.add_argument("--beta1", default=0.9, type=float, help="ADAM beta1")
    parser.add_argument("--beta2", default=0.999, type=float, help="ADAM beta2")
    parser.add_argument("--wd", default=2.0, type=float, help="weight decay")
    parser.add_argument("--cn", default=0.001, type=float, help="grad. clipping")
    parser.add_argument(
        "--lr_decay", default=0.99996, type=float, help="learning rate decay"
    )
    parser.add_argument("--kl_ae", default=1000, type=float, help="kl annealing epochs")
    parser.add_argument(
        "--maf", default=0.2, type=float, help="minimum annealing factor"
    )
    parser.add_argument("--dropout", default=0.5, type=float, help="rnn dropout rate")
    parser.add_argument("--ckpt_f", default=10, type=int, help="ckpt frequency")
    parser.add_argument(
        "--load_opt", default="", type=str, help="fpath to saved optimizer"
    )
    parser.add_argument(
        "--load_model", default="", type=str, help="fpath to saved model"
    )
    parser.add_argument(
        "--save_opt", default="./save_opt.pt", type=str, help="fpath to save optimizer"
    )
    parser.add_argument(
        "--save_model", default="./save_model.pt", type=str, help="fpath to save model"
    )
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--tmc", action="store_true")
    parser.add_argument("--tmcelbo", action="store_true")
    parser.add_argument(
        "--tmc-num-samples", default=10, type=int, help="samples for tensor monte carlo"
    )
    parser.add_argument("--log", default="log.log", type=str, help="fpath to log")

    args = parser.parse_args()

    main(args)
