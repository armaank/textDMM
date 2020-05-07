"""utils
todo: add comments
"""
import logging

import torch

from torch import nn


def get_logger(log_file):
    logging.basicConfig(
        level=logging.DEBUG, format="%(message)s", filename=log_file, filemode="w"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)

    def log(s):
        logging.info(s)

    return log


def reverse_seq(batch, seqlens):
    """
    reverses a batch temporally
    """
    reversed_batch = torch.zeros_like(batch)

    for ii in range(batch.size(0)):

        T = seqlens[ii]
        t_slice = torch.arange(T - 1, -1, -1, device=batch.device)  # TODO: add device?
        reversed_seq = torch.index_select(batch[ii, :, :], 0, t_slice)
        reversed_batch[ii, 0:T, :] = reversed_seq

    return reversed_batch


def pad_and_reverse(rnn_output, seqlens):
    """
    unpacks hidden state as output of a torch rnn and reverses each seq
    """

    rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    reversed_output = reverse_seq(rnn_output, seqlens)

    return reversed_output


def generate_batch_mask(batch, seqlens):
    """
    makes a temporal mask
    """

    mask = torch.zeros(batch.shape[0:2])
    for ii in range(batch.shape[0]):
        mask[ii, 0 : seqlens[ii]] = torch.ones(seqlens[ii])

    return mask
