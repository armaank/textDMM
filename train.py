"""train
"""
import argparse
import os


class Trainer(object):
    """
    Trainer

    Trainer class used to instantiate and train a network

    Properties
    ----------
    """

    def __init__(self, args):

        pass

    def train(self):
        """
        trains a network

        todo: train/validation split
        """

        return 0

    def load_ckpt(self):
        """
        loads a checkpoint

        """

        pass

    def save_ckpt(self):
        """ 
        saves a model checkpoint
        """

        pass


def main(args):
    """
    main

    calls Trainer to train a network
    """

    net = Trainer(args)

    net.train()

    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train a text categorizer")

    parser.add_argument("--rand_seed", default=1, type=int, help="random seed")
    parser.add_argument("--cuda", default=True, type=bool, help="enable cuda")
    parser.add_argument("--n_iter", default=1e6, type=float, help="num of grad. steps")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--lr", default=1e-4, type=float, help="ADAM learning rate")
    parser.add_argument("--beta1", default=0.9, type=float, help="ADAM beta1")
    parser.add_argument("--beta2", default=0.999, type=float, help="ADAM beta2")
    parser.add_argument("--data_dir", default="data", type=str, help="data directory")
    parser.add_argument("--output_dir", default="outputs", type=str, help="output dir")
    parser.add_argument("--n_workers", default=2, type=int, help="dataloader n_workers")
    parser.add_argument("--embed", default="typ", type=int, help="type of embedding")
    parser.add_argument("--embed_dim", default="128", type=int, help="embedding dim.")
    parser.add_argument("--hidden_dim", default="32", type=int, help="lstm hidden dim.")

    parser.add_argument(
        "--ckpt_dir",
        default="outputs/saved_models/ckpts/experiment",
        type=str,
        help="checkpoint dir",
    )
    # modify loading of ckpts
    parser.add_argument(
        "--save_step",
        default=100,
        type=int,
        help="num of grad steps before checkpoints are saved",
    )

    parser.add_argument(
        "--ckpt_name",
        default="last",
        type=str,
        help="load previous checkpoint. insert checkpoint filename",
    )

    # add lr decay, sgd lr

    args = parser.parse_args()

    main(args)

    pass
