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

        # argument gathering
        self.rand_seed = args.rand_seed
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.optim = args.optim
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.data_dir = args.data_dir
        self.output_dir = args.output
        self.embed = args.embed
        self.embed_dim = args.embed_dim
        self.hidden_dim = args.hidden_dim
        self.ckpt_dir = args.ckpt_dir
        self.ckpt_name = args.ckpt

        self.global_iter = 0

        pass

    def train(self):
        """
        trains a network
        """

        return 0
    
    def load_ckpt(self, filename):
        """
        load checkpoint
        """

        file_path = os.path.join(self.ckpt_dir, filename)

        if os.path.isfile(file_path):

            checkpoint = torch.load(file_path)

            self.global_iter = checkpoint["iter"]
            self.net.load_state_dict(checkpoint["model_states"]["net"])
            self.optim.load_state_dict(checkpoint["optim_states"]["optim"])

            print("=> loaded ckpt '{} (iter {})'".format(file_path, self.global_iter))

        else:
            print("=> no ckpt at '{}'".format(file_path))
        
        pass

   def save_ckpt(self, filename):
        """
        save a model checkpoint
        saves best checkpoint always
        """

        model_states = {"net": self.net.state_dict()}
        optim_states = {"optim": self.optim.state_dict()}

        states = {
            "iter": self.global_iter,
            "model_states": model_states,
            "optim_states": optim_states,
        }

        file_path = os.path.join(self.ckpt_dir, filename)

        with open(file_path, mode="wb+") as f:
            torch.save(states, f)

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
    parser.add_argument("--optim", default="ADAM", type=int, help="optimizer")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--beta1", default=0.9, type=float, help="ADAM beta1")
    parser.add_argument("--beta2", default=0.999, type=float, help="ADAM beta2")
    parser.add_argument("--data_dir", default="data", type=str, help="data directory")
    parser.add_argument("--output_dir", default="outputs", type=str, help="output dir")
    parser.add_argument("--embed", default="typ", type=int, help="type of embedding")
    parser.add_argument("--embed_dim", default="128", type=int, help="embedding dim.")
    parser.add_argument("--hidden_dim", default="32", type=int, help="lstm hidden dim.")
    parser.add_argument("--load_cktp", default=FALSE, type=bool, help="load ckpt or nah")
    parser.add_argument("--ckpt_name", default="last", type=str, help="checkpoint name")
    parser.add_argument(
        "--ckpt_dir",
        default="outputs/saved_models/ckpts/",
        type=str,
        help="where to save ckpts",
    )
    # maybe add lr scheduler? 


    args = parser.parse_args()

    main(args)

    pass
