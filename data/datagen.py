"""datagen
"""

import argparse
import os

import torch
import torchtext
from torchtext.datasets import text_classification

def main():

    parser = argparse.ArgumentParser(
        description="create Tensors for training and testing from a dataset"
    )
    parser.add_argument("--dataset", default="AG_NEWS", help="dataset name")
    parser.add_argument("--ngrams", type=int, default=2, help="ngrams")
    parser.add_argument("--root", default="./data", help="data directory")
    args = parser.parse_args()

    dataset = args.dataset
    
    train_dataset, test_dataset = text_classification.DATASETS[dataset](
        root=args.root, ngrams=args.ngrams
    )
    train_data_path = os.path.join(
        args.root, args.dataset + "_ngrams_{}_train.data".format(args.ngrams)
    )
    test_data_path = os.path.join(
        args.root, args.dataset + "_ngrams_{}_test.data".format(args.ngrams)
    )
    print("saving train data to {}".format(train_data_path))
    torch.save(train_dataset, train_data_path)
    print("saving test data to {}".format(test_data_path))
    torch.save(test_dataset, test_data_path)

if __name__=="__main__":

    main()

    pass
