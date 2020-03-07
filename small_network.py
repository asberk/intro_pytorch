"""
small_network.py

This file is an introduction to using PyTorch for training and evaluating neural networks.

Author: Aaron Berk <aberk@math.ubc.ca>
Copyright © 2020, Aaron Berk, all rights reserved.
Created: 10 February 2020

Commentary:
   Run this file from the terminal via:
   > python3 main.py
"""
import os
from datetime import datetime
import numpy as np

from data import DATA_DIR, load_small_mnist_data
from nnet import SmallNetwork
from train import fancy_train_model, train_setup
from util import checkpoint


def main(lr=1e-4, step_size=30, gamma=0.5, hidden_size=100, num_epochs=20):
    loaders = load_small_mnist_data()
    dataset_sizes = {
        g: np.prod(dl.dataset[0][0].shape) for g, dl in loaders.items()
    }
    print(dataset_sizes)

    input_size = dataset_sizes["train"]

    model = SmallNetwork(input_size, hidden_size=hidden_size, num_classes=10)
    criterion, optimizer, scheduler = train_setup(model, lr=lr)

    final_model, history = fancy_train_model(
        model, loaders, criterion, optimizer, scheduler, num_epochs=num_epochs
    )
    print(history)
    return final_model


if __name__ == "__main__":
    model = main()

    subdir = "small_network_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    parent_dir = os.path.join(DATA_DIR, subdir)
    checkpoint(model, "my_model.pth", parent_dir=parent_dir)

# # main.py ends here
