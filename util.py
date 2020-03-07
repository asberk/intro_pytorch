"""
util.py

Helpful utilities for the intro_pytorch script

Author: Aaron Berk <aberk@math.ubc.ca>
Copyright © 2020, Aaron Berk, all rights reserved.
Created: 10 February 2020

Commentary:
   Mostly for decluttering the other script
"""
import os
import argparse
import numpy as np
import torch
import torch.optim as optim


def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--width",
        nargs="*",
        default=[],
        type=int,
        help="The width of each of the hidden layers.",
    )
    parser.add_argument(
        "--lr", default=1e-4, type=float, help="The learning rate."
    )
    parser.add_argument(
        "--stepsize",
        default=30,
        type=int,
        help="The step size for tempering the learning rate.",
    )
    parser.add_argument(
        "--gamma", default=0.5, type=float, help="The tempering rate of the LR."
    )
    parser.add_argument(
        "--opt", default="adam", type=str, help="The optimizer to use."
    )
    parser.add_argument(
        "--act",
        default="relu",
        type=str,
        help="The activation function to use.",
    )
    parser.add_argument(
        "--net", default="dense", type=str, help="The network type to use."
    )

    return parser


def get_optimizer_class(optimizer_name, **kwargs):
    """
    get_optimizer_class(optimizer_name)

    Returns a torch optimizer according to the input string
    """
    assert isinstance(
        optimizer_name, str
    ), f"Expected string for optimizer_name but got {optimizer_name}."

    optimizer_name = optimizer_name.lower()

    if optimizer_name == "adam":
        # params, lr, betas, eps, weight_decay, amsgrad
        adam_args = ["lr", "betas", "eps", "weight_decay", "amsgrad"]
        passed_adam_args = {
            key: value for key, value in kwargs if key in adam_args
        }

        def optimizer(parms):
            return optim.Adam(parms, **passed_adam_args)

    elif optimizer_name == "sgd":
        # params, lr, momentum, weight_decay, dampening, nesterov
        sgd_args = ["lr", "momentum", "weight_decay", "dampening", "nesterov"]
        passed_sgd_args = {
            key: value for key, value in kwargs if key in sgd_args
        }

        def optimizer(parms):
            return optim.SGD(parms, **passed_sgd_args)

    elif optimizer_name == "adagrad":
        # params, lr, lr_decay, weight_decay, eps
        adagrad_args = ["lr", "lr_decay", "weight_decay", "eps"]
        passed_adagrad_args = {
            key: value for key, value in kwargs if key in adagrad_args
        }

        def optimizer(parms):
            return optim.Adagrad(parms, **passed_adagrad_args)

    else:
        print(
            "Warning: optimizer_name {optimizer_name} not recognized."
            "\nSoft-failing: returning optim.SGD instead."
        )
        sgd_args = ["lr", "momentum", "weight_decay", "dampening", "nesterov"]
        passed_sgd_args = {
            key: value for key, value in kwargs if key in sgd_args
        }

        def optimizer(parms):
            return optim.SGD(parms, **passed_sgd_args)

    return optimizer


def _parse_batch_size(batch_size):
    if batch_size is None:
        batch_size = (16, 128)
    elif isinstance(batch_size, np.int):
        batch_size = (batch_size, batch_size)
    elif isinstance(batch_size, (tuple, list)):
        assert len(batch_size) == 2, f"expected two entries for batch_size"
        assert all(
            isinstance(x, np.int) for x in batch_size
        ), f"expected ints for batch_size"
    else:
        raise TypeError(f"batch_size {batch_size} not recognized.")
    return batch_size


def _parse_shuffle(shuffle):
    if shuffle is None:
        shuffle = (True, False)
    elif isinstance(shuffle, bool):
        shuffle = (shuffle, shuffle)
    elif isinstance(shuffle, (tuple, list)):
        assert len(shuffle) == 2, f"expected two entries for shuffle"
        assert all(
            isinstance(x, bool) for x in shuffle
        ), f"expected bools for shuffle"
    else:
        raise TypeError(f"shuffle {shuffle} not recognized.")
    return shuffle


def checkpoint(model, filename, parent_dir=None):
    if parent_dir is None:
        parent_dir = "."
    filepath = os.path.join(parent_dir, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Saved model to {filepath}")
    return


def load_model(model, filename, parent_dir=None, map_location=None):
    if parent_dir is None:
        parent_dir = "."
    filepath = os.path.join(parent_dir, filename)
    return model.load_state_dict(
        torch.load(filepath, map_location=map_location)
    )


# # util.py ends here
