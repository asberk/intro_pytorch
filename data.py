"""
data

Utilities for managing and loading data and DataLoader objects

Author: Aaron Berk <aberk@math.ubc.ca>
Copyright © 2020, Aaron Berk, all rights reserved.
Created:  6 March 2020
"""
import numpy as np
from sklearn.datasets import make_swiss_roll as _make_swiss_roll
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from util import _parse_batch_size, _parse_shuffle

DATA_DIR = "./data/"


def load_small_mnist_data(
    train_size=None,
    batch_size=None,
    shuffle=True,
    transform=None,
    target_transform=None,
):
    """
    load_mnist_data(train_size=None, train_batch_size=None,
                    val_batch_size=None, test_batch_size=None, shuffle=True,
                    transform=None, target_transform=None)

    Loads MNIST data as train/val/test dataloaders.

    Inputs
    ------
    train_size : int
        An integer between 0 and 60000. I don't know what happens if train_size
        = 0 or 60000; may not work in this case. Default: 50 000
    batch_size : int or tuple
        The size of each batch for train, val, test sets respectively.
        Default: (8, 64, 64)
    shuffle : bool or tuple
        Whether to shuffle dataloaders or not. If tuple, must have length 3.
        Default: True
    transform: torch transform
        The transform to apply to the images. If None, then images are
        normalized and resized to (3, 224, 224).
    target_transform : object
        The transform to apply to the target/labels.

    Returns
    -------
    dl : dict
        With keys 'train', 'val', 'test' each containing the corresponding
        DataLoader instance with MNIST digits.
    """
    if train_size is None:
        train_size = 50000
    assert (
        train_size <= 60000
    ), f"Expected 0 < train_size < 60000 but got {train_size}."
    if not isinstance(train_size, np.int):
        train_size = np.int(train_size)
    val_size = 60000 - train_size
    assert (val_size > 0) and (
        val_size < 60000
    ), f"Expected 0 < train_size < 60000 but got {train_size}."
    if transform is None:
        nmz = transforms.Normalize((0.1307,), (0.3081,))
        toTensor = transforms.ToTensor()  # converts to [0,1] too

        transform = transforms.Compose([toTensor, nmz])
    mnist_data = {
        "train": datasets.MNIST(
            DATA_DIR,
            train=True,
            transform=transform,
            target_transform=target_transform,
            download=True,
        ),
        "test": datasets.MNIST(
            DATA_DIR,
            train=False,
            transform=transform,
            target_transform=target_transform,
            download=True,
        ),
    }

    train_indices = np.random.choice(
        60000, size=train_size, replace=False
    ).astype(int)
    val_indices = np.setdiff1d(list(range(60000)), train_indices).astype(int)
    train_img = (
        mnist_data["train"].data[train_indices].reshape(train_size, -1).float()
    )
    train_lab = mnist_data["train"].targets[train_indices]
    val_img = (
        mnist_data["train"].data[val_indices].reshape(val_size, -1).float()
    )
    val_lab = mnist_data["train"].targets[val_indices]

    if batch_size is None:
        batch_size = (8, 64, 64)
    elif not isinstance(batch_size, int):
        if isinstance(batch_size, tuple) and len(batch_size) == 3:
            assert all(isinstance(x, np.int) for x in batch_size), (
                f"Expected all tuple elements to be of type int but"
                f" got {batch_size}."
            )
        else:
            raise TypeError(
                f"Expected int or 3-tuple for batch_size but got"
                f" {batch_size}."
            )
    else:
        batch_size = (batch_size, batch_size, batch_size)

    if not isinstance(shuffle, bool):
        if isinstance(shuffle, tuple) and len(shuffle) == 3:
            assert all(isinstance(x, bool) for x in shuffle), (
                f"Expected all tuple elements to be of type bool but"
                f" got {shuffle}."
            )
        else:
            raise TypeError(
                f"Expected bool or 3-tuple for shuffle but got {shuffle}."
            )
    else:
        shuffle = (shuffle, shuffle, shuffle)

    train_loader = DataLoader(
        list(zip(train_img, train_lab)),
        batch_size=batch_size[0],
        shuffle=shuffle[0],
    )
    val_loader = DataLoader(
        list(zip(val_img, val_lab)),
        batch_size=batch_size[1],
        shuffle=shuffle[1],
    )

    dl = {
        "train": train_loader,
        "val": val_loader,
        "test": DataLoader(
            mnist_data["test"], batch_size=batch_size[2], shuffle=shuffle[2]
        ),
    }

    return dl


def make_swiss_roll_classification(
    n_samples=10000, noise=1, seed=None, n_segments=6
):
    X, tt = _make_swiss_roll(n_samples, noise=noise, random_state=seed)
    bins = np.linspace(tt.min(), tt.max(), n_segments, endpoint=False)
    tt_dig = np.digitize(tt, bins)
    y = np.where((tt_dig % 2) == 0, 0, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test


def make_dataloaders(
    *arrays, batch_size=None, shuffle=None, make_eval_train=False
):
    batch_size = _parse_batch_size(batch_size)
    shuffle = _parse_shuffle(shuffle)
    X_train, X_test, y_train, y_test = arrays

    dset_train = TensorDataset(
        torch.tensor(X_train).float(), torch.tensor(y_train).long()
    )
    dl_train = DataLoader(
        dset_train, batch_size=batch_size[0], shuffle=shuffle[0]
    )

    if make_eval_train:
        J = np.random.choice(X_train.shape[0], X_test.shape[0], replace=False)
        dset_eval_train = Subset(dset_train, J)
        dl_eval_train = DataLoader(
            dset_eval_train, batch_size=batch_size[1], shuffle=shuffle[1]
        )

    dset_test = TensorDataset(
        torch.tensor(X_test).float(), torch.tensor(y_test).long()
    )
    dl_test = DataLoader(
        dset_test, batch_size=batch_size[1], shuffle=shuffle[1]
    )

    if make_eval_train:
        return dl_train, dl_eval_train, dl_test
    return dl_train, dl_test


# # data.py ends here
