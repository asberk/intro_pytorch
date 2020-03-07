"""
wavelet_example

Machine learning example using wavelet transformed MNIST digits.

Author: Aaron Berk <aberk@math.ubc.ca>
Copyright © 2020, Aaron Berk, all rights reserved.
Created:  6 March 2020
"""
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST

from data import DATA_DIR
from nnet import SimpleNet
from train import train_setup, create_train_fn, create_eval_fn
from util import _parse_batch_size, _parse_shuffle
from wvlt import WaveletTransformer

mnist_train = MNIST(DATA_DIR, download=False)
mnist_test = MNIST(DATA_DIR, download=False)


def get_wcoefs(tens_array, wname="db1", dim=2):
    ww = WaveletTransformer(wname, dim=dim)
    cx0, *Tx0 = ww.decompose(tens_array[0].numpy())
    wcoef_array = np.row_stack([ww.decompose(x.numpy())[0] for x in tens_array])
    return (wcoef_array, *Tx0)


def make_mapped_wavelet_dataloaders(
    X_train,
    y_train,
    X_test,
    y_test,
    m=224,
    wname="db1",
    dim=2,
    batch_size=None,
    shuffle=None,
):
    wcoef_train, *Twc_train = get_wcoefs(X_train, wname, dim)
    wcoef_test, *Twc_test = get_wcoefs(X_test, wname, dim)
    N = wcoef_train.shape[1]
    A = np.random.randn(m, N) / m ** 0.5
    Acoef_train = wcoef_train.dot(A.T)
    Acoef_test = wcoef_test.dot(A.T)
    dset_train = TensorDataset(
        torch.tensor(Acoef_train).float(), torch.tensor(y_train).long()
    )
    dset_test = TensorDataset(
        torch.tensor(Acoef_test).float(), torch.tensor(y_test).long()
    )

    batch_size = _parse_batch_size(batch_size)
    shuffle = _parse_shuffle(shuffle)
    dl_train = DataLoader(
        dset_train, batch_size=batch_size[0], shuffle=shuffle[0]
    )
    dl_test = DataLoader(
        dset_test, batch_size=batch_size[1], shuffle=shuffle[1]
    )
    return dl_train, dl_test


if __name__ == "__main__":
    m = 50
    dl_train, dl_test = make_mapped_wavelet_dataloaders(
        mnist_train.data,
        mnist_train.targets,
        mnist_test.data,
        mnist_test.targets,
        m=m,
    )

    num_classes = np.unique(mnist_train.targets).size
    model = SimpleNet(m, num_classes)
    criterion, optimizer, lr_scheduler = train_setup(model, lr=1e-5)
    train_fn = create_train_fn(model, criterion, optimizer)
    eval_fn = create_eval_fn(model, criterion)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for epoch in range(100):
        for batch in dl_train:
            tmp_train_loss = []
            tmp_train_acc = []
            l, a = train_fn(batch)
            tmp_train_loss.append(l)
            tmp_train_acc.append(a)

        for batch in dl_test:
            tmp_test_loss = []
            tmp_test_acc = []
            l, a = eval_fn(batch)
            tmp_test_loss.append(l)
            tmp_test_acc.append(a)
        train_loss.append(np.mean(tmp_train_loss))
        train_acc.append(np.mean(tmp_train_acc))
        test_loss.append(np.mean(tmp_test_loss))
        test_acc.append(np.mean(tmp_test_acc))

        print("epoch", epoch)
        print("train loss:", train_loss[-1], "acc:", train_acc[-1])
        print(" test loss:", test_loss[-1], "acc:", test_acc[-1])

# # wavelet_example.py ends here
