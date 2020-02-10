"""
intro_pytorch

This file is an introduction to using PyTorch for training and evaluating neural networks.

Author: Aaron Berk <aberk@math.ubc.ca>
Copyright © 2020, Aaron Berk, all rights reserved.
Created: 10 February 2020

Commentary:
   Run this file from the terminal via:
   > python3 main.py
"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, sampler

from torchvision import datasets, models, transforms

from util import get_parser, get_optimizer_class


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


class SmallNetwork(nn.Module):
    def __init__(self, input_size=None, hidden_size=25, num_classes=None):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, num_classes, bias=True)

    def forward(self, x):
        """
        forward(self, x)

        Dense network with ReLU activation function applied to all but final layer.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x


def show_image(img_or_sample):
    if isinstance(img_or_sample, tuple):
        img = img_or_sample[0]
        label = img_or_sample[1]
    else:
        img = img_or_sample
        label = None
    img = img.cpu().numpy().transpose(1, 2, 0).squeeze()
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(img, cmap="gray_r")
    ax.axis("off")
    if label is not None:
        ax.set_title(f"label: {label}")
    plt.tight_layout()
    plt.show()
    return fig, ax


def train_model(model, loaders, criterion, optimizer, scheduler, **kwargs):
    """
    train_model(model, loaders, **kwargs)

    Inputs
    ------
    model
    loaders
    criterion
    optimizer
    scheduler
    num_epochs
    device

    Returns
    -------
    model
    history
    """
    num_epochs = kwargs.get("num_epochs", 100)
    history = defaultdict(list)
    dataset_sizes = {g: len(dl.dataset) for g, dl in loaders.items()}

    t00 = time()
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            t0 = time()
            if phase == "train":
                scheduler.step()
                model.train()  # set model to training mode
            else:
                model.eval()  # set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data in batches
            for batch in loaders[phase]:
                images, labels = batch

                # zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # track history only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(predictions == labels.data)
                avg_loss = running_loss / dataset_sizes[phase]
                avg_acc = (
                    running_corrects.double().cpu().numpy()
                    / dataset_sizes[phase]
                )
            history[f"{phase}_loss"].append(avg_loss)
            history[f"{phase}_acc"].append(avg_acc)
            t1 = time()
            print(
                f"Epoch {epoch} "
                f"{phase} loss: {avg_loss:.4f} {phase} acc: {avg_acc:.4f} "
                f"({t1 - t0:.1f} sec)"
            )
    t_end = time()
    duration = t_end - t00
    duration_s = duration % 60
    duration_m = duration // 60
    duration_h = int(duration_m // 60)
    duration_m = int(duration_m % 60)
    print(
        f"Training complete in {duration_h}h {duration_m}m "
        f"{duration_s:.1f}s"
    )
    print("Returning FINAL model.")
    return model, history


def main(lr=1e-4, step_size=30, gamma=0.5, hidden_size=100, num_epochs=20):
    loaders = load_small_mnist_data()
    dataset_sizes = {
        g: np.prod(dl.dataset[0][0].shape) for g, dl in loaders.items()
    }
    print(dataset_sizes)

    input_size = dataset_sizes["train"]

    model = SmallNetwork(input_size, hidden_size=hidden_size, num_classes=10)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )

    criterion = nn.CrossEntropyLoss()

    final_model, history = train_model(
        model, loaders, criterion, optimizer, scheduler, num_epochs=num_epochs
    )
    print(history)


if __name__ == "__main__":
    main()


# # main.py ends here
