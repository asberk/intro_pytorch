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


class RepeatAlongAxis(object):
    """
    RepeatAlongAxis takes a "grayscale" tensor and returns an rgb tensor. For
    example, both (28, 28) and (1, 28, 28) are returned as a (3, 28, 28) tensor
    whose original contents have been tiled along the first axis.

    This is just a wrapper around torch.Tensor.repeat(3, 1, 1).
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.repeat(3, 1, 1)


class EvenOddTransform(object):
    """
    EvenOddTransform is used for the MNIST sandbox: takes target (i.e., the
    default labels in {0, 1, 2, ..., 9}) and transforms them to even/odd (i.e.,
    odd=1, even=0). Can either return (target, even_odd) or just even_odd.
    """

    def __init__(self, return_both=False):
        """
        EvenOddTransform(return_both=False)
        """
        self.return_both = return_both

    def __call__(self, target):
        eo = 0 if (target % 2 == 0) else 1
        if self.return_both:
            return [target, eo]
        else:
            return eo


class ChunkSampler(sampler.Sampler):
    """
    ChunkSampler samples elements sequentially from some offset.

    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """

    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


def load_big_mnist_data(
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
        resizer224 = transforms.Resize(224)
        centerCrop = transforms.CenterCrop(224)
        repeater = RepeatAlongAxis()

        transform = transforms.Compose(
            [resizer224, centerCrop, toTensor, nmz, repeater]
        )
    mnist_data = {
        "train": datasets.MNIST(
            DATA_DIR,
            train=True,
            transform=transform,
            target_transform=target_transform,
            download=True,
        ),
        "val": datasets.MNIST(
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
    dl = {
        "train": DataLoader(
            mnist_data["train"],
            batch_size=batch_size[0],
            shuffle=shuffle[0],
            sampler=ChunkSampler(train_size, 0),
        ),
        "val": DataLoader(
            mnist_data["val"],
            batch_size=batch_size[1],
            shuffle=shuffle[1],
            sampler=ChunkSampler(val_size, train_size),
        ),
        "test": DataLoader(
            mnist_data["test"], batch_size=batch_size[2], shuffle=shuffle[2]
        ),
    }

    return dl


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
    train_img = mnist_data["train"].data[train_indices]
    train_lab = mnist_data["train"].targets[train_indices]
    val_img = mnist_data["train"].data[val_indices]
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


class Network(nn.Module):
    def __init__(
        self, input_size=None, num_classes=None, widths=None, activation="relu"
    ):
        super().__init__()
        if widths is None:
            self.widths = (25, 25)
        else:
            assert isinstance(
                widths, (tuple, list)
            ), f"require tuple or int for widths but got {widths}."
            self.widths = tuple([input_size, *widths, num_classes])
        self._set_activation(activation)
        self.layers = [
            nn.Linear(i, o) for i, o in zip(self.widths[:-1], self.widths[1:])
        ]
        for i, layer in enumerate(self.layers):
            for j, parm in enumerate(layer.parameters()):
                self.register_parameter(f"{i}_{j}", parm)

    def forward(self, x):
        """
        forward(self, x)

        Dense network with ReLU activation function applied to all but final layer.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x.float())
            if (i + 1) < len(self.layers):
                x = self._act(x)
        return x

    def _set_activation(self, activation):
        activation = activation.lower()
        assert activation in ["relu", "tanh", "sigmoid", "softmax"], (
            f"expected activation to be one of 'relu', 'tanh', 'sigmoid', "
            f"'softmax' but got {activation}."
        )
        if activation == "relu":
            self._act = torch.relu
        elif activation == "tanh":
            self._act = torch.tanh
        elif activation == "sigmoid":
            self._act = torch.sigmoid
        elif activation == "softmax":
            self._act = torch.softmax
        else:
            raise TypeError(f"Activation function {activation} not recognized.")
        self.activation = activation
        return


def load_network(network_name=None, **kwargs):
    """
    load_network(network_name=None, **kwargs)

    Inputs
    ------
    network_name : str
    pretrained : bool
    num_classes : int
    widths : tuple or list
    input_size : tuple or int
    activation : str
    device : torch.device

    Returns
    -------
    model : nn.Module
    """
    pretrained = kwargs.get("pretrained", True)
    num_classes = kwargs.get("num_classes", 10)
    widths = kwargs.get("widths", (25, 25))
    input_size = kwargs.get("input_size", 28 * 28)
    activation = kwargs.get("activation", "relu")
    device = kwargs.get("device", None)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    network_name = network_name.lower()
    if network_name == "resnet":
        model = models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif network_name == "vgg":
        model = models.vgg16(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif network_name == "inception":
        model = models.inception_v3(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    else:
        model = Network(input_size, num_classes, widths, activation)

    return model


def get_optimizer(optim_name, model_or_parameters, **kwargs):
    """
    get_optimizer(
        optim_name,
        model_or_parameters,
        lr,
        momentum ,
        betas,
        eps,
        weight_decay ,
        amsgrad,
        dampening,
        nesterov ,
        step_size,
        gamma,
    )

    Inputs
    ------
    optim_name : str
        Must be one of ['adam', 'sgd', 'adagrad']
    model_or_parameters : nn.Module or dict
        The model or model parameters to optimize. For instance,
        pass model or model.parameters()
    lr : float
        The learning rate to use for the optimizer. Default: 1e-4
    momentum  : float
        The momentum to use if optim_class is SGD
    betas : tuple
        Used if optim_class is Adam. See optim.Adam documentation for more details.
    eps : float
        Used for Adam and Adagrad
    weight_decay  : float
        L2 penalty for weights.
    amsgrad : bool
        Used for Adam only.
    dampening : float
        Used for SGD only.
    nesterov  : bool
        Used for SGD only.
    step_size : int
        For StepLR learning rate scheduler.
    gamma : float
        For StepLR learning rate scheduler.

    """
    if isinstance(model_or_parameters, nn.Module):
        model = model_or_parameters
        parameters = model.parameters()
    elif isinstance(model_or_parameters, (dict, list)):
        parameters = model_or_parameters
    else:
        # Just try it anyway in case it's a generator.
        parameters = model_or_parameters

    step_size = kwargs.pop("step_size", 30)
    gamma = kwargs.pop("gamma", 0.5)
    optimizer_class = get_optimizer_class(optim_name, **kwargs)
    optimizer = optimizer_class(parameters)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    return optimizer, scheduler


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


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args(["--lr", "1e-4"])
    lr = args.lr
    widths = args.width
    activation = args.act
    network_name = args.net
    optim_name = args.opt
    step_size = args.stepsize
    gamma = args.gamma

    if network_name in ["resnet", "vgg", "inception"]:
        input_size = (3, 224, 224)
        loaders = load_big_mnist_data(shuffle=False)
    else:
        input_size = 28 * 28
        loaders = load_small_mnist_data()

    net = load_network(network_name)
    optimizer, scheduler = get_optimizer(
        optim_name, net.parameters(), lr=lr, step_size=step_size, gamma=gamma
    )

    criterion = nn.CrossEntropyLoss()

    final_model, history = train_model(
        net, loaders, criterion, optimizer, scheduler, num_epochs=3
    )
    print(history)


# # main.py ends here
