import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR


def train_setup(
    model, lr=1e-3, momentum=0.9, weight_decay=0.001, nesterov=True, gamma=0.975
):
    """
    train_setup(
        model, lr=1e-3, momentum=0.9, weight_decay=0.001, nesterov=True, gamma=0.975
    )

    Inputs
    ------
    model : nn.Module
    lr : float
        Default: 1e-3
    momentum : float
        Default: 0.9
    weight_decay : float
        Default: 0.001
    nesterov : bool
        Default: True
    gamma : float
        Default: 0.975

    Outputs
    -------
    criterion : torch.nn.CrossEntropyLoss
        Cross-entropy loss
    optimizer : torch.optim.SGD
        Stochastic Gradient Descent implementation with bells and whistles
    scheduler : lr_scheduler.ExponentialLR
        Learning rate annealer
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    return criterion, optimizer, scheduler


def create_train_fn(model, criterion, optimizer):
    def train_fn(batch):
        model.train()
        x, y = batch
        y_prob = model(x)
        loss = criterion(y_prob, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred = torch.max(y_prob, 1).indices
        loss_value = loss.item()
        accuracy_value = (y_pred == y.detach()).float().mean()
        return loss_value, accuracy_value

    return train_fn


def create_eval_fn(model, criterion):
    def eval_fn(batch):
        model.eval()
        x, y = batch
        y_prob = model(x)
        loss = criterion(y_prob, y)
        y_pred = torch.max(y_prob, 1).indices
        loss_value = loss.item()
        accuracy_value = (y_pred == y.detach()).float().mean()
        return loss_value, accuracy_value

    return eval_fn
