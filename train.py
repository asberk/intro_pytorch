from time import time
from collections import defaultdict
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


def fancy_train_model(
    model, loaders, criterion, optimizer, scheduler, **kwargs
):
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
