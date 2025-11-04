import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Union, Dict
from .metrics import calculate_classification_metrics
from config import AUGMENTATION_CONFIG
import numpy as np


class Mixup:
    """Mixup augmentation for label smoothing."""

    def __init__(self, alpha: float = 0.8):
        self.alpha = alpha

    def __call__(self, x, y):
        if self.alpha > 0 and AUGMENTATION_CONFIG["use_augmentation"]:
            lam = np.random.beta(self.alpha, self.alpha)
            index = torch.randperm(x.size(0))
            x_mix = lam * x + (1 - lam) * x[index]
            y_a, y_b = y, y[index]
            return x_mix, y_a, y_b, lam
        return x, y, y, 1.0  # No mixup if disabled


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixup: Optional[Mixup] = None,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: The neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on

    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        if mixup:
            inputs, labels_a, labels_b, lam = mixup(inputs, labels)

        optimizer.zero_grad()
        outputs = model(inputs)

        if mixup:
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(
                outputs, labels_b
            )
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 10,
    detailed_metrics: bool = True,
) -> Union[Tuple[float, float], Tuple[float, float, Dict]]:
    """
    Evaluate the model on data.

    Args:
        model: The neural network model
        dataloader: Data loader
        criterion: Loss function
        device: Device to run on
        num_classes: Number of classes
        detailed_metrics: Whether to compute detailed metrics

    Returns:
        If detailed_metrics=False: (loss, accuracy)
        If detailed_metrics=True: (loss, accuracy, metrics_dict)
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(dataloader)
    accuracy = 100.0 * correct / total

    if detailed_metrics:
        metrics = calculate_classification_metrics(
            model, dataloader, device, num_classes=num_classes
        )
        return test_loss, accuracy, metrics
    else:
        return test_loss, accuracy
