from functools import partial
from typing import Callable

import torch
import torch.nn as nn
from privacy_lint.attack_results import AttackResults
from torch.utils.data import DataLoader
from tqdm import tqdm


def _compute_loss(
    model: nn.Module, dataloader: DataLoader, criterion: torch.nn.modules.loss._Loss
) -> torch.Tensor:
    losses = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    with torch.no_grad():
        for img, target in tqdm(dataloader):
            img = img.to(device)
            target = target.to(device)
            outputs = model(img)
            batch_losses = criterion(outputs, target)
            batch_losses = torch.einsum("i...->i", batch_losses)
            losses += batch_losses.tolist()

    return torch.Tensor(losses)


compute_loss_cross_entropy = partial(
    _compute_loss, criterion=nn.CrossEntropyLoss(reduction="none")
)
compute_loss_mse = partial(_compute_loss, criterion=nn.MSELoss(reduction="none"))


class LossAttack:
    def __init__(
        self,
        compute_loss: Callable[
            [nn.Module, DataLoader], torch.Tensor
        ] = compute_loss_cross_entropy,
    ):
        self.compute_loss = compute_loss

    def launch(
        self,
        private_model: nn.Module,
        private_train: DataLoader,
        private_heldout: DataLoader,
    ):
        losses_train = self.compute_loss(private_model, private_train)
        losses_heldout = self.compute_loss(private_model, private_heldout)

        return AttackResults(-losses_train, -losses_heldout)
