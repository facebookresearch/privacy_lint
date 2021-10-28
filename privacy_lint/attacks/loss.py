# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable

import torch
import torch.nn as nn
from privacy_lint.attack_results import AttackResults
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def compute_loss_cross_entropy(
    model: nn.Module, dataloader: DataLoader
) -> torch.Tensor:
    """
    Computes the losses given by the model over the dataloader.
    """

    losses = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.to(device)

    for img, target in tqdm(dataloader):
        img = img.to(device)
        target = target.to(device)
        outputs = model(img)
        batch_losses = criterion(outputs, target)
        losses += batch_losses.tolist()

    return torch.Tensor(losses)


class LossAttack:
    """
    Given a function to compute the loss:
        - Computes the losses of the private model on both the private
          train and heldout sets
        - Returns an AttackResults object to analyze the results
    """

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
