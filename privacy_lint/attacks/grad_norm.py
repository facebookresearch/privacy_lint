from functools import partial
from typing import Callable

import torch
import torch.nn as nn

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from privacy_lint.attack_results import AttackResults
from torch.utils.data import DataLoader
from tqdm import tqdm


def _compute_grad_norm(
    model: nn.Module, dataloader: DataLoader, criterion: torch.nn.modules.loss._Loss
):
    """
    Computes the per-sample gradient norms given by the model over the dataloader.
    """
    norms = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    for inputs, targets in tqdm(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        for i in range(len(inputs)):
            model.zero_grad()
            outputs = model(inputs[i : i + 1])
            loss = criterion(outputs, targets[i : i + 1])
            loss.backward()

            norms.append(
                sum([torch.sum(torch.pow(p.grad, 2)).cpu() for p in model.parameters()])
            )

    return torch.Tensor(norms)


compute_grad_norm_cross_entropy = partial(
    _compute_grad_norm, criterion=nn.CrossEntropyLoss()
)
compute_grad_norm_mse = partial(
    _compute_grad_norm, criterion=nn.MSELoss(reduction="sum")
)


class GradNormAttack:
    """
    Given a function to compute the gradient norms:
        - Computes the gradient norms of the private model on both the private
          train and heldout sets
        - Returns an AttackResults object to analyze the results
    """

    def __init__(
        self,
        compute_grad_norm: Callable[
            [nn.Module, DataLoader], torch.Tensor
        ] = compute_grad_norm_cross_entropy,
    ):
        self.compute_grad_norm = compute_grad_norm

    def launch(
        self,
        private_model: nn.Module,
        private_train: DataLoader,
        private_heldout: DataLoader,
    ):
        grad_norm_train = self.compute_grad_norm(private_model, private_train)
        grad_norm_heldout = self.compute_grad_norm(private_model, private_heldout)

        return AttackResults(-grad_norm_train, -grad_norm_heldout)
