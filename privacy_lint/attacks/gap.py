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
def default_compute_accuracies(model: nn.Module, dataloader: DataLoader):
    """
    Computes 0-1 accuracy of the model for each sample in the dataloader.
    """

    accuracies = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    for inp, target in tqdm(dataloader):
        inp = inp.to(device)
        target = target.to(device)
        outputs = model(inp)
        accuracies += (outputs.argmax(dim=1) == target).tolist()

    return torch.Tensor(accuracies)


class GapAttack:
    """
    Given a function to compute the accuracies:
        - Computes the accuracies of the private model on both the private
          train and heldout sets
        - Returns an AttackResults object to analyze the results
    """

    def __init__(
        self,
        compute_accuracies: Callable[
            [nn.Module, DataLoader], torch.Tensor
        ] = default_compute_accuracies,
    ):
        self.compute_accuracies = compute_accuracies

    def launch(
        self,
        private_model: nn.Module,
        private_train: DataLoader,
        private_heldout: DataLoader,
    ):
        accuracies_train = self.compute_accuracies(private_model, private_train)
        accuracies_heldout = self.compute_accuracies(private_model, private_heldout)

        return AttackResults(accuracies_train, accuracies_heldout)
