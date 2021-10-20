# from collections.abc import Callable
from typing import Callable

import torch
import torch.nn as nn
from privacy_lint.attack_results import AttackResults
from torch.utils.data import DataLoader
from tqdm import tqdm


def default_compute_accuracies(model: nn.Module, dataloader: DataLoader):
    accuracies = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    with torch.no_grad():
        for inp, target in tqdm(dataloader):
            inp = inp.to(device)
            target = target.to(device)
            outputs = model(inp)
            accuracies += (outputs.argmax(dim=1) == target).tolist()

    return torch.Tensor(accuracies)


class GapAttack:
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
