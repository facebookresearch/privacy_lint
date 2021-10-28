# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from privacy_lint.attack_results import AttackResults
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


@torch.no_grad()
def compute_softmax(model: nn.Module, dataloader: DataLoader) -> torch.Tensor:
    softmaxes, labels = [], []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    for img, target in tqdm(dataloader):
        img = img.to(device)
        outputs = F.softmax(model(img), dim=-1)

        softmaxes.append(outputs.cpu())
        labels.append(target)

    return torch.cat(softmaxes, dim=0), torch.cat(labels, dim=0)


def train_shadow(
    train_X: torch.Tensor, train_Y: torch.Tensor, verbose: bool = False
) -> nn.Module:
    n, d = tuple(train_X.shape)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"{n} data in dimension {d}")

    dataset = TensorDataset(train_X, train_Y)
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    model = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, 2))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(10):

        losses = []
        accuracies = []
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            accuracies += (output.argmax(dim=1) == y).int().tolist()

        avg_loss = sum(losses) / len(losses)
        print(f"Avg loss: {avg_loss:.2f}, Acc: {sum(accuracies)/len(accuracies):.2f}")

    return model


@torch.no_grad()
def run_attack(
    target_model: nn.Module, dataloader: DataLoader, attack_models: dict
) -> torch.Tensor:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    target_model.to(device)

    scores = []
    for img, target in tqdm(dataloader):
        img = img.to(device)
        target = target.to(device)
        outputs = F.softmax(target_model(img), dim=1)
        for i in range(img.size(0)):
            scores.append(
                F.softmax(attack_models[target[i].item()](outputs[i : i + 1]), dim=1)[
                    0, 1
                ].item()
            )

    return torch.Tensor(scores)


class ShadowModelsAttack:
    def __init__(
        self, masks: dict, models: dict, public_data: DataLoader, verbose: bool = False
    ):
        """
        masks: dictionary of name to mask
        models: dictionary of name to model
        """
        self.verbose = verbose

        self.softmaxes, self.labels, self.masks = [], [], []
        for split, model in models.items():
            softmaxes, labels = compute_softmax(model, public_data)
            self.softmaxes.append(softmaxes)
            self.labels.append(labels)
            self.masks.append(masks[split])
            break

        self.softmaxes = torch.cat(self.softmaxes, dim=0)
        self.labels = torch.cat(self.labels, dim=0)
        self.masks = torch.cat(self.masks, dim=0)

        self.train_attack_models()

    def train_attack_models(self):
        self.attack_models = {}
        for label in range(self.labels.max() + 1):
            print(f"Training shadow model on label {label}")
            train_X = self.softmaxes[self.labels == label]
            train_Y = self.masks[self.labels == label]
            self.attack_models[label] = train_shadow(
                train_X, train_Y.long(), verbose=self.verbose
            )

    def launch(
        self,
        private_model: nn.Module,
        private_train: DataLoader,
        private_heldout: DataLoader,
    ):
        scores_train = run_attack(private_model, private_train, self.attack_models)
        scores_heldout = run_attack(private_model, private_heldout, self.attack_models)

        return AttackResults(scores_train, scores_heldout)
