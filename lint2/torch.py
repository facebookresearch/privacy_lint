from typing import Callable, Optional, List, Dict, Type

import torch
from torch import nn, Tensor
import torch.utils.data
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

Model = nn.Module
Dataset = torch.utils.data.Dataset


def hinge_loss_conf(target: Tensor, pred: Tensor) -> Tensor:
    return (pred - F.one_hot(target) * pred).max(axis=1).values  # achtung! may work incorrectly


def neg_log_loss_conf(target: Tensor, pred: Tensor) -> Tensor:
    return -F.cross_entropy(pred, target, reduction='none')


class TorchApplier:
    """
    Describes how we should apply the model to get confidence scores (e.g. losses).

    Example:
    >>> def loss_conf(target: Tensor, pred: Tensor) -> Tensor:
    ...     loss = F.cross_entropy(pred, target)
    ...     return -loss
    ...
    >>> apply_fn = TorchApplier(loss_conf, batch_size=32)
    >>> apply_fn(model, data)
    array([1, 2, 3, 4, 5])
    """

    def __init__(
        self,
        conf_fn: Callable[[Tensor, Tensor], Tensor],
        batch_size: int = 128,
        transformations: Optional[List[Callable]] = None,
    ):
        self.conf_fn = conf_fn
        self.batch_size = batch_size
        self.transformations = transformations

    def __call__(self, model: Model, data: Dataset) -> Tensor:
        """Applies the model on the dataset, gets confidence scores: conf(model(X), y)"""

        loader = DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
        )
        if self.transformations is None:
            confs = torch.concat([  # [len(data)]
                self.conf_fn(target, model(input))
                for input, target in loader
            ])
        else:
            confs = torch.concat([  # [len(data), len(transformations)]
                torch.stack([
                    # TODO: check this doesn't apply the same transformation to all examples in a batch
                    self.conf_fn(target, model(transformation(input)))
                    for transformation in self.transformations
                ], dim=1)

                for input, target in loader
            ], dim=0)

        return confs.detach().numpy()


class LightningTrainer:
    def __init__(
        self,
        module: Type[pl.LightningModule],
        params: Dict = None,
        batch_size: int = 32,
        max_epochs: int = 10,
    ):
        self.module = module
        self.params = params or {}
        self.batch_size = batch_size
        self.max_epochs = max_epochs

    def __call__(self, dataset: Dataset) -> Model:
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            enable_model_summary=False,
        )

        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True,
        )

        model = self.module(**self.params)

        trainer.fit(model, data_loader)
        return model