# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
from typing import Union, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# TODO: Remove this submodule, use torch.utils.data instead

class MaskDataset(Dataset):
    def __init__(self, dataset: Dataset, mask: Union[torch.Tensor, np.ndarray]):
        """
        Creating a subset of original dataset, where only samples with mask=1 are included
        TODO: Add test of this
        Example:
        mask: [0, 1, 1]
        cumul: [-1, 0, 1]
        remap: {0: 1, 1: 2}
        """

        if isinstance(mask, np.ndarray):
            mask = torch.Tensor(mask).bool()
        assert mask.dim() == 1
        assert mask.size(0) == len(dataset)
        assert mask.dtype == torch.bool

        mask = mask.long()
        cumul = torch.cumsum(mask, dim=0) - 1
        self.remap = {}
        for i in range(mask.size(0)):
            if mask[i] == 1:
                self.remap[cumul[i].item()] = i
            assert mask[i] in [0, 1]

        self.dataset = dataset
        self.mask = mask
        self.length = cumul[-1].item() + 1

    def __getitem__(self, i: int):
        return self.dataset[self.remap[i]]

    def __len__(self):
        return self.length


class UnitedDataset(Dataset):
    def __init__(self, dataset_1: Dataset, dataset_2: Dataset):
        self.dataset_1 = dataset_1
        self.dataset_2 = dataset_2
        self.indices = [(1, i) for i in range(len(dataset_1))] + [(2, i) for i in range(len(dataset_2))]
        random.shuffle(self.indices)

    def __len__(self):
        return len(self.dataset_1) + len(self.dataset_2)

    def __getitem__(self, i: int):
        dataset_n, idx = self.indices[i]
        dataset = self.dataset_1 if dataset_n == 1 else self.dataset_2
        return dataset[idx]
