# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Union

import numpy as np
import torch


def idx_to_mask(n_data, indices):
    mask = torch.zeros(n_data, dtype=bool)
    mask[indices] = 1

    return mask


def multiply_round(n_data: int, cfg: dict):
    """
    Given a configuration {split: percentage}, return a configuration {split: n} such that
    the sum of all is equal to n_data
    """
    print(cfg)
    s_total = sum(cfg.values())
    sizes = {name: int(s * n_data / s_total) for name, s in cfg.items()}

    max_name = max(sizes.items(), key=operator.itemgetter(1))[0]
    sizes[max_name] += n_data - sum(sizes.values())

    return sizes


def generate_subsets(
    mask: torch.Tensor, size_split: int, n_splits: int, prefix="split_"
):
    """
    size_split: number of samples in a split split
    n_splits: number of split splits
    """
    assert mask.ndim == 1
    idx = torch.nonzero(mask)[:, 0]

    split_masks = {}
    distribution = torch.ones_like(
        idx
    ).float()  # Each sample is drawn with equal probability
    for i_split in range(n_splits):
        idx_shadow = torch.multinomial(
            distribution, num_samples=size_split, replacement=False
        )
        split_masks[f"{prefix}{i_split}"] = idx_to_mask(mask.shape[0], idx[idx_shadow])

    return split_masks


def flatten(d: Union[dict, int]):
    if isinstance(d, dict):
        r = {}
        for k, v in d.items():
            if isinstance(v, dict):
                flat_v = flatten(v)
                for k2, v2 in flat_v.items():
                    r[f"{k}/{k2}"] = v2
            else:
                r[k] = v

        return r
    else:
        return d


def generate_splits(n_data: int, split_config: dict):
    """
    Generate splits for a dataset of n_data samples, with split_config specifying how to divide data samples

    """
    flat_config = flatten(split_config)
    flat_config = multiply_round(n_data, flat_config)

    permutation = np.random.permutation(n_data)
    masks = {}
    offset = 0
    for split, n_split in flat_config.items():
        masks[split] = idx_to_mask(n_data, permutation[offset : offset + n_split])
        offset += n_split

    return masks
