from typing import Optional, Union

import torch
from privacy_lint.dataset.masks import generate_splits, generate_subsets

_default_split_config = {
    "public": 0.5,  # 50% of the data will be in the public bucket
    "private": {"train": 0.25, "heldout": 0.25},
}


def divide_data(n_data: int, split_config: dict = _default_split_config) -> dict:
    """
    Divides data into subsets, according to the configuration split_config
    n_data: total number of

    Returns: Dict of {subset: mask}
    e.g. {
        'private/train': [0, 0, 0, 1, 0],
        'private/heldout': [0, 0, 0, 0, 1],
        'split_0': [1, 0, 1, 0, 0],
        'split_1': ...
    }
    """
    coarse_masks = generate_splits(n_data, split_config)
    splits = generate_subsets(
        coarse_masks["public"],
        size_split=int(0.25 * n_data),
        n_splits=10,
        prefix="public/split_",
    )
    # {'split_0': [1, 0, 1, 0, 0], 'split_1': ...}
    splits["private/train"] = coarse_masks["private/train"]
    splits["private/heldout"] = coarse_masks["private/heldout"]

    torch.save(splits, "data/splits.pth")

if __name__ == "__main__":
    n_data = 50000
    divide_data(n_data)
