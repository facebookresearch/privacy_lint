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
        'public': [1, 1, 1, 0, 0],
        'public/split_0': [1, 0, 1, 0, 0],
        'public/split_1': ...
    }
    """
    coarse_masks = generate_splits(n_data, split_config)
    splits = generate_subsets(
        coarse_masks["public"],
        size_split=int(0.25 * n_data),
        n_splits=10,
        prefix="public/split_",
    )

    splits.update(coarse_masks)

    return splits


if __name__ == "__main__":
    n_data = 50000
    splits = divide_data(n_data)

    torch.save(splits, "data/cifar_splits.pth")
