import operator

import numpy as np
import torch


def idx_to_mask(n_data, indices):
    mask = torch.zeros(n_data, dtype=bool)
    mask[indices] = 1

    return mask


def multiply_round(n_data, cfg):
    s_total = sum(cfg.values())
    sizes = {name: int(s * n_data / s_total) for name, s in cfg.items()}

    max_name = max(sizes.items(), key=operator.itemgetter(1))[0]
    sizes[max_name] += n_data - sum(sizes.values())

    return sizes


def create_splits(mask: torch.Tensor, size_split: int, n_splits: int, prefix="split_"):
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


def generate_masks(n_data: int, split_config: dict):
    """
    Generate masks for a dataset of n_data samples, with split_config specifying how to divide data samples

    """
    assert type(split_config) is dict
    assert "public" in split_config and "private" in split_config
    assert type(split_config["private"]) is dict

    permutation = np.random.permutation(n_data)
    if type(split_config["public"]) is dict:
        n_public = int(sum(split_config["public"].values()) * n_data)
    else:
        n_public = int(split_config["public"] * n_data)
    n_private = n_data - n_public

    known_masks = {}
    known_masks["public"] = idx_to_mask(n_data, permutation[:n_public])
    known_masks["private"] = idx_to_mask(n_data, permutation[n_public:])

    hidden_masks = {}

    hidden_masks["private"] = {}

    sizes = multiply_round(n_private, split_config["private"])
    offset = n_public
    for name, size in sizes.items():
        hidden_masks["private"][name] = idx_to_mask(
            n_data, permutation[offset : offset + size]
        )
        offset += size

    assert offset == n_data

    if type(split_config["public"]) is dict:
        hidden_masks["public"] = {}
        public_sizes = multiply_round(n_public, split_config["public"])
        print("Public", public_sizes)
        public_offset = 0
        for name, size in public_sizes.items():
            hidden_masks["public"][name] = idx_to_mask(
                n_data, permutation[public_offset : public_offset + size]
            )
            public_offset += size
        assert public_offset == n_public
    else:
        hidden_masks["public"] = known_masks["public"]

    return known_masks, hidden_masks
