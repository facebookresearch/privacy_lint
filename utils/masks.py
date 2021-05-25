import numpy as np
import torch

import operator

def to_mask(n_data, indices):
    mask = torch.zeros(n_data, dtype=bool)
    mask[indices] = 1

    return mask


def multiply_round(n_data, cfg):
    s_total = sum(cfg.values())
    sizes = {name: int(s * n_data / s_total) for name, s in cfg.items()}

    max_name = max(sizes.items(), key=operator.itemgetter(1))[0]
    sizes[max_name] += n_data - sum(sizes.values())

    return sizes


def generate_masks(n_data, split_config):
    assert type(split_config) is dict
    assert "public" in split_config and "private" in split_config
    assert type(split_config["private"]) is dict

    permutation = np.random.permutation(n_data)
    n_public = int(split_config["public"] * n_data)
    n_private = n_data - n_public

    known_masks = {}
    known_masks["public"] = to_mask(n_data, permutation[:n_public])
    known_masks["private"] = to_mask(n_data, permutation[n_public:])

    hidden_masks = {}
    hidden_masks["public"] = known_masks["public"]
    hidden_masks["private"] = {}

    sizes = multiply_round(n_private, split_config["private"])
    print(sizes)
    offset = n_public
    for name, size in sizes.items():
        hidden_masks["private"][name] = to_mask(n_data, permutation[offset:offset+size])
        offset += size

    assert offset == n_data


    return known_masks, hidden_masks

def evaluate_masks(guessed_membership, private_masks, threshold):
    true_positives = (guessed_membership[private_masks["train"]] >= threshold).float()
    true_negatives = (guessed_membership[private_masks["heldout"]] < threshold).float()
    false_negatives = (guessed_membership[private_masks["heldout"]] >= threshold).float()
    
    recall = torch.sum(true_positives) / torch.sum(private_masks["train"].float())
    precision = torch.sum(true_positives) / (torch.sum(true_positives) + torch.sum(false_negatives))

    accuracy = (torch.sum(true_positives) + torch.sum(true_negatives)) / (torch.sum(private_masks["heldout"].float()) + torch.sum(private_masks["train"].float()))

    return precision, recall, accuracy
