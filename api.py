from utils.masks import generate_masks, evaluate_masks
import torch

def train(*args, **kwargs):
    return {}
params = {}

n_data = 400

# Specify dataset configuration
split_config = {
    "public": 0.5, # 50% of the data will be in the public bucket
    "private": {
        "train": 0.25,
        "heldout": 0.25
    }
}

# Randomly split the data according to the configuration
known_masks, hidden_masks = generate_masks(n_data, split_config)

print(known_masks, hidden_masks)

# Typical output
typical_known_masks = {
    # Data sample number 0 is in the private set, data sample 1 ...
    "public":  [0, 1, 1, 0],
    "private": [1, 0, 0, 1]
}
typical_hidden_masks = {
    "public": [0, 1, 1, 0],
    "private": {
        "train":   [1, 0, 0, 0],
        "heldout": [0, 0, 0, 1]
    }
}

# Private model is trained once
model_private = train(params, hidden_masks["private"]["train"])


# Attacker can then use the "public masks" that he knows about to make their privacy attacks
# Note that the attacker does not have access to hidden_masks
model_public = train(params, known_masks["public"])

def privacy_attack(model_private, private_masks):
    """
    Random attack model
    """
    return torch.rand(len(private_masks))


guessed_membership = privacy_attack(model_private, known_masks["private"])
# guessed_membership is typically something like [0.5, 0.7]

# At evaluation time, the guessed membership are compared to the true ones
# Only then can hidden_masks be checked
# import ipdb;ipdb.set_trace()
print(evaluate_masks(guessed_membership, hidden_masks["private"], threshold=0.5))
# Computes precision, recall, accuracy, etc. 