import torch
from torch.utils.data import Dataset, TensorDataset


class MaskDataset(Dataset):
    def __init__(self, dataset: Dataset, mask: torch.Tensor):
        """
        Creating a subset of original dataset, where only samples with mask=1 are included
        TODO: Add test of this
        Example:
        mask: [0, 1, 1]
        cumul: [-1, 0, 1]
        remap: {0: 1, 1: 2}
        """
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
