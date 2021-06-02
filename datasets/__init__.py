import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, TensorDataset

from .text_data import TextIterator

class IdxDataset(Dataset):
    """
    Wraps a dataset so that with each element is also returned its index
    """
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, i: int):
        sample = self.dataset[i]
        if type(sample) is tuple:
            sample = list(sample)
            sample.insert(0, i)
            return tuple(sample)
        else:
            return i, sample

    def __len__(self):
        return len(self.dataset)


class MaskDataset(Dataset):
    def __init__(self, dataset: Dataset, mask: torch.Tensor):
        """
        example:
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


def get_transform(dataset):
    if dataset == "cifar10":
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    return transform


def get_dataset(*, params, is_train, mask=None):
    if is_train:
        assert mask is not None
    if params.dataset == "cifar10":
        if is_train:
            transform = get_transform(params.dataset)
        else:
            transform = get_transform(params.dataset)

        dataset = torchvision.datasets.CIFAR10(root=params.data_root, train=is_train, download=True, transform=transform)
        dataset = IdxDataset(dataset)
        if mask is not None:
            dataset = MaskDataset(dataset, mask)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
        n_data = len(dataset)
        params.num_classes = 10

        return dataloader, n_data

    elif params.dataset == "dummy":
        # Creates a dummy dataset for NLP
        n_data, delta = 10000, 3
        data = torch.randint(-delta, delta, size=(n_data, params.seq_len))
        data = torch.cumsum(data, dim=1)
        data = torch.remainder(data, params.n_vocab)

        iterator = TextIterator(data.view(-1), params.batch_size, params.seq_len)

        return iterator, n_data



