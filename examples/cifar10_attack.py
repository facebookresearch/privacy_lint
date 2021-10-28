# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from privacy_lint.attacks import ShadowModelsAttack
from privacy_lint.dataset import MaskDataset


def convnet(num_classes):
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(128, num_classes, bias=True),
    )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=256)
    parser.add_argument("--workers", default=2)
    parser.add_argument(
        "--data-root", default="/private/home/asablayrolles/data"
    )  # default="../cifar10")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    splits = torch.load("data/cifar_splits.pth")
    models_path = {
        "public/split_0": "/path/to/checkpoint_0.pth",
        "public/split_1": "/path/to/checkpoint_1.pth",
        "public/split_2": "/path/to/checkpoint_2.pth",
        "public/split_3": "/path/to/checkpoint_3.pth",
        "public/split_4": "/path/to/checkpoint_4.pth",
        "public/split_5": "/path/to/checkpoint_5.pth",
        "public/split_6": "/path/to/checkpoint_6.pth",
        "public/split_7": "/path/to/checkpoint_7.pth",
        "public/split_8": "/path/to/checkpoint_8.pth",
        "public/split_9": "/path/to/checkpoint_9.pth",
    }

    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    test_transform = transforms.Compose(normalize)

    public_dataset = MaskDataset(
        CIFAR10(
            root=args.data_root, train=True, download=True, transform=test_transform
        ),
        mask=splits["public"],
    )

    public_dataloader = torch.utils.data.DataLoader(
        public_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    models = {}
    for split, path in models_path.items():
        model = convnet(10)
        ckpt = torch.load(path)
        model.load_state_dict(ckpt["state_dict"])
        models[split] = model

    masks = {k: v[splits["public"]] for k, v in splits.items() if "public/split" in k}
    attack = ShadowModelsAttack(masks, models, public_dataloader, verbose=True)

    # We load the model trained on the private dataset
    target_model = convnet(10)
    ckpt = torch.load("/path/to/private/checkpoint.pth")
    target_model.load_state_dict(ckpt["state_dict"])

    private_train_loader = torch.utils.data.DataLoader(
        MaskDataset(
            CIFAR10(
                root=args.data_root, train=True, download=True, transform=test_transform
            ),
            mask=splits["private/train"],
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    private_test_loader = torch.utils.data.DataLoader(
        MaskDataset(
            CIFAR10(
                root=args.data_root, train=True, download=True, transform=test_transform
            ),
            mask=splits["private/heldout"],
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    results = attack.launch(target_model, private_train_loader, private_test_loader)
    print(f"Attack accuracy: {results.get_accuracy(0.5)}")
