#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Runs CIFAR10 training on subsets of data.
"""

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torch.utils.tensorboard as tensorboard
import torchvision.transforms as transforms
from privacy_lint.dataset import MaskDataset
from torchvision.datasets import CIFAR10
from tqdm import tqdm


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


def save_checkpoint(state, is_best, directory):
    torch.save(state, os.path.join(directory, "checkpoint.pth"))
    if is_best:
        shutil.copyfile(
            os.path.join(directory, "checkpoint.pth"),
            os.path.join(directory, "model_best.pth"),
        )


def accuracy(preds, labels):
    return (preds == labels).mean()


def train(args, model, train_loader, optimizer, epoch, device):
    start_time = datetime.now()

    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    for i, (images, target) in enumerate(tqdm(train_loader)):

        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()

        # measure accuracy and record loss
        acc1 = accuracy(preds, labels)

        losses.append(loss.item())
        top1_acc.append(acc1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print(
                f"\tTrain Epoch: {epoch} \t"
                f"Loss: {np.mean(losses):.6f} "
                f"Acc@1: {np.mean(top1_acc):.6f} "
            )
    train_duration = datetime.now() - start_time
    return train_duration


@torch.no_grad()
def test(args, model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    for images, target in tqdm(test_loader):
        images = images.to(device)
        target = target.to(device)

        output = model(images)
        loss = criterion(output, target)
        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()
        acc1 = accuracy(preds, labels)

        losses.append(loss.item())
        top1_acc.append(acc1)

    top1_avg = np.mean(top1_acc)

    print(f"\tTest set:" f"Loss: {np.mean(losses):.6f} " f"Acc@1: {top1_avg :.6f} ")
    return np.mean(top1_acc)


# flake8: noqa: C901
def main():
    args = parse_args()
    device = args.device

    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    train_transform = transforms.Compose(augmentations + normalize)

    test_transform = transforms.Compose(normalize)

    masks = torch.load(args.mask_path)
    mask = masks[args.mask]
    train_dataset = MaskDataset(
        CIFAR10(
            root=args.data_root, train=True, download=True, transform=train_transform
        ),
        mask=mask,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
    )

    test_dataset = CIFAR10(
        root=args.data_root, train=False, download=True, transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    best_acc1 = 0

    model = convnet(num_classes=10)
    model = model.to(device)

    if args.optim == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError("Optimizer not recognized. Please check spelling")

    # Store some logs
    accuracy_per_epoch = []
    time_per_epoch = []

    for epoch in range(args.epochs):
        if args.lr_schedule == "cos":
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / (args.epochs + 1)))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        train_duration = train(args, model, train_loader, optimizer, epoch, device)
        top1_acc = test(args, model, test_loader, device)

        # remember best acc@1 and save checkpoint
        is_best = top1_acc > best_acc1
        best_acc1 = max(top1_acc, best_acc1)

        time_per_epoch.append(train_duration)
        accuracy_per_epoch.append(float(top1_acc))

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": "Convnet",
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            directory=args.checkpoint_dir,
        )

    time_per_epoch_seconds = [t.total_seconds() for t in time_per_epoch]
    avg_time_per_epoch = sum(time_per_epoch_seconds) / len(time_per_epoch_seconds)
    metrics = {
        "accuracy": best_acc1,
        "accuracy_per_epoch": accuracy_per_epoch,
        "avg_time_per_epoch_str": str(timedelta(seconds=int(avg_time_per_epoch))),
        "time_per_epoch": time_per_epoch_seconds,
    }

    print(metrics)


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument(
        "-j",
        "--workers",
        default=2,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        help="mini-batch size for test dataset, this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="SGD momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0,
        type=float,
        metavar="W",
        help="SGD weight decay",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=".",
        help="path to save check points",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../cifar10",
        help="Where CIFAR10 is/will be stored",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/tmp/stat/tensorboard",
        help="Where Tensorboard log will be stored",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="SGD",
        help="Optimizer to use (Adam, RMSprop, SGD)",
    )
    parser.add_argument(
        "--lr-schedule", type=str, choices=["constant", "cos"], default="cos"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device on which to run the code.",
    )

    parser.add_argument(
        "--mask_path", type=str, required=True, help="Path to masks file"
    )

    parser.add_argument(
        "--mask", type=str, required=True, help="Name of the mask to use on data"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
