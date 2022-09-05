import os
import numpy as np
import pytorch_lightning as pl
import torchmetrics
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from opacus.data_loader import DPDataLoader
from opacus import PrivacyEngine
from privacy_lint.dataset import MaskDataset
import torchvision
from tqdm import tqdm
import itertools
from torchvision.transforms import functional as TF

import submitit


class SimpleConvNet(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 10,
        lr: float = 0.1,
        enable_dp: bool = True,
        delta: float = 1e-5,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
    ):
        """A simple conv-net for classifying MNIST with differential privacy training
        Args:
            lr: Lesarning rate
            enable_dp: Enables training with privacy guarantees using Opacus (if True), vanilla SGD otherwise
            delta: Target delta for which (eps, delta)-DP is computed
            noise_multiplier: Noise multiplier
            max_grad_norm: Clip per-sample gradients to this norm
        """
        super().__init__()

        # Hyper-parameters
        self.lr = lr

        # Network
        self.classifier = nn.Sequential(
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

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

        # Differential privacy
        self.enable_dp = enable_dp
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        if self.enable_dp:
            self.privacy_engine = PrivacyEngine()

    def forward(self, x):
        return self.classifier(x)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0)

        if self.enable_dp:
            data_loader = (
                # soon there will be a fancy way to access train dataloader,
                # see https://github.com/PyTorchLightning/pytorch-lightning/issues/10430
                self.trainer._data_connector._train_dataloader_source.dataloader()
            )

            # transform (model, optimizer, dataloader) to DP-versions
            if hasattr(self, "dp"):
                self.dp["model"].remove_hooks()
            dp_model, optimizer, dataloader = self.privacy_engine.make_private(
                module=self,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=isinstance(data_loader, DPDataLoader),
            )
            self.dp = {"model": dp_model}

        return optimizer

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)
        self.train_accuracy(output, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)
        self.test_accuracy(output, target)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", self.test_accuracy, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        if self.enable_dp:
            # Logging privacy spent: (epsilon, delta)
            epsilon = self.privacy_engine.get_epsilon(self.delta)
            self.log("epsilon", epsilon, on_epoch=True, prog_bar=True)


def create_resnet_model():
    model = torchvision.models.wide_resnet50_2(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model


class LitResnet(pl.LightningModule):
    def __init__(self, lr=0.1, weight_decay=5e-4):
        super().__init__()
        self.save_hyperparameters()

        self.classifier = create_resnet_model()

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        out = self.classifier(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)
        self.train_accuracy(output, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = F.cross_entropy(output, target)
        self.test_accuracy(output, target)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", self.test_accuracy, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


class AttackAugmentation(torch.nn.Module):
    def __init__(self, dx=0, dy=0, flip=False, padding=4):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.flip = flip
        self.padding = padding

    def forward(self, img):
        padding = 4
        h, w = TF.get_image_size(img)
        if self.flip:
            img = TF.hflip(img)
        img = TF.pad(img, self.padding, fill=0, padding_mode='constant')
        img = TF.crop(img, padding + self.dy, padding + self.dx, h, w)
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(dx={self.dx}, dy={self.dy}, flip={self.flip})"


DATA_ROOT = './data'
BENCHMARK_PATH = 'benchmark_data/cifar10_epoch100_dp_noise1/'


def train_shadow_model(shadow_idx):
    # Load Masks
    masks = np.load(os.path.join(BENCHMARK_PATH, 'masks.npy'))

    # Data Loaders
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    dataset_train = MaskDataset(
        CIFAR10(
            root=DATA_ROOT,
            train=True,
            download=True,
            transform=transforms.Compose(augmentations + normalize),
        ),
        mask=masks[shadow_idx, :],
    )
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=256,
    )

    dataset = CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transforms.Compose(normalize),
    )
    loader_all = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
    )

    # Training

    model = SimpleConvNet(enable_dp=True)
    # model = LitResnet()

    trainer = pl.Trainer(
        max_epochs=100,
        enable_model_summary=False,
        gpus=1,
    )
    trainer.fit(model, loader_train)

    trainer.save_checkpoint(os.path.join(BENCHMARK_PATH, f'shadow_{shadow_idx:04d}.ckpt'))

    # Initialize augmentations

    dx_values = sorted(range(-4, 4 + 1), key=lambda x: abs(x))
    dy_values = sorted(range(-4, 4 + 1), key=lambda x: abs(x))
    flip_values = [False, True]

    attack_augmentations = [
        AttackAugmentation(dx, dy, flip)
        for dx, dy, flip in itertools.product(dx_values, dy_values, flip_values)
    ]

    # Predicting with Augmentations

    model_cuda = model.to('cuda').eval()

    values = []
    targets = []
    with torch.no_grad():
        for input, target in tqdm(loader_all):
            targets.append(target)

            x = input.cuda()
            xx = torch.stack([
                model_cuda(transformation(x))
                for transformation in attack_augmentations
            ], dim=1)

            values.append(xx)

    preds = torch.cat(values, dim=0)
    targets = torch.cat(targets, dim=0)

    preds = np.array(preds.detach().cpu())
    targets = np.array(targets.detach().cpu())

    acc = (preds[:, 0, :].argmax(1) == targets).mean()
    print('Accuracy:', acc)

    # logit using the scheme from the LiRA paper
    # https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/score.py

    p = np.array(
        np.exp(preds - np.max(preds, -1, keepdims=True)),
        dtype=np.float64
    )
    p = p / np.sum(p, -1, keepdims=True)

    y_true = p[np.arange(p.shape[0]), :, targets].copy()

    p[np.arange(p.shape[0]), :, targets] = 0
    y_wrong = p.sum(axis=-1)

    logit = np.log(y_true + 1e-45) - np.log(y_wrong + 1e-45)

    np.save(os.path.join(BENCHMARK_PATH, f'logit_{shadow_idx:04d}.npy'), logit)


def submit_jobs():
    executor = submitit.AutoExecutor(folder="log_test")
    executor.update_parameters(
        gpus_per_node=1,
        tasks_per_node=1,  # one task per GPU
        cpus_per_task=10,  # 10 cpus per gpu is generally good
        nodes=1,
        timeout_min=60 * 10,
        # Below are cluster dependent parameters
        slurm_account="all",
        slurm_partition="lowpri",
        slurm_signal_delay_s=120,
        slurm_array_parallelism=50,
    )

    executor.map_array(train_shadow_model, [i for i in range(256) if not os.path.exists(
        os.path.join(BENCHMARK_PATH, f'logit_{i:04d}.npy'))])

submit_jobs()
