import os

import numpy as np
import pytorch_lightning as pl
import torchmetrics
import torch
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
from scipy import interpolate
from sklearn import metrics
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms, Compose

import lint2
import lint2.metrics
from lint2.attacks import ScoreAttack
from lint2.torch import LightningTrainer, TorchApplier, neg_log_loss_conf
from privacy_lint.dataset import MaskDataset


class ConvNet(pl.LightningModule):
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
        # TODO: Is this AlexNet or what?
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


def run_experiment():
    data_root = './data'
    results_root = './data/attack_cifar10/'
    os.makedirs(results_root, exist_ok=True)


    ## Prepare Dataset

    dataset = CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
    )

    # Split dataset to train/test
    split_size = 5000

    mask = np.zeros(len(dataset), dtype=int)
    mask[:split_size] = 1
    mask[split_size:2 * split_size] = 2

    np.random.seed(123)
    np.random.shuffle(mask)

    mask_train = (mask == 1)
    mask_holdout = (mask == 2)

    mask_test = (mask_train | mask_holdout)
    member_true = mask_train[mask_test]

    dataset_train = MaskDataset(dataset, mask_train)
    dataset_holdout = MaskDataset(dataset, mask_holdout)
    dataset_test = MaskDataset(dataset, mask_test)

    ### Model training definition & Training target model

    train_fn = LightningTrainer(
        ConvNet,
        params=dict(enable_dp=False, lr=0.1),
        batch_size=32,
        max_epochs=20,
    )
    apply_fn = TorchApplier(
        neg_log_loss_conf,
        batch_size=128,
    )

    model_target = train_fn(dataset_train)

    torch.save(model_target, os.path.join(results_root, 'cifar10_model_target.pt'))

    ### Train Attack

    attack = ScoreAttack(num_shadow=10)

    result_online = attack.fit_predict(
        dataset_test,
        model_target,
        train_fn=train_fn,
        apply_fn=apply_fn,
    )

    result_offline = attack.predict(
        dataset_test,
        model_target,
        apply_fn=apply_fn,
    )

    torch.save(attack, os.path.join(results_root, 'cifar10_attack.pt'))
    torch.save(result_online, os.path.join(results_root, 'cifar10_attack_result_online.pt'))
    torch.save(result_offline, os.path.join(results_root, 'cifar10_attack_result_offline.pt'))

    ### Multiple queries

    result_multivariate = attack.predict(
        dataset_test,
        model_target,
        apply_fn=TorchApplier(
            neg_log_loss_conf,
            batch_size=128,
            transformations=[
                ## List of augmentations
                transforms.Compose([]),  # Ident transform (original image)
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]),
            ]
        )
    )

    torch.save(result_multivariate, os.path.join(results_root, 'cifar10_attack_result_multivariate.pt'))

    ### Logit scaling

    def logit_transform(l):
        p = np.exp(l)  # -l?
        return np.log(p + 1e-8) - np.log(1 - p + 1e-8)

    result_logit = result_online.transform_confs(logit_transform)
    torch.save(result_multivariate, os.path.join(results_root, 'cifar10_attack_result_logit.pt'))


    ### Analyse attack results

    member_preds = result_online.member_scores(online=True)

    best_thresh = np.median(member_preds)
    accuracy = metrics.accuracy_score(member_true, member_preds > best_thresh)
    balanced_accuracy = metrics.balanced_accuracy_score(member_true, member_preds > best_thresh)

    fpr, tpr, thresholds = metrics.roc_curve(member_true, member_preds)

    tpr_at_fpr = interpolate.interp1d(fpr, tpr)
    tpr_at_m3 = tpr_at_fpr(1e-3)
    tpr_at_m1 = tpr_at_fpr(1e-1)

    time_to_fp = lint2.metrics.time_to_first_error(member_true, member_preds)

    report = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "tpr_at_m3": tpr_at_m3,
        "tpr_at_m1": tpr_at_m1,
        "time_to_fp": time_to_fp,
        "roc_curve": {
            "thresholds": thresholds,
            "fpr": fpr,
            "tpr": tpr,
        },
    }

    torch.save(report, os.path.join(results_root, "report.pt"))

    print("Accuracy:", accuracy)
    print("TPR @ 0.1%:", tpr_at_m3)
    print("Time to FP:", time_to_fp)



if __name__ == '__main__':
    run_experiment()


