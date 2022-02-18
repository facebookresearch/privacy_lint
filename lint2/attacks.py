from typing import Callable, Optional

from scipy import stats
from torch import nn, Tensor
import torch.utils.data
import numpy as np

from lint2.torch import TorchApplier, neg_log_loss_conf
from privacy_lint.dataset import MaskDataset

Model = nn.Module
Dataset = torch.utils.data.Dataset


DEFAULT_APPLY_FN = TorchApplier(conf_fn=neg_log_loss_conf)


# TODO: parallelized model training
# TODO: track attack metrics progressively as we have more models


def random_mask(size: int, ones: int) -> np.ndarray:
    assert size >= 0 and ones >= 0
    assert size - ones >= 0

    mask = np.zeros(size, dtype=bool)
    mask[:ones] = True
    np.random.shuffle(mask)
    return mask


def generate_repeated2fold_masks(n: int, len: int):
    mask = np.empty(0)
    for i in range(n):
        if i % 2 == 0:
            mask = random_mask(len, len // 2)
        else:
            mask = ~mask
        yield i, mask


def agg_cols_masked(f, arr, masks, **args):
    return np.array([
        f(arr[masks[:, i], i], **args)
        for i in range(masks.shape[1])
    ])


class ScoreAttackResult:
    def __init__(
        self,
        confs_target,  # [len(dataset), num_transform]
        confs,  # [num_shadow, len(dataset), num_transform], float
        masks,  # [num_shadow, len(dataset), num_transform], bool
    ):
        self.confs_target = confs_target
        self.confs = confs
        self.masks = masks

    def __len__(self):
        return len(self.confs_target)

    def first_n_models(self, n: int):
        """Show results based on first `n` shadow models only"""
        return ScoreAttackResult(
            self.confs_target,
            self.confs[:n, :],
            self.masks[:n, :],
        )

    def transform_confs(self, f: Callable[[np.ndarray], np.ndarray]):
        """Apply transformation to confidence scores, i.e. rescale them with logit function"""
        return ScoreAttackResult(
            f(self.confs_target),
            f(self.confs),
            self.masks,
        )

    def member_scores(
        self,
        online: bool = False,
        individual_sigma: bool = False,
        gaussian_likelihood: bool = False,
    ):
        masks_in = self.masks
        masks_out = ~masks_in

        if self.confs.shape[1] == 0:
            # just confs if no shadow models
            return self.confs

        mu_out = agg_cols_masked(np.mean, self.confs, masks_out, axis=0)  # [len(dataset)] / [len(dataset), num_transforms]
        if online:
            mu_in = agg_cols_masked(np.mean, self.confs, masks_in, axis=0)

        univariate = len(mu_out.shape) == 1

        # TODO: simplify this
        if univariate:
            if gaussian_likelihood:
                sigma_out = (
                    agg_cols_masked(np.std, self.confs, masks_out)
                    if individual_sigma else np.std(self.confs[masks_out])
                )
                if online:
                    sigma_in = (
                        agg_cols_masked(np.std, self.confs, masks_in)
                        if individual_sigma else np.std(self.confs[masks_in])
                    )
                    return (
                        stats.norm(loc=mu_in, scale=sigma_in).logpdf(self.confs_target)
                        - stats.norm(loc=mu_out, scale=sigma_out).logpdf(self.confs_target)
                    )
                else:
                    return stats.norm(loc=mu_out, scale=sigma_out).cdf(self.confs_target)
            else:
                if online:
                    return self.confs_target - (mu_in + mu_out) / 2
                else:
                    return self.confs_target - mu_out
        else:
            if gaussian_likelihood:
                # Multivariate Gaussian
                sigma_out = (
                    np.stack([
                        np.cov(self.confs[masks_out[:, i], i], rowvar=False)
                        for i in range(masks_out.shape[1])
                        if masks_out[:, i].sum() > 1
                    ], axis=0)
                    if individual_sigma else np.cov(self.confs[masks_out], rowvar=False)
                )
                if online:
                    sigma_in = (
                        np.stack([
                            np.cov(self.confs[masks_in[:, i], i], rowvar=False)
                            for i in range(masks_in.shape[1])
                            if masks_in[:, i].sum() > 1
                        ], axis=0)
                        if individual_sigma else np.cov(self.confs[masks_in], rowvar=False)
                    )
                    return np.array([
                        stats.multivariate_normal(mean=mu_in[i, :], scale=sigma_in[i, :]).logpdf(self.confs_target[i, :])
                        - stats.multivariate_normal(loc=mu_out[i, :], scale=sigma_out[i, :]).logpdf(self.confs_target[i, :])
                        for i in range(self.confs_target.shape[0])
                    ])
                else:
                    raise RuntimeError("WTF?")
            else:
                # Euclidean norm
                if online:
                    return np.linalg.norm(self.confs_target - (mu_in + mu_out) / 2, axis=1)
                else:
                    return np.linalg.norm(self.confs_target - mu_out, axis=1)



class ScoreAttack:
    def __init__(
        self,
        num_shadow: int,
    ):
        # parameters
        self.num_shadow = num_shadow  # TODO: specify split generator instead of number

        # state
        self.models = None
        self.masks = None

    def fit(
        self,
        data_public: Dataset,
        train_fn: Callable[[Dataset], Model],
        apply_fn: Callable[[Model, Dataset], Tensor] = DEFAULT_APPLY_FN,
    ):
        self.fit_predict(data_public, model_target=None, train_fn=train_fn, apply_fn=apply_fn)
        return self

    def fit_predict(
        self,
        data_target: Dataset,
        model_target: Optional[Model],
        train_fn: Callable[[Dataset], Model],
        apply_fn: Callable[[Model, Dataset], Tensor] = DEFAULT_APPLY_FN,
    ):
        """
        Train adversary and apply online attack on the data known on training time.
        """

        self.models = []
        self.masks = np.zeros((self.num_shadow, len(data_target)), dtype=bool)
        confs = []

        # Train shadow models
        for n, mask in generate_repeated2fold_masks(self.num_shadow, len(data_target)):
            print(f"Preparing shadow model {n+1}/{self.num_shadow}")

            # train model
            data = MaskDataset(data_target, mask)
            model = train_fn(data)
            self.models.append(model)

            # collecting confs
            confs.append(apply_fn(model, data_target))
            self.masks[n, :] = mask

        confs = np.stack(confs, axis=0)

        # Collecting confs for the target model
        confs_target = None
        if model_target is not None:
            confs_target = apply_fn(model_target, data_target)  # [len(dataset)] / [len(dataset), len(trans)]

        return ScoreAttackResult(
            confs_target=confs_target,
            confs=confs,
            masks=self.masks,
        )

    def predict(
        self,
        data_target: Dataset,
        model_target: Model,
        apply_fn: Callable[[Model, Dataset], Tensor] = DEFAULT_APPLY_FN,
    ):
        """
        Predict membership by applying offline attack on a new dataset.
        """
        confs_target = apply_fn(model_target, data_target)  # [len(dataset)]
        confs = np.stack([  # [num_shadow, len(dataset)]
            apply_fn(model, data_target)
            for model in self.models
        ], axis=0)
        masks = np.zeros_like(confs, dtype=bool)

        return ScoreAttackResult(
            confs_target=confs_target,
            confs=confs,
            masks=masks,
        )

