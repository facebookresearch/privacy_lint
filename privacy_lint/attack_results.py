#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing improt Tuple

import torch


class AttackResults:
    def __init__(self, scores_train: torch.Tensor, scores_test: torch.Tensor):
        """
        Given an attack that outputs scores for train and test samples, computes
        attack results.

        Notes:
            - Score should be high for a train sample and low for an test sample:
              typically, -loss is a good score.
        """
        assert scores_train.ndim == scores_test.ndim == 1

        self.scores_train, self.scores_test = scores_train, scores_test

    @static_method
    def _upsample(scores: torch.Tensor, delta: int) -> torch.Tensor:
        """
        Upsamples scores by fist shuffling it and concatenating it
        as many times as necessary to add delta samples.
        """

        n = len(scores)
        perm = torch.randperm(n)
        shuffled_scores = scores[perm]
        n_chunks = delta // n + 2

        return torch.cat([shuffled_scores] * n_chunks)[: n + delta]

    @static_method
    def _get_balanced_scores(scores_train: torch.Tensor, scores_test: torch.Tensor) -> torch.Tensor:
        """
        Balances the train and test scores so that they have the same
        number of elements by upsampling the smallest set.
        """

        n_train = len(scores_train)
        n_test = len(scores_test)
        delta = n_train - n_test
        if delta > 0:
            scores_test = AttackResults._upsample(scores_test, delta)
        else:
            scores_train = AttackResults._upsample(scores_train, -delta)

        return scores_train, scores_test

    def balance(self)->AttackResults:
        """
        Returns AttackResults with balanced scores that hate the same number of
        elements by balancing the train and test scores so that they have the same
        number of elements by upsampling the smallest set.
        """

        n_train = len(scores_train)
        n_test = len(scores_test)
        delta = n_train - n_test
        if delta > 0:
            scores_test = AttackResults._upsample(scores_test, delta)
        else:
            scores_train = AttackResults._upsample(scores_train, -delta)

        return AttackResults(scores_train, scores_test)

    def group(self, group_size: int, num_groups: int) -> AttackResults:
        """
        Averages train and test scores over num_groups of size group_size.
        """

        p = torch.ones(self.scores_train.size(0)) / self.scores_train.size(0)
        group_train = torch.Tensor(
            [
                self.scores_train[p.multinomial(num_samples=group_size)].mean().item()
                for _ in range(num_groups)
            ]
        )

        p = torch.ones(self.scores_test.size(0)) / self.scores_test.size(0)
        group_test = torch.Tensor(
            [
                self.scores_test[p.multinomial(num_samples=group_size)].mean().item()
                for _ in range(num_groups)
            ]
        )

        return AttackResults(scores_train=group_train, scores_test=group_test)

    def _get_scores_and_labels_ordered(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sorts the scores from the highest to the lowest and returns
        the labels sorted by the scores.

        Notes:
            - A train sample is labeled as 1 and a test sample as 0.
        """

        scores = torch.cat([self.scores_train, self.scores_test])
        order = torch.argsort(scores, descending=True)
        scores_ordered = scores[order]

        labels = torch.cat(
            [torch.ones_like(self.scores_train), torch.zeros_like(self.scores_test)]
        )
        labels_ordered = labels[order]
        return labels_ordered, scores_ordered

    @static_method
    def _get_area_under_curve(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Computes the area under the parametric curve defined by (x, y).

        Notes:
            - x is assumed to be sorted in ascending order
            - y is not assumed to be monotonous
        """

        dx = x[1:] - x[:-1]
        dy = (y[1:] - y[:-1]).abs()
        result = (dx * y[:-1]).sum() + (dy * dx).sum()
        return result.item()

    def get_max_accuracy_threshold(self) -> Tuple[float, float]:
        """
        Computes the score threshold that allows for maximum accuracy of the attack.
        All samples below this threshold will be classified as train and all samples
        above as test.
        """

        labels_ordered, scores_ordered = self._get_scores_and_labels_ordered()

        cum_train_from_left = torch.cumsum(labels_ordered == 1, 0)
        cum_heldout_from_right = torch.cumsum(labels_ordered.flip(0) == 0, 0).flip(0)

        pad = torch.zeros(1, device=cum_train_from_left.device)
        cum_train_from_left = torch.cat((pad, cum_train_from_left[:-1]))

        n = labels_ordered.shape[0]
        accuracies = (cum_train_from_left + cum_heldout_from_right) / n

        max_accuracy_threshold = scores_ordered[accuracies.argmax()].item()
        max_accuracy = accuracies.max().item()

        return max_accuracy_threshold, max_accuracy

    def get_accuracy(self, threshold: float) -> float:
        """
        Given the maximum accuracy threshold, computes the accuracy of the attack.
        """

        n_samples = self.scores_train.shape[0] + self.scores_test.shape[0]
        n_true_positives = (self.scores_train > threshold).sum().float()
        n_true_negatives = (self.scores_test <= threshold).sum().float()

        accuracy = (n_true_positives + n_true_negatives) / n_samples
        return accuracy

    def get_precision_recall(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes precision and recall, useful for plotting PR curves and
        computing mAP.
        """

        labels_ordered, _ = self._get_scores_and_labels_ordered()

        true_positives = torch.cumsum(labels_ordered, 0)
        precision = true_positives / torch.arange(1, labels_ordered.shape[0] + 1)
        recall = true_positives / labels_ordered.sum()

        return precision, recall

    def get_map(self) -> float:
        """
        Computes the area under the PR curve.
        """
        precision, recall = self.get_precision_recall()
        result = AttackResults._get_area_under_curve(recall, precision)

        return result

    def get_tpr_fpr(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes true positive rate and true negative rate,, useful for plotting
        ROC curves and computing AUC.
        """
        labels_ordered, _ = self._get_scores_and_labels_ordered()

        true_positive_rate = (
            torch.cumsum(labels_ordered == 1, 0) / self.scores_train.shape[0]
        )
        false_positive_rate = (
            torch.cumsum(labels_ordered == 0, 0) / self.scores_test.shape[0]
        )
        return true_positive_rate, false_positive_rate

    def get_auc(self) -> float:
        """
        Computes the area under the ROC curve.
        """

        true_positive_rate, false_positive_rate = self.get_tpr_fpr()
        result = AttackResults._get_area_under_curve(false_positive_rate, true_positive_rate)
        return result
