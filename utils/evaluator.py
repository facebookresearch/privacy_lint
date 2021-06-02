# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch


logger = getLogger()


class Evaluator(object):
    # TODO: get ridd of params and only give model in eval()

    def __init__(self, model, params):
        """
        Initialize evaluator.
        """
        self.model = model
        self.params = params

    @torch.no_grad()
    def run_all_evals(self, evals, data_loader, *args, **kwargs):
        """
        Run all evaluations.
        """
        assert type(evals) is list
        scores = OrderedDict()

        if evals is None or 'classif' in evals:
            self.eval_classif(scores, data_loader)

        return scores


    def eval_classif(self, scores, data_loader):
        """
        Evaluate classification.
        """
        params = self.params
        self.model.eval()

        # stats
        accuracies = []
        topk = [1, 5, 10, 20, 50, 100, 200, 500]
        topk = [k for k in topk if k <= params.num_classes]

        for _, images, targets in data_loader:
            images = images.cuda()
            output = self.model(images)
            accuracies.append(accuracy(output.cpu(), targets, topk=tuple(topk)))

        # accuracy
        for i_k, k in enumerate(topk):
            scores['valid_top%d_acc' % k] = np.mean([x[i_k] for x in accuracies])



def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res
