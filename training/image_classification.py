# -*- coding: utf-8 -*-
import argparse
import json
import os

import torch

from models import build_model
from datasets import get_dataset
from utils.evaluator import Evaluator
from utils.logger import create_logger
from utils.misc import bool_flag
from utils.trainer import Trainer


def check_parameters(params):
    assert params.dump_path is not None
    os.makedirs(params.dump_path, exist_ok=True)


def get_parser():
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser(description='Train/evaluate image classification models')

    # config parameters
    parser.add_argument("--dump_path", type=str, default=None)
    parser.add_argument('--print_freq', type=int, default=5)
    parser.add_argument("--save_periodic", type=int, default=0)

    # Data parameters
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--dataset", type=str, choices=["cifar10"], default="cifar10")
    parser.add_argument("--mask_path", type=str, required=True)

    # Model parameters
    parser.add_argument("--architecture", choices=["lenet", "smallnet"], default="lenet")

    # training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--optimizer", default="sgd,lr=0.001,momentum=0.9")
    parser.add_argument("--num_workers", type=int, default=2)

    # privacy parameters
    parser.add_argument("--private", type=bool_flag, default=False)
    parser.add_argument("--noise_multiplier", type=float, default=None)
    parser.add_argument("--privacy_epsilon", type=float, default=None)
    parser.add_argument("--privacy_delta", type=float, default=None)
    parser.add_argument("--log_gradients", type=bool_flag, default=False)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    return parser


def train(params, mask):
    # Create logger and print params
    logger = create_logger(params)

    trainloader, n_data = get_dataset(params=params, is_train=True, mask=mask)
    validloader, _ = get_dataset(params=params, is_train=False)

    model = build_model(params)
    model.cuda()

    trainer = Trainer(model, params, n_data=n_data)
    trainer.reload_checkpoint()

    evaluator = Evaluator(model, params)

    # evaluation
    # if params.eval_only:
    #     scores = evaluator.run_all_evals(trainer, evals=['classif'], data_loader=validloader)

    #     for k, v in scores.items():
    #         logger.info('%s -> %.6f' % (k, v))
    #     logger.info("__log__:%s" % json.dumps(scores))
    #     exit()


    # training
    for epoch in range(trainer.epoch, params.epochs):

        # update epoch / sampler / learning rate
        trainer.epoch = epoch
        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        # train
        for (idx, images, targets) in trainloader:
            trainer.classif_step(idx, images, targets)
            trainer.end_step()

        logger.info("============ End of epoch %i ============" % trainer.epoch)

        # evaluate classification accuracy
        scores = evaluator.run_all_evals(evals=['classif'], data_loader=validloader)
        for name, val in trainer.get_scores().items():
            scores[name] = val

        # print / JSON log
        for k, v in scores.items():
            logger.info('%s -> %.6f' % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))

        # end of epoch
        trainer.end_epoch(scores)

    return model



if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    check_parameters(params)

    mask = torch.load(params.mask_path)
    train(params, mask)
