# -*- coding: utf-8 -*-
import argparse
import json
import os

from models import build_model
from datasets import get_dataset
from utils.trainer import Trainer
from utils.logger import create_logger
from utils.misc import bool_flag


def check_parameters(params):
    if params.private:
        assert params.privacy_epsilon is not None
    
    assert params.dump_path is not None
    os.makedirs(params.dump_path, exist_ok=True)


def get_parser():
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser(description='Train/evaluate a language model')

    # Config parameters
    parser.add_argument("--dump_path", type=str, default=None)
    parser.add_argument('--print_freq', type=int, default=5)
    parser.add_argument("--save_periodic", type=int, default=0)

    # Data parameters
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--dataset", choices=["dummy"], default='dummy')
    parser.add_argument("--n_vocab", type=int, default=256)

    # Model parameters
    parser.add_argument("--architecture", type=str, default='lstm')
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--optimizer", default="sgd,lr=0.001,momentum=0.9")
    parser.add_argument("--seq_len", type=int, default=256)

    # Privacy parameters
    parser.add_argument("--private", type=bool_flag, default=False)
    parser.add_argument("--noise_multiplier", type=float, default=None)
    parser.add_argument("--privacy_epsilon", type=float, default=None)
    parser.add_argument("--privacy_delta", type=float, default=None)
    parser.add_argument("--privacy_fake_samples", type=int, default=None)
    parser.add_argument("--log_gradients", type=bool_flag, default=False)

    return parser


def main(params):
    # Create logger and print params (very useful for debugging)
    logger = create_logger(params)

    trainloader, n_data = get_dataset(params, split='train', is_train=True)
    validloader, _ = get_dataset(params, split='valid', is_train=False)

    model = build_model(params)
    model.cuda()

    trainer = Trainer(model, params, n_data=n_data)
    trainer.reload_checkpoint()

    # evaluator = Evaluator(trainer, params)

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
        for (idx, sentence) in trainloader:
            trainer.lm_step(idx, sentence)
            trainer.end_step()

        logger.info("============ End of epoch %i ============" % trainer.epoch)

        # evaluate classification accuracy
        # scores = evaluator.run_all_evals(trainer, evals=['classif'], data_loader=validloader)

        scores = {}
        for name, val in trainer.get_scores().items():
            scores[name] = val

        # print / JSON log
        for k, v in scores.items():
            logger.info('%s -> %.6f' % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))

        # end of epoch
        trainer.end_epoch(scores)



if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    check_parameters(params)

    main(params)
