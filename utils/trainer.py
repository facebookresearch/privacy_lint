from collections import OrderedDict
import functools
import os
import time

import numpy as np
import torch
from torch.nn import functional as F

from logging import getLogger
from .optimizer import get_optimizer, create_privacy_engine

logger = getLogger()

def log_grad(trainer, param_name, *args, **kwargs):
    if param_name is not None:
        g = kwargs['per_sample_grad']
        trainer.current_grad_sample.append(g.view(g.size(0), -1).clone())
    else:
        trainer.current_grad_sample = torch.cat(trainer.current_grad_sample, dim=1)


class Trainer:
    def __init__(self, model, params, n_data=-1):
        # model / params
        self.model = model
        self.params = params

        # set optimizers
        self.n_data = n_data
        if params.private and params.privacy_delta is None:
            params.privacy_delta = 1 / n_data
            print(f"Setting privacy delta to {params.privacy_delta}")

        self.privacy_engine = create_privacy_engine(model, params, n_data=n_data)
        self.optimizer, self.schedule = get_optimizer(model.parameters(), params.optimizer, params.epochs)

        if self.privacy_engine is not None: 
            self.privacy_engine.attach(self.optimizer)
            if params.log_gradients:
                self.privacy_engine.clipper.set_on_batch_clip_func(functools.partial(log_grad, self))
            self.current_grad_sample = []
            self.all_grad_samples = None

        # training statistics
        self.epoch = 0
        self.indices = []
        self.n_iter = 0
        self.step = 0
        self.stats = OrderedDict(
            [('processed_i', 0)] +
            [('XE', [])] +
            [('time', [])]
        )

        self.last_time = time.time()


    def update_learning_rate(self):
        """
        Sets the learning rate to follow the learning schedule
        """
        if self.schedule is None:
            return
        lr = self.schedule[self.epoch]
        logger.info("New learning rate for %f" % lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


    def end_step(self):
        self.n_iter += 1
        self.step += 1
        self.print_stats()

    def print_stats(self):
        """
        Prints statistics about the training.
        Statistics are computed on batches since the last print.
        (i.e. if printing every 5 batches then it shows speed on the last 5 batches)
        """
        if self.n_iter % self.params.print_freq != 0:
            return

        s_iter = f"Batch {self.n_iter} - "
        s_stat = ''
        s_stat += ' || '.join([
            '{}: {:7.4f}'.format(k, np.mean(v[-self.params.print_freq:])) for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])

        # learning rates
        s_lr = ""
        s_lr = s_lr + (" - LR: ") + " / ".join("{:.4e}".format(group['lr']) for group in self.optimizer.param_groups)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} images/s - ".format(self.stats['processed_i'] * 1.0 / diff)
        self.stats['processed_i'] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)
    
    
    def save(self, name):
        """
        Save the model.
        """

        path = os.path.join(self.params.dump_path, name)
        state_dict = self.state_dict()
        logger.info("Saving model to %s ..." % path)
        torch.save(state_dict, path)

    def state_dict(self):
        r"""
        Returns state_dict, i.e. model parameters as well as general parameters
        """
        model = self.model
        data = {
            'model': model.state_dict(),
            'epoch': self.epoch,
            'params': vars(self.params)
        }
        data['optimizer'] = self.optimizer.state_dict()
        if self.params.private:
            data['privacy_engine'] = self.privacy_engine.state_dict()
        if self.params.log_gradients:
            data['gradients'] = self.all_grad_samples

        return data


    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path = os.path.join(self.params.dump_path, "checkpoint.pth")
        if not os.path.isfile(checkpoint_path):
            return
        logger.warning('Reloading checkpoint from %s ...' % checkpoint_path)
        state_dict = torch.load(checkpoint_path)

        self.model.load_state_dict(state_dict['model'])

        if self.params.private:
            self.privacy_engine.load_state_dict(state_dict['privacy_engine'])

        # reload optimizer
        self.optimizer.load_state_dict(state_dict['optimizer'])

        # reload stats
        self.epoch = state_dict['epoch'] + 1
        logger.warning('Checkpoint reloaded. Resuming at epoch %i ...' % self.epoch)


    def end_epoch(self, scores):
        # Update learning rate
        self.update_learning_rate()

        # Reset statistics
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]
        self.epoch += 1
       
        # Save checkpoints
        self.save("checkpoint.pth")
        if self.params.save_periodic > 0 and self.epoch % self.params.save_periodic == 0:
            self.save("periodic-%d.pth" % self.epoch)
        self.all_grad_samples = None

    def maybe_log_gradients(self, idx):
        # Log per sample gradient
        if self.params.log_gradients:
            if self.all_grad_samples is None:
                self.all_grad_samples = torch.zeros(self.n_data, self.current_grad_sample.size(1), dtype=self.current_grad_sample.dtype, device=torch.device('cpu'))
            self.all_grad_samples[idx] = self.current_grad_sample.cpu()
            self.current_grad_sample = []


    def lm_step(self, idx, sentence):
        """
        Language modeling step.
        """
        start = time.time()
        self.model.train()
        sentence = sentence.cuda(non_blocking=True)

        # Forward + loss
        output = self.model(sentence[:, 1:])
        loss = F.cross_entropy(output.view(-1, output.size(-1)), sentence[:, :-1].reshape(-1), reduction='mean')

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.maybe_log_gradients(idx)

        # statistics
        self.stats['processed_i'] += self.params.batch_size
        self.stats['XE'].append(loss.item())
        self.stats['time'].append(time.time() - start)


    def classif_step(self, idx, images, targets):
        """
        Classification step.
        """
        start = time.time()
        self.model.train()
        images = images.cuda(non_blocking=True)

        # Forward + loss
        output = self.model(images)
        loss = F.cross_entropy(output, targets.cuda(non_blocking=True), reduction='mean')

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.maybe_log_gradients(idx)

        # statistics
        self.stats['processed_i'] += self.params.batch_size
        self.stats['XE'].append(loss.item())
        self.stats['time'].append(time.time() - start)


    def get_scores(self):
        scores = {
            "speed": self.params.batch_size / np.mean(self.stats['time']),
            "learning_rate": self.schedule[self.epoch]
        }
        if self.params.private:
            scores["privacy_epsilon"] = self.privacy_engine.get_privacy_spent(1 / self.n_data)[0]

        for stat_name in self.stats.keys():
            if type(self.stats[stat_name]) is list and len(self.stats[stat_name]) >= 1:
                scores[stat_name] = np.mean(self.stats[stat_name])

        return scores
