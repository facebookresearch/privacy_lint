import re
import inspect
import json
import itertools
from torch import optim
import numpy as np
from logging import getLogger
from opacus import PrivacyEngine
import opacus.privacy_analysis as privacy_analysis

logger = getLogger()

def repeat(l, r):
    """
    Repeat r times each value of list l.
    """
    return list(itertools.chain.from_iterable(itertools.repeat(x, r) for x in l))


def repeat_to(l, r):
    """
    Repeat values in list l so that it has r values
    """
    assert r % len(l) == 0

    return repeat(l, r // len(l))


def get_optimizer(parameters, opt_config, epochs):
    """
    Parse optimizer parameters.
    opt_config should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """

    lr_schedule = None
    if "," in opt_config:
        method = opt_config[:opt_config.find(',')]
        optim_params = {}
        for x in opt_config[opt_config.find(',') + 1:].split(','):
            # e.g. split = ('lr', '0.1-0.01) or split = ('weight_decay', 0.001)
            split = x.split('=')
            assert len(split) == 2
            param_name, param_value = split
            assert any([
                re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)$", param_value) is not None,
                param_name == "lr" and re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)$", param_value) is not None,
                param_name == "lr" and ("-" in param_value),
                param_name == "lr" and re.match(r"^cos:[+-]?(\d+(\.\d*)?|\.\d+)$", param_value) is not None
            ])
            if param_name == "lr":
                if param_value.startswith("cos:"):
                    lr_init = float(param_value[4:])
                    lr_schedule = [lr_init * (1 + np.cos(np.pi * epoch / epochs)) / 2 for epoch in range(epochs)]
                else:
                    lr_schedule = [float(lr) for lr in param_value.split("-")]
                optim_params[param_name] = float(lr_schedule[0])
                lr_schedule = repeat_to(lr_schedule, epochs)
            else:
                optim_params[param_name] = float(param_value)
    else:
        method = opt_config
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
        optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    logger.info("Schedule of %s: %s" % (opt_config, str(lr_schedule)))

    return optim_fn(parameters, **optim_params), lr_schedule


PRIVACY_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

def getNoiseMultiplier(epsilon, delta, q, steps):
    sigma_min, sigma_max = 0.01, 10
    
    while sigma_max - sigma_min > 0.01:
        sigma = (sigma_min + sigma_max) / 2
        rdp = privacy_analysis.compute_rdp(q, sigma, steps, PRIVACY_ALPHAS)
        eps = privacy_analysis.get_privacy_spent(PRIVACY_ALPHAS, rdp, delta)[0]

        if eps < epsilon:
            sigma_max = sigma
        else:
            sigma_min = sigma

    logger.info(f"Inferred σ={sigma} for ε={epsilon}, δ={delta}")
    logger.info("__log__:%s" % json.dumps({"noise_multiplier": sigma}))

    return sigma


def create_privacy_engine(model, params, n_data):
    if params.private:
        if params.noise_multiplier is None:
            _n_data = n_data# if params.privacy_fake_samples is None else params.privacy_fake_samples
            sample_rate = params.batch_size / _n_data
            steps = params.epochs * _n_data / params.batch_size
            params.noise_multiplier = getNoiseMultiplier(params.privacy_epsilon, params.privacy_delta, sample_rate, steps)

        if params.max_grad_norm == "mean":
            max_grad_norm = 1.0
        else:
            max_grad_norm = params.max_grad_norm

    else:
        max_grad_norm = float("inf")
        params.noise_multiplier = 0

    if params.private or params.log_gradients:
        if params.log_gradients and not params.private:
            logger.info("Creating privacy engine to compute per sample gradients and log them.")
        privacy_engine = PrivacyEngine(
            model,
            batch_size=params.batch_size,
            sample_size=n_data,
            alphas=PRIVACY_ALPHAS,
            noise_multiplier=params.noise_multiplier,
            max_grad_norm=max_grad_norm
        )
    else:
        privacy_engine = None

    return privacy_engine