import numpy as np
from sklearn import metrics


def time_to_first_error(y_true, y_pred):
    nonzero_idx, = np.nonzero(~y_true[np.argsort(-y_pred)])
    first_fp = nonzero_idx.min()
    return first_fp


def min_exposed_dp(y_true, y_pred, delta=1e-5):
    # TODO: take into account delta to get (epsilon, delta) guarantee
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    eps = (np.log(tpr + 1e-8) - np.log(fpr + 1e-8)).max()
    return eps