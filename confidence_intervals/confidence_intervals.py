"""Main module.

Bootstrapping strategies taken from:
https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html?s=09
"""
from dataclasses import dataclass

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score


@dataclass
class ConfidenceResult:
    n_iter: int
    ci_level: int
    mean: np.float
    ci_lower: np.float
    ci_upper: np.float


def get_alpha(ci_level):
    return (100 - ci_level) / 200


def bootstrap_test(
    y_true: np.array,
    y_pred: np.array,
    n_iter=200,
    score_func=sklearn.metrics.accuracy_score,
    score_func_args={},
    ci_level: int = 95,
):
    """Compute confidence intervals using bootstrapping on the test set."""

    rng = np.random.RandomState(seed=12345)
    idx = np.arange(y_true.shape[0])

    test_perfs = list()

    for _ in range(n_iter):
        pred_idx = rng.choice(idx, size=idx.shape[0], replace=True)
        boot_perf = score_func(y_true[pred_idx], y_pred[pred_idx], **score_func_args)
        test_perfs.append(boot_perf)

    mean_perf = np.mean(test_perfs)

    alpha = get_alpha(ci_level)
    ci_lower = np.percentile(test_perfs, alpha * 100)
    ci_upper = np.percentile(test_perfs, (1 - alpha) * 100)

    return ConfidenceResult(
        n_iter=n_iter,
        ci_level=ci_level,
        mean=mean_perf,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )
