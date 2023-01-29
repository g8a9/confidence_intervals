"""Main module.

Bootstrapping strategies taken from:
https://sebastianraschka.com/blog/2022/confidence-intervals-for-ml.html?s=09
"""
from dataclasses import dataclass
from typing import List

import numpy as np
import scipy
import sklearn
from beartype import beartype


@dataclass
class ConfidenceResult:
    ci_level: int
    mean: np.float
    ci_lower: np.float
    ci_upper: np.float
    n_iter: int = None


def get_alpha(ci_level: int):
    return (100 - ci_level) / 200


def check_ci_level(ci_level: int):
    if ci_level < 0 or ci_level > 100:
        raise ValueError("Confidence interval level must be between 0 and 100.")


@beartype
def bootstrap_test(
    y_true: np.array,
    y_pred: np.array,
    n_iter=200,
    score_func=sklearn.metrics.accuracy_score,
    score_func_args={},
    ci_level: int = 95,
):
    """Compute confidence intervals using bootstrapping on the test set.

    Args:
        y_true (ndarray): ground truth
        y_pred (ndarray): predictions
        n_iter (int): number of bootstrapping iterations. Default 200.
        score_func: any callable that takes as input the firt two parameters of this function and all named parameters specified, i.e., score_func(y_true, y_pred, **score_func_args). The callable must return a scalar value. Default sklearn.metrics.accuracy_score
        score_func_args (dict): additional named paramters used for the scoring function. Default dict()
        ci_level (int): confidence interval level, it must be between 0 and 100. Default 95

    Returns:
        ConfidenceResult: object containing all information about the run, including estimated confidence intervals).

    """
    check_ci_level(ci_level)

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


@beartype
def confidence_over_seeds(
    y_true: np.array,
    y_pred: List[np.array],
    score_func=sklearn.metrics.accuracy_score,
    score_func_args={},
    ci_level: int = 95,
):
    """Compute confidence intervals using the predictions of different seeds."""
    check_ci_level(ci_level)

    n_seeds = len(y_pred)

    # Compute scores for each seed
    test_perfs = [score_func(y_true, yp, **score_func_args) for yp in y_pred]

    confidence = ci_level / 100
    t_value = scipy.stats.t.ppf((1 + confidence) / 2.0, df=n_seeds - 1)

    sd = np.std(test_perfs, ddof=1)
    se = sd / np.sqrt(n_seeds)

    ci_length = t_value * se
    mean_perf = np.mean(test_perfs)

    ci_lower = mean_perf - ci_length
    ci_upper = mean_perf + ci_length

    return ConfidenceResult(
        ci_level=ci_level,
        mean=mean_perf,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )
