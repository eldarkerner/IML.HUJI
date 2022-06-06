from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # raise NotImplementedError()
    avg_train_err = 0
    avg_validation_err = 0

    groups = np.remainder(np.arange(X.shape[0]), cv)
    for fold in range(cv):
        train_k_x = X[groups != fold]
        train_k_y = y[groups != fold]
        validate_k_x = X[groups == fold]
        validate_k_y = y[groups == fold]

        # train_k_x = train_k_x.reshape(-1,)
        # train_k_y = train_k_y.reshape(-1,)
        # validate_k_x = validate_k_x.reshape(-1,)
        # validate_k_y = validate_k_y.reshape(-1,)

        estimator.fit(train_k_x, train_k_y)
        avg_train_err += scoring(train_k_y, estimator.predict(train_k_x))
        avg_validation_err += scoring(validate_k_y, estimator.predict(validate_k_x))

    return avg_train_err / cv, avg_validation_err / cv
