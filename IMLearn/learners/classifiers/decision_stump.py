from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def __predict(self, values, threshold, sign):
        return sign * ((values >= threshold) * 2 - 1)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # raise NotImplementedError()
        # min_loss = None
        #
        # for i in range(X.shape[1]):
        #     threshold_i_plus, loss_i_plus = self._find_threshold(X[:, i], y, 1)
        #     threshold_i_minus, loss_i_minus = self._find_threshold(X[:, i], y, -1)
        #
        #     if min_loss is None or loss_i_plus < min_loss:
        #         min_loss = loss_i_plus
        #         self.j_ = i
        #         self.threshold_ = threshold_i_plus
        #         self.sign_ = 1
        #
        #     if min_loss is None or loss_i_minus < min_loss:
        #         min_loss = loss_i_minus
        #         self.j_ = i
        #         self.threshold_ = threshold_i_minus
        #         self.sign_ = -1

        loss_star, threshold = np.inf, None
        for sign, j in product([-1, 1], range(X.shape[1])):
            threshold_i, loss = self._find_threshold(X[:, j], y, sign)
            if loss < loss_star:
                self.sign_, self.threshold_, self.j_ = sign, threshold_i, j
                loss_star = loss

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        # raise NotImplementedError()
        return self.__predict(X[:, self.j_], self.threshold_, self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        D = np.abs(labels)
        D /= np.sum(D)
        sort_idx = np.argsort(values)
        values, labels, D = values[sort_idx], np.sign(labels[sort_idx]), D[sort_idx]
        thresholds = np.concatenate([[-np.inf], (values[1:] + values[:-1]) / 2, [np.inf]])
        minimal_threshold_loss = np.sum(D[labels == sign])
        losses = np.append(minimal_threshold_loss, minimal_threshold_loss - np.cumsum(D * labels * sign))
        min_loss_idx = np.argmin(losses)
        return thresholds[min_loss_idx], losses[min_loss_idx]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        # raise NotImplementedError()
        from IMLearn.metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
