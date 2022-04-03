from __future__ import annotations
from typing import NoReturn
from . import LinearRegression
from ...base import BaseEstimator
import numpy as np
import IMLearn.metrics


class PolynomialFitting(BaseEstimator):
    """
    Polynomial Fitting using Least Squares estimation
    """
    def __init__(self, k: int) -> PolynomialFitting:
        """
        Instantiate a polynomial fitting estimator

        Parameters
        ----------
        k : int
            Degree of polynomial to fit
        """
        super().__init__()
        # raise NotImplementedError()
        self.k = k
        self.linear_regressor = LinearRegression()
        self.coefs = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to polynomial transformed samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # raise NotImplementedError()

        newX = np.vander(X)
        self.linear_regressor.fit(newX, y)
        self.coefs = self.linear_regressor.coefs_

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # raise NotImplementedError()

        return np.vander(X) @ self.coefs

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        # raise NotImplementedError()
        y_pred = self._predict(X)
        return IMLearn.metrics.mean_square_error(y, y_pred)

    def __transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform given input according to the univariate polynomial transformation

        Parameters
        ----------
        X: ndarray of shape (n_samples,)

        Returns
        -------
        transformed: ndarray of shape (n_samples, k+1)
            Vandermonde matrix of given samples up to degree k
        """

        return np.vander(X)
        # raise NotImplementedError()
