from __future__ import annotations
from typing import NoReturn

from sklearn import linear_model
from IMLearn import BaseEstimator

import IMLearn.metrics
# from ...base import BaseEstimator
import numpy as np
from numpy.linalg import pinv


class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        # raise NotImplementedError()

        # number_of_features = X.shape[1]
        # self.coefs_ = np.zeros(number_of_features + 1)
        # self.coefs_ = np.linalg.pinv(X) @ y

        # if not self.include_intercept_:
        #     self.coefs_[0] = 0

        if self.include_intercept_:
            inter = np.ones((X.shape[0], 1))
            X = np.concatenate((inter, X), axis=1)

        U, Sigma, transpose_V = np.linalg.svd(X, full_matrices=False)
        inv_sigma = np.linalg.pinv(np.diag(Sigma))
        self.coefs_ = np.transpose(transpose_V) @ inv_sigma @ np.transpose(U) @ y

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
        if self.include_intercept_:
            inter_x = np.ones((X.shape[0], 1))
            X = np.concatenate((inter_x, X), axis=1)
        return X @ self.coefs_

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


if __name__ == '__main__':
    np.random.seed(0)
    X = 5 * np.random.random_sample([3, 3])
    y = 5 * np.random.random_sample([3, 1])
    y = y.reshape([-1, 1])
    pred = 5 * np.random.random_sample([3, 3])

    reg = linear_model.LinearRegression()
    reg.fit(X, y)

    print('\nsklearn LR Corfficients: \n', reg.coef_)
    # print('\nsklearn LR variance score: {}'.format(reg.score(x_test,y_test)))

    my_reg = LinearRegression()
    my_reg.fit(X, y)
    print('\n MY Corfficients: \n', my_reg.coefs_)
    print('\n MY reg: \n', my_reg.predict(pred))
    print('\n SKlern reg: \n', reg.predict(pred))