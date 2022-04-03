from __future__ import annotations
from typing import NoReturn

import sklearn.metrics

from IMLearn.base import BaseEstimator
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, precision_score, recall_score


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------

        Attributes
        ----------

        """
        super().__init__()
        self.model = linear_model.LogisticRegression(max_iter=300)
        self.coefs = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """
        self.model.fit(X, y)
        self.coefs = self.model.coef_

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
        return self.model.predict(X)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """

        print(y)
        prediction = self.predict(X)
        print("logistic loss - precision:", precision_score(y, prediction))
        print("logistic loss - recall:", recall_score(y, prediction))
        return mean_squared_error(y, prediction)

        # tot_loss = 0
        # print(X)
        # X = np.array(X)
        # for index, row in enumerate(X):
        #     print(row)
        #     tot_loss += np.power(self.predict(np.ndarray(row)) - y[index], 2)
        #
        # return tot_loss
