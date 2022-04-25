from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # raise NotImplementedError()
        self.classes_ = np.unique(y)

        self.n = np.zeros((len(self.classes_), 1))
        for index, class_i in enumerate(self.classes_):
            self.n[index] = np.count_nonzero(y == class_i)

        self.mu_ = np.zeros((X.shape[1], len(self.classes_)))
        for index, class_i in enumerate(self.classes_):
            self.mu_[:, index] = X[y == class_i].mean(axis=0)

        self.cov_ = np.zeros((X.shape[1], X.shape[1]))
        for index, class_i in enumerate(self.classes_):
            self.cov_ += (X[y == class_i] - self.mu_[:, class_i]).T @ (X[y == class_i] - self.mu_[:, class_i])
        self.cov_ /= y.size
        self._cov_inv = np.linalg.inv(self.cov_)

        self.pi_ = np.zeros((len(self.classes_), 1))
        for index, class_i in enumerate(self.classes_):
            self.pi_[index] = (y == class_i).mean()

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
        return np.argmax(self.likelihood(X), axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        # raise NotImplementedError()
        n_classes = len(self.classes_)
        n_samples = X.shape[0]

        likelihoods = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            x = X[i]
            for k in range(n_classes):
                a_k = self._cov_inv @ self.mu_.T[k]
                b_k = np.log(self.pi_[k]) - 0.5 * (self.mu_.T[k] @ self._cov_inv @ self.mu_.T[k])

                likelihoods[i, k] = a_k @ x + b_k

        return likelihoods

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
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
