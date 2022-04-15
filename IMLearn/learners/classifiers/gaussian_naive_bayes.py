from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # raise NotImplementedError()
        self.classes_ = np.unique(y)

        self.mu_ = np.zeros((X.shape[1], len(self.classes_)))
        for index, class_i in enumerate(self.classes_):
            self.mu_[:, index] = X[y == class_i].mean(axis=0)

        self.vars_ = np.zeros((len(self.classes_), X.shape[1]))
        for index, class_i in enumerate(self.classes_):
            self.vars_[index] = np.var(X[y == class_i], axis=0)

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
        for k in range(n_classes):
            log_pi = np.log(self.pi_[k])

            for i in range(n_samples):
                x = X[i]

                # print(log_pi.shape, x.shape, ((x - self.mu_.T[k]) ** 2).shape, self.vars_[k].shape)
                # (x - self.mu_.T[k]) ** 2 - 2 * self.vars_[k]
                likelihoods[i, k] = log_pi - np.sum((x - self.mu_.T[k]) ** 2 + 2 * self.vars_[k])

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
        misclassification_error(y, self._predict(X))
