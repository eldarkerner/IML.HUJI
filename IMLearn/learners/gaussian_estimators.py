from __future__ import annotations

import math

import numpy
import numpy as np
from numpy.linalg import inv, det, slogdet

import IMLearn.learners


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=True
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        # raise NotImplementedError()

        self.mu_ = np.mean(X)
        if self.biased_:
            self.var_ = np.var(X) * (len(X) - 1) / len(X)
        else:
            self.var_ = np.var(X)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        # raise NotImplementedError()
        pdfs = IMLearn.learners.UnivariateGaussian.calc_pdf(self.mu_, self.var_, X, lambda q: q)
        return pdfs

    @staticmethod
    def calc_pdf(mu: float, var: float, X: np.ndarray, func) -> np.ndarray:
        func_pdfs = np.zeros(len(X))
        for index, x in enumerate(X):
            prefix = 1 / (2 * math.pi * var) ** 0.5
            exponent = math.exp(-(((x - mu) ** 2) / (2 * var)))

            func_pdfs[index] = func(prefix * exponent)

        return func_pdfs

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        # raise NotImplementedError()

        log_pdfs = IMLearn.learners.UnivariateGaussian.calc_pdf(mu, sigma ** 2, X, lambda q: np.log(q))
        return log_pdfs.sum()


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.ft`
            function.

        cov_: float
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.ft`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        # raise NotImplementedError()
        self.mu_ = np.mean(X, axis=0)
        self.cov_ = np.cov(np.transpose(X))

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        # raise NotImplementedError()
        return IMLearn.learners.MultivariateGaussian.calc_pdf(self.mu_, self.cov_, X, lambda q: q)

    @staticmethod
    def calc_pdf(mu: np.ndarray, cov: np.ndarray, X: np.ndarray, func):
        func_pdfs = np.zeros(X.shape)
        for index, x in enumerate(X):
            two_pi_powered = (2 * math.pi) ** len(mu)
            prefix = 1 / (two_pi_powered * np.linalg.det(cov)) ** 0.5
            exponent = np.exp(-0.5 * ((np.transpose(x - mu)).dot(np.linalg.inv(cov)).dot(x - mu)))

            func_pdfs[index] = func(prefix * exponent)

        return func_pdfs

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        # raise NotImplementedError()
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)

        k = np.shape(X)[1]
        num_samples = np.shape(X)[0]

        log_likelihood_sum = 0
        for index, x in enumerate(X):
            log_likelihood_sum += (x - mu) @ inv_cov @ (x - mu).T

        return -0.5 * (num_samples * np.log(det_cov) + log_likelihood_sum + k * num_samples * np.log(2 * np.pi))
        # return IMLearn.learners.MultivariateGaussian.calc_pdf(mu, cov, X, lambda q: np.log(q)).sum()
