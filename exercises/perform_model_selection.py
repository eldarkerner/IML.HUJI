from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    # raise NotImplementedError()
    min_val = -1.2
    max_val = 2

    X = np.linspace(min_val, max_val, n_samples)
    y = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2) + np.random.normal(0, noise, size=n_samples)

    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), 2 / 3)
    train_x, train_y, test_x, test_y = np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

    train_x = train_x.reshape(-1, )
    train_y = train_y.reshape(-1, )
    test_x = test_x.reshape(-1, )
    test_y = test_y.reshape(-1, )

    plt.plot(train_x, train_y, '*', label="train")
    plt.plot(test_x, test_y, '*', label="test")
    plt.legend()
    plt.title("the data")

    plt.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    # raise NotImplementedError()
    max_degree = 10
    train_loss = []
    validation_loss = []
    for k in range(1, max_degree + 1):
        train_loss_k, validation_loss_k = cross_validate(PolynomialFitting(k), train_x, train_y, mean_square_error)
        train_loss.append(train_loss_k)
        validation_loss.append(validation_loss_k)

    plt.plot(np.arange(max_degree) + 1, train_loss, '-*', label="train")
    plt.plot(np.arange(max_degree) + 1, validation_loss, '-*', label="validation")
    plt.legend()
    plt.title("cross validation - polynomial fitting with #samples " + str(n_samples) + ", and " + str(noise) + " noise")

    plt.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    # raise NotImplementedError()
    best_deg_index = np.argmin(np.array(validation_loss))
    best_deg = best_deg_index + 1
    poly = PolynomialFitting(best_deg)

    poly.fit(train_x, train_y)
    test_loss = poly.loss(test_x, test_y)

    print('The test loss on best validation ' + str(best_deg) + ' is: ', round(test_loss, 2))
    print('The validation loss on best validation k is: ', round(validation_loss[best_deg_index], 2))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    # raise NotImplementedError()
    X, y = datasets.load_diabetes(return_X_y=True)
    train_x = X[:n_samples, :]
    train_y = y[:n_samples]
    test_x = X[n_samples:, :]
    test_y = y[n_samples:]

    train_x, train_y, test_x, test_y = np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    # raise NotImplementedError()
    min_val_lasso = 0.0001
    max_val_lasso = 2.5
    lambdas_lasso = np.linspace(min_val_lasso, max_val_lasso, n_evaluations)

    min_val_ridge = 0.0001
    max_val_ridge = 2.5
    lambdas_ridge = np.linspace(min_val_ridge, max_val_ridge, n_evaluations)

    train_loss_lasso = []
    validation_loss_lasso = []

    train_loss_ridge = []
    validation_loss_ridge = []

    for i in range(n_evaluations):
        train_loss_k, validation_loss_k = cross_validate(Lasso(lambdas_ridge[i]), train_x, train_y, mean_square_error)
        train_loss_lasso.append(train_loss_k)
        validation_loss_lasso.append(validation_loss_k)

        train_loss_k, validation_loss_k = cross_validate(RidgeRegression(lambdas_ridge[i]), train_x, train_y, mean_square_error)
        train_loss_ridge.append(train_loss_k)
        validation_loss_ridge.append(validation_loss_k)

    plt.plot(lambdas_lasso, train_loss_lasso, '-*', label="lasso train loss")
    plt.plot(lambdas_lasso, validation_loss_lasso, '-*', label="lasso validation loss")
    plt.legend()
    plt.title("lasso loss with #" + str(n_evaluations) + " in the spectrum: [" + str(min_val_lasso) + ", " + str(max_val_lasso) + "]")
    plt.show()

    plt.plot(lambdas_ridge, train_loss_ridge, '-*', label="ridge train loss")
    plt.plot(lambdas_ridge, validation_loss_ridge, '-*', label="ridge validation loss")
    plt.legend()
    plt.title("ridge loss with #" + str(n_evaluations) + " in the spectrum: [" + str(min_val_ridge) + ", " + str(max_val_ridge) + "]")
    plt.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    # raise NotImplementedError()
    best_lasso_index = np.argmin(np.array(validation_loss_lasso))
    best_ridge_index = np.argmin(np.array(validation_loss_ridge))

    lasso = Lasso(lambdas_lasso[best_lasso_index])
    lasso.fit(train_x, train_y)
    test_loss_lasso = mean_square_error(test_y, lasso.predict(test_x))

    ridge = RidgeRegression(lambdas_ridge[best_ridge_index])
    ridge.fit(train_x, train_y)
    test_loss_ridge = ridge.loss(test_x, test_y)

    print('The test loss on lasso: ' + str(lambdas_lasso[best_lasso_index]) + ' is: ', round(test_loss_lasso, 2))
    print('The test loss on ridge: ' + str(lambdas_ridge[best_ridge_index]) + ' is: ', round(test_loss_ridge, 2))

    linear = LinearRegression()
    linear.fit(train_x, train_y)
    test_loss_linear = linear.loss(test_x, test_y)
    print('The test loss on least squares is: ', round(test_loss_linear, 2))


if __name__ == '__main__':
    np.random.seed(0)
    # raise NotImplementedError()
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(1500, 10)
    select_regularization_parameter()
