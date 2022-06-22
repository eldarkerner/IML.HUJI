import copy

import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from IMLearn.model_selection import cross_validate
from IMLearn.metrics import mean_square_error


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    # raise NotImplementedError()

    vals = []
    weights = []

    def innerCallback(descentModel, val, weight, **kwargs):
        # print("val:", val, "weights:", weight)
        vals.append(val)
        weights.append(weight)

    return innerCallback, vals, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    # raise NotImplementedError()
    for eta in etas:
        fixedLR = FixedLR(eta)
        l2 = L2(copy.deepcopy(init))
        callbackTupleL2 = get_gd_state_recorder_callback()
        gradientDescentL2 = GradientDescent(learning_rate=fixedLR, out_type="best", callback=callbackTupleL2[0])
        gradientDescentL2.fit(l2, None, None)
        print(min(callbackTupleL2[1]))

        plot_descent_path(L2, np.array(callbackTupleL2[2]), "L2 fixed, eta: " + str(eta)).show()

        plt.plot(range(1, len(callbackTupleL2[1]) + 1), callbackTupleL2[1])
        plt.title("L2, fixed learning rate, eta: " + str(eta))
        plt.xlabel("iterations")
        plt.ylabel("vals")
        # plt.savefig("L2 Eta " + str(eta) + ".png")
        plt.show()

        print("eta:", eta, "weights:", len(callbackTupleL2[2]))

    print("---------------------------------")
    for eta in etas:
        fixedLR = FixedLR(eta)
        l1 = L1(copy.deepcopy(init))
        callbackTupleL1 = get_gd_state_recorder_callback()
        gradientDescentL1 = GradientDescent(learning_rate=fixedLR, out_type="best", callback=callbackTupleL1[0])
        gradientDescentL1.fit(l1, None, None)
        print(min(callbackTupleL1[1]))

        plot_descent_path(L1, np.array(callbackTupleL1[2]), "L1 fixed, eta: " + str(eta)).show()

        plt.plot(range(1, len(callbackTupleL1[1]) + 1), callbackTupleL1[1])
        plt.title("L1, fixed learning rate, eta: " + str(eta))
        plt.xlabel("iterations")
        plt.ylabel("vals")
        # plt.savefig("L1 Eta " + str(eta) + ".png")
        plt.show()

        print("eta:", eta, "weights:", len(callbackTupleL1[2]))


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    # raise NotImplementedError()
    print("---------------------------------")

    for gamma in gammas:
        exponentialLR = ExponentialLR(eta, gamma)
        l1 = L1(copy.deepcopy(init))
        callbackTupleL1 = get_gd_state_recorder_callback()
        gradientDescentL1 = GradientDescent(learning_rate=exponentialLR, out_type="best", callback=callbackTupleL1[0])
        a = gradientDescentL1.fit(l1, None, None)
        print(min(callbackTupleL1[1]))

        plot_descent_path(L1, np.array(callbackTupleL1[2]), "L1 exponential, gamma: " + str(gamma)).show()

        plt.plot(range(1, len(callbackTupleL1[1]) + 1), callbackTupleL1[1])
        plt.title("L1, exponential learning rate, gamma: " + str(gamma))
        plt.xlabel("iterations")
        plt.ylabel("vals")
        # plt.savefig("L1 Gamma " + str(gamma) + ".png")
        plt.show()

        print("gamma:", gamma, "weights:", len(callbackTupleL1[2]))

        # Plot algorithm's convergence for the different values of gamma
        # raise NotImplementedError()

        # Plot descent path for gamma=0.95
        # raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    # raise NotImplementedError()
    gd = GradientDescent(callback=get_gd_state_recorder_callback()[0], learning_rate=FixedLR(1e-4), max_iter=20000)
    lr = LogisticRegression(solver=gd)
    lr.fit(X_train, y_train)
    probs = lr.predict_proba(X_train)

    fpr, tpr, thresholds = roc_curve(y_train, probs)

    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()

    alphaI = np.argmax(tpr - fpr)
    max_alpha = round(thresholds[alphaI], 2)

    lr = LogisticRegression(alpha=max_alpha, solver=gd)
    lr.fit(X_train, y_train)
    print("Loss:", lr.loss(X_test, y_test), "max alpha:", max_alpha)

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    # raise NotImplementedError()
    train_loss_l1 = []
    validation_loss_l1 = []

    train_loss_l2 = []
    validation_loss_l2 = []

    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

    for i in range(len(lambdas)):
        train_loss_k, validation_loss_k = cross_validate(LogisticRegression(penalty="l1", lam=lambdas[i], solver=gd), X_train, y_train, mean_square_error)
        train_loss_l1.append(train_loss_k)
        validation_loss_l1.append(validation_loss_k)

        train_loss_k, validation_loss_k = cross_validate(LogisticRegression(penalty="l2", lam=lambdas[i], solver=gd), X_train, y_train, mean_square_error)
        train_loss_l2.append(train_loss_k)
        validation_loss_l2.append(validation_loss_k)

    best_l1_index = np.argmin(np.array(validation_loss_l1))
    best_l2_index = np.argmin(np.array(validation_loss_l2))

    lrL1 = LogisticRegression(penalty="l1", lam=lambdas[best_l1_index], solver=gd)
    lrL1.fit(X_train, y_train)
    print("Loss L1:", lrL1.loss(X_test, y_test), "lambda:", lambdas[best_l1_index])

    lrL2 = LogisticRegression(penalty="l2", lam=lambdas[best_l2_index], solver=gd)
    lrL2.fit(X_train, y_train)
    print("Loss L2:", lrL2.loss(X_test, y_test), "lambda:", lambdas[best_l2_index])


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
