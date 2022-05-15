import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    # raise NotImplementedError()
    adaB = AdaBoost(DecisionStump, n_learners)
    adaB.fit(train_X, train_y)

    loss_train = np.zeros(n_learners)
    loss_test = np.zeros(n_learners)
    for i in range(1, n_learners + 1):
        loss_train[i - 1] = adaB.partial_loss(train_X, train_y, i)
        loss_test[i - 1] = adaB.partial_loss(test_X, test_y, i)

    plt.plot(range(n_learners), loss_train, label="train")
    plt.plot(range(n_learners), loss_test, label="test")
    plt.xlabel("number of learners")
    plt.ylabel("loss")
    plt.title("train and test of AdaBoost with noise: " + str(noise))
    plt.legend()

    plt.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    # raise NotImplementedError()

    symbols = np.array(["circle", "x"])

    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{t} iterations}}$" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces([decision_surface(lambda x: adaB.partial_predict(x, t), lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y.astype(int), symbol=symbols[test_y.astype(int)],
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(title=rf"$\textbf{{Decision Surfaces Of Different Amount Of Iterations}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)

    fig.show()
    # Question 3: Decision surface of best performing ensemble
    # raise NotImplementedError()

    min_loss, min_idx = np.inf, None
    for idx, loss_i in enumerate(loss_test):
        if loss_i < min_loss:
            min_loss = loss_i
            min_idx = idx + 1

    go.Figure(data=[decision_surface(lambda x: adaB.partial_predict(x, min_idx + 1), lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y.astype(int), symbol=symbols[test_y.astype(int)],
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
              layout=go.Layout(title=rf"$\textbf{{The Maximal accuracy is {1 - min_loss}, and it gets with {min_idx} ensembles}}$")).show()

    # Question 4: Decision surface with weighted samples
    # raise NotImplementedError()

    D = adaB.D_ / max(adaB.D_) * 5
    go.Figure(data=[decision_surface(adaB.predict, lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y.astype(int), symbol=symbols[test_y.astype(int)],
                                               size=D, colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
              layout=go.Layout(title=rf"$\textbf{{Decision Surfaces Of Different Weights - Training Set}}$")).show()


if __name__ == '__main__':
    np.random.seed(0)
    # raise NotImplementedError()
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)

