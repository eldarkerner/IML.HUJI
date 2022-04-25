import matplotlib.pyplot as plt
import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():

    def print_loss_iter(fit: Perceptron, sample: np.ndarray, label: int):
        perceptron.fitted_ = True
        losses.append(perceptron.loss(x, y))
        perceptron.fitted_ = False

    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        # raise NotImplementedError()

        data = np.load("C:/Users/user/IML.HUJI/datasets/" + f)

        x, y = [], []   # np.zeros((len(data), 2)), np.zeros((len(data), 1))

        # for index, row in enumerate(data):
        #     x[index] = row[:2]
        #     y[index] = row[2]

        for row in data:
            x.append(row[:2])
            y.append(row[2])

        # Fit Perceptron and record loss in each fit iteration
        # raise NotImplementedError()
        losses = []
        perceptron = Perceptron(callback=print_loss_iter)

        x = np.array(x)
        y = np.array(y)

        perceptron.fit(x, y)

        # Plot figure of loss as function of fitting iteration
        # raise NotImplementedError()
        plt.plot(range(len(losses)), losses)
        plt.title(n + " loss as a function of number of fitting iteration")
        plt.xlabel("number of iteration")
        plt.ylabel("loss")
        plt.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        # raise NotImplementedError()
        x, y = load_dataset("C:/Users/user/IML.HUJI/datasets/" + f)

        # Fit models and predict over training set
        # raise NotImplementedError()
        lda = LDA()
        gnb = GaussianNaiveBayes()     # TODO: change to gnb

        lda.fit(x, y)
        gnb.fit(x, y)

        pred_lda = lda.predict(x)
        pred_gnb = gnb.predict(x)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        # raise NotImplementedError()
        from IMLearn.metrics import accuracy

        sub_fig1 = go.Scatter(x=x[:, 0], y=x[:, 1],
                              marker=dict(symbol=np.uint32(y).tolist(), color=np.uint32(pred_lda)),
                              mode='markers',
                              text=['True class: ' + str(y[i]) + ' Predicted class: ' + str(pred_lda[i])
                                    for i in range(x.shape[0])],
                              hovertemplate='%{text}')

        sub_fig2 = go.Scatter(x=x[:, 0], y=x[:, 1],
                              marker=dict(symbol=np.uint32(y).tolist(), color=np.uint32(pred_gnb)),
                              mode='markers',
                              text=['True class: ' + str(y[i]) + ' Predicted class: ' + str(pred_gnb[i])
                                    for i in range(x.shape[0])],
                              hovertemplate='%{text}')

        main_fig = make_subplots(rows=1, cols=2,
                                 subplot_titles=("LDA: accuracy = " + str(accuracy(y, pred_lda)),
                                                 "GNB: accuracy = " + str(accuracy(y, pred_gnb))))

        main_fig.append_trace(sub_fig1, row=1, col=1)
        main_fig.append_trace(sub_fig2, row=1, col=2)

        main_fig.append_trace(go.Scatter(x=lda.mu_.T[:, 0], y=lda.mu_.T[:, 1],
                                         marker=dict(symbol='x', size=14), mode='markers'), row=1, col=1)
        main_fig.append_trace(go.Scatter(x=gnb.mu_.T[:, 0], y=gnb.mu_.T[:, 1],
                                         marker=dict(symbol='x', size=14), mode='markers'), row=1, col=2)

        for class_i in range(len(lda.classes_)):
            main_fig.append_trace(get_ellipse(lda.mu_.T[class_i, :], lda.cov_), row=1, col=1)

        for class_i in range(len(gnb.classes_)):
            main_fig.append_trace(get_ellipse(gnb.mu_.T[class_i, :], np.diag(gnb.vars_[class_i])), row=1, col=2)

        main_fig.update_layout(title_text="Graph for " + str(f), showlegend=False)
        main_fig.show()
        main_fig.write_image("graph_"+str(f)+".jpeg")


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
