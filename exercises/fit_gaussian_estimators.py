import matplotlib.pyplot as plt
import seaborn as sns

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    # raise NotImplementedError()
    uni = UnivariateGaussian()

    samples = np.random.normal(10, 1, 1000)
    uni.fit(samples)

    est_mean = uni.mu_
    est_var = uni.var_

    print("(" + str(est_mean) + ", " + str(est_var) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    # raise NotImplementedError()
    distances = []
    Xs = np.arange(10, 1001, 10)

    for x in Xs:
        distances.append(np.abs(est_mean - np.mean(samples[0:int(x)])))

    plt.plot(Xs, distances)
    plt.title("Q2 - distance from estimated mean")
    plt.xlabel("number of samples")
    plt.ylabel("distance")
    plt.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    # raise NotImplementedError()
    samples.sort()
    plt.plot(samples, uni.pdf(samples))
    plt.title("Q3 - estimated PDF")
    plt.xlabel("sorted samples")
    plt.ylabel("PDF")
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    # raise NotImplementedError()
    multi = MultivariateGaussian()

    mu = np.zeros(4)
    mu[2] = 4
    mu = np.transpose(mu)

    cov = np.zeros((4, 4))
    cov[0, 0] = 1
    cov[0, 1] = 0.2
    cov[1, 0] = 0.2
    cov[1, 1] = 2
    cov[3, 0] = 0.5
    cov[0, 3] = 0.5
    cov[2, 2] = 1
    cov[3, 3] = 1

    samples = np.random.multivariate_normal(mu, cov, 1000)

    multi.fit(samples)
    print(multi.mu_)
    print(multi.cov_)

    # Question 5 - Likelihood evaluation
    # raise NotImplementedError()
    F1s = np.linspace(-10, 10, 200)
    F3s = np.linspace(-10, 10, 200)

    comb_and_log = np.zeros((200, 200))
    optional_mu = np.zeros(4)

    max_val = None
    max_row = 0
    max_col = 0

    for row, f1 in enumerate(F1s):
        optional_mu[0] = f1
        for col, f3 in enumerate(F3s):
            optional_mu[2] = f3
            comb_and_log[row, col] = multi.log_likelihood(optional_mu, cov, samples)
            if max_val is None or max_val < comb_and_log[row, col]:
                max_val = comb_and_log[row, col]
                max_row = f1
                max_col = f3

    go.Figure(go.Heatmap(x=F1s, y=F3s, z=comb_and_log), layout=go.Layout(title="Q5 - log likelihood - heatmap",
                                                                  xaxis_title="f1 indexes",
                                                                  yaxis_title="f3 indexes")).show()

    # Question 6 - Maximum likelihood

    print(max_row, max_col)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()