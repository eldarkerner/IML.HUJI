import IMLearn.utils
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
import matplotlib.pyplot as plt

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # raise NotImplementedError()
    y_label = "price"
    full_data = pd.read_csv(filename).drop_duplicates()

    full_data = full_data.drop(columns=["id", "lat", "long", "date"])

    # obtaining the data is legal to the given rules and parameters
    for column in ["price", "sqft_living", "sqft_lot", "yr_built", "sqft_living15", "sqft_lot15"]:
        full_data = full_data[full_data[column] > 0]

    for column in ["bathrooms", "floors", "sqft_basement", "yr_renovated", "sqft_above"]:
        full_data = full_data[full_data[column] >= 0]

    full_data = full_data[full_data["waterfront"].isin([0, 1]) &
                          full_data["view"].isin(range(5)) &
                          full_data["condition"].isin(range(1, 6)) &
                          full_data["grade"].isin(range(1, 14))]

    # processing the data
    full_data["ten_years_built"] = (full_data["yr_built"] / 10).astype(int)

    threshold = np.percentile(full_data.yr_renovated.unique(), 70)
    full_data["recently_renovated"] = np.where(full_data["yr_renovated"] >= threshold, 1, 0)

    full_data = full_data.drop(columns=["yr_built", "yr_renovated"])

    # as explained in the Stackoverflow question given to us
    full_data = pd.get_dummies(full_data, prefix='zipcode_', columns=['zipcode'])
    full_data = pd.get_dummies(full_data, prefix='ten_years_built_', columns=['ten_years_built'])

    # removing extreme cases so won't extra affect the model
    full_data = full_data[full_data["bedrooms"] < 20]
    full_data = full_data[full_data["sqft_lot"] < 1250000]
    full_data = full_data[full_data["sqft_lot15"] < 500000]

    x = full_data.drop(columns=[y_label])
    y = full_data[y_label]

    return x, y, full_data


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # raise NotImplementedError()

    # varY = np.std(y)
    #
    # choosen_features = ["sqft_living", "lat"]
    #
    # for i, feature_name in enumerate(choosen_features):
    #     feature = X.loc[:, feature_name]
    #
    #     var_feature = np.std(feature)
    #     cov_featureY = np.cov(feature, y)
    #
    #     pearson_correlation = cov_featureY / (var_feature * varY)
    #
    #     print(feature.shape)
    #     print(pearson_correlation)
    #     print(varY, var_feature)
    #     print(cov_featureY)
    #
    #     plt.plot(feature, pearson_correlation)
    #     plt.title("Pearson Correlation between " + feature_name + " and the response")
    #     plt.savefig(output_path + feature_name)

    X = X.loc[:, ~(X.columns.str.contains('^zipcode_', case=False) |
                   X.columns.str.contains('^decade_built_', case=False))]

    stdY = np.std(y)
    for feature_name in X:
        std_feature = np.std(X[feature_name])
        cov_featureY = np.cov(X[feature_name], y)[0, 1]
        pearson_correlation = cov_featureY / (std_feature * stdY)

        print(feature_name)
        print(pearson_correlation)

        plt.scatter(X[feature_name], y)
        plt.title("Pearson Correlation between " + feature_name + " and the response: " +
                  str(pearson_correlation) + "\n")
        plt.savefig(output_path + "/" + feature_name)


def q4():
    fraction = np.linspace(0.1, 1, 91)
    lr = LinearRegression()
    con = np.zeros(91)
    mean_loss = np.zeros(91)

    for j in range(len(fraction)):
        loss = np.zeros(10)
        for i in range(10):
            train_x["price"] = train_y
            sampled_train = train_x.sample(frac=fraction[j])
            # print(train_x)
            # print(sampled_train)
            lr.fit(sampled_train.drop("price", axis=1), sampled_train.price)
            loss[i] = lr.loss(test_x.to_numpy(), test_y.to_numpy())
        con[j] = 2 * np.std(loss)
        mean_loss[j] = np.mean(loss)

    fig = go.Figure(data=go.Scatter(x=100 * fraction, y=mean_loss,
                                    error_y=dict(type='data', array=con, visible=True)),
                    layout=go.Layout(title="AAAA"))

    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    # raise NotImplementedError()
    x, y, data = load_data("C:/Users/user/IML.HUJI/datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # raise NotImplementedError()
    feature_evaluation(x, y, "C:/Users/user/IML.HUJI/exercises/ex2_images")

    # Question 3 - Split samples into training- and testing sets.
    # raise NotImplementedError()
    train_x, train_y, test_x, test_y = IMLearn.utils.split_train_test(x, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
    # q4()
    linear_regression = LinearRegression()

    x_range = []
    loss_mean = []
    loss_std = []

    train_x["price"] = train_y
    for i in range(10, 101):
        loss_i = []
        for r in range(1, 11):
            sampled_train = train_x.sample(frac=(i / 100.0))
            linear_regression.fit(sampled_train.drop(columns=["price"]), sampled_train.price)
            loss_i.append(linear_regression.loss(test_x.to_numpy(), test_y.to_numpy()))

        x_range.append(i)
        loss_mean.append(np.mean(loss_i))
        loss_std.append(np.std(loss_i))

    loss_mean = np.asarray(loss_mean)
    loss_std = np.asarray(loss_std)

    fig = go.Figure([go.Scatter(x=x_range,
                                y=loss_mean,
                                mode='lines'),
                     go.Scatter(x=x_range,
                                y=loss_mean + 2 * loss_std,
                                mode='lines', marker=dict(color='#444'), showlegend=False),
                     go.Scatter(x=x_range,
                                y=loss_mean - 2 * loss_std,
                                mode='lines', marker=dict(color='#444'), showlegend=False,
                                fillcolor='rgba(68,68,68,0.3)', fill='tonexty')])

    fig.update_xaxes(ticksuffix="%", title_text="percents of training-set")
    fig.update_yaxes(title_text="loss over test-set")
    fig.update_layout(title_text="average loss as function of training size with error ribbon")
    # fig.write_image("loss.png")
    fig.show()
    # print(x)
    #
    # plt.scatter(x, loss_mean)
    # plt.scatter(x, loss_mean + 2 * loss_std)
    # plt.scatter(x, loss_mean - 2 * loss_std)
    # #
    # plt.title("average loss scaled to the training data size along with potential error")
    # plt.xlabel("% of the training")
    # plt.ylabel("average loss")
    #
    # plt.savefig("average_loss")

    # linear_regression = LinearRegression()
    # loss_mean, loss_var = [], []
    #
    # for p in range(10, 101):
    #     n = round(len(trainY) * (p / 100))
    #
    #     loss_samples = []
    #     for _ in range(10):
    #         sampled = train_df.sample(n)
    #         linear_regression.fit(sampled.drop("price", axis=1), sampled["price"])
    #         loss_samples.append(linear_regression.loss(testX, testY))
    #         loss_samples = np.asarray(loss_samples)
    #
    #     loss_mean.append(loss_samples.mean())
    #     loss_var.append(loss_samples.var())
    #
    #     loss_var = np.asarray(loss_var)
    #     loss_mean = np.asarray(loss_mean)
    # x_range = np.linspace(10, 100, 91)
    #
    # fig = go.Figure([go.Scatter(x=x_range,
    #                             y=loss_mean,
    #                             mode='lines'),
    #                  go.Scatter(x=x_range,
    #                             y=loss_mean + 2 * loss_var,
    #                             mode='lines', marker=dict(color='#444'), showlegend=False),
    #                  go.Scatter(x=x_range,
    #                             y=loss_mean - 2 * loss_var,
    #                             mode='lines', marker=dict(color='#444'), showlegend=False,
    #                             fillcolor='rgba(68,68,68,0.3)', fill='tonexty')])
    #
    # fig.update_xaxes(ticksuffix="%", title_text="percents of training-set")
    # fig.update_yaxes(title_text="loss over test-set")
    # fig.update_layout(title_text="average loss as function of training size with error ribbon")
    # fig.write_image("loss.png")
    # print("done")


