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
    full_data = pd.read_csv(filename).dropna().drop_duplicates()

    full_data = full_data.drop(columns=["id", "lat", "long", "date"])

    # obtaining the data is legal to the given rules and parameters
    for column in ["price", "sqft_living", "sqft_lot", "yr_built", "sqft_living15", "sqft_lot15", "sqft_above"]:
        full_data = full_data[full_data[column] > 0]

    for column in ["bathrooms", "floors", "sqft_basement", "yr_renovated"]:
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
    full_data = full_data[full_data["bedrooms"] < 18]
    full_data = full_data[full_data["sqft_lot"] < 1000000]
    full_data = full_data[full_data["sqft_lot15"] < 600000]

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

    X = X.loc[:, ~(X.columns.str.contains('^zipcode_', case=False) |
                   X.columns.str.contains('^ten_years_built_', case=False))]

    stdY = np.std(y)
    for feature_name in X:
        std_feature = np.std(X[feature_name])
        cov_featureY = np.cov(X[feature_name], y)[0, 1]
        pearson_correlation = cov_featureY / (std_feature * stdY)

        print(feature_name)
        print(pearson_correlation)

        fig = go.Figure([go.Scatter(x=X[feature_name],
                                    y=y,
                                    mode='markers')])

        fig.update_xaxes(title_text=feature_name + " values")
        fig.update_yaxes(title_text="response values")
        fig.update_layout(title_text="Pearson Correlation: " + feature_name + " and the response: " +
                                     str(pearson_correlation) + "\n")
        fig.write_image(output_path + "/" + feature_name + ".png")


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
    train_x, train_y, test_x, test_y = split_train_test(x, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()

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
    fig.write_image("loss.png")
    # fig.show()
