import sklearn.model_selection

from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
from IMLearn.base import BaseEstimator

import numpy as np
import pandas as pd


def complete_row(full_data, row_name, zero_change=True):
    nonempty_row_count = 0
    avg_row = 0
    for row in full_data[row_name]:
        if not pd.isna(row):
            nonempty_row_count += 1
            if zero_change and row == 0:
                avg_row += -1
            else:
                avg_row += row

    pred = avg_row / float(nonempty_row_count)

    for index, row in enumerate(full_data[row_name]):
        if pd.isna(row):
            full_data.at[index, row_name] = pred
        elif zero_change and full_data.at[index, row_name] == 0:
            full_data.at[index, row_name] = -1

    return full_data


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing

    subparams = ["no_of_adults", "no_of_children", "no_of_extra_bed", "no_of_room", "hotel_star_rating",
                 "original_selling_amount", "request_nonesmoke", "request_latecheckin", "request_highfloor",
                 "request_largebed", "request_twinbeds", "request_airport", "request_earlycheckin"]

    need_to_be_completed = ["request_nonesmoke", "request_latecheckin", "request_highfloor",
                 "request_largebed", "request_twinbeds", "request_airport", "request_earlycheckin"]

    full_data = pd.read_csv(filename)

    for row_name in need_to_be_completed:
        full_data = complete_row(full_data, row_name)

    full_data = full_data.dropna(subset=subparams)

    features = full_data[subparams]
    labels = full_data["cancellation_datetime"]

    modified_labels = np.zeros(len(labels))
    for index, label in enumerate(labels):
        if not pd.isna(label):
            modified_labels[index] = 1

    return features, modified_labels


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """

    predictions = estimator.predict(X)

    # modified_predictions = np.zeros(len(predictions))

    # for index, prediction in enumerate(predictions):
    #     if prediction == 1:
    #         modified_predictions[index] = 0
    #     else:
    #         modified_predictions[index] = 1

    pd.DataFrame(predictions, columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
    # print(cancellation_labels)
    train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(df, cancellation_labels)

    # Fit model over data
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    estimator = AgodaCancellationEstimator().fit(train_X, train_y)

    # Store model predictions over test set

    modified_y = np.zeros(len(test_y))
    for index, label in enumerate(test_y):
        if not pd.isna(label):
            modified_y[index] = 1

    print(estimator.loss(test_X, modified_y))
    evaluate_and_export(estimator, test_X, "logistic.csv")
