import matplotlib.pyplot as plt

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # raise NotImplementedError()
    three_in_four = [[0, 0], [31, 31], [28, 59], [31, 90], [30, 120], [31, 151], [30, 181],
                     [31, 212], [31, 243], [30, 273], [31, 304], [30, 334], [31, 365]]

    once_in_four = [[0, 0], [31, 31], [29, 60], [31, 91], [30, 121], [31, 152], [30, 182],
                    [31, 213], [31, 244], [30, 274], [31, 305], [30, 335], [31, 366]]

    four_loop = [once_in_four, three_in_four, three_in_four, three_in_four]

    full_data = pd.read_csv(filename, parse_dates=[2]).dropna().drop_duplicates()

    into_the_year = []
    for index, sample in enumerate(full_data["Date"]):
        row = full_data.loc[[index]]
        if 1 <= sample.month <= 12 and 1 <= sample.day <= four_loop[sample.year % 4][sample.month][0]:
            if int(row["Month"]) == sample.month and int(row["Day"]) == sample.day and int(row["Year"]) == sample.year:
                into_the_year.append(sample.day_of_year)
                # alter.append(sample.day + four_loop[sample.year % 4][sample.month - 1][1])

    full_data["DayOfYear"] = into_the_year

    full_data = pd.get_dummies(full_data, prefix='Month', columns=["Month"])
    full_data = pd.get_dummies(full_data, prefix='Country', columns=["Country"])
    full_data = pd.get_dummies(full_data, prefix='City', columns=["City"])

    full_data = full_data[full_data["Temp"] < 55]
    full_data = full_data[-25 < full_data["Temp"]]

    return full_data


if __name__ == '__main__':
    np.random.seed(0)

    countries = ["South Africa", "Israel", "The Netherlands", "Jordan"]
    # Question 1 - Load and preprocessing of city temperature dataset
    # raise NotImplementedError()
    data = load_data("C:/Users/user/IML.HUJI/datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    # raise NotImplementedError()
    Israel_data = data[data["Country_Israel"] == 1]

    colors = ['brown', 'royalblue', 'gold', 'limegreen', 'darkorchid', 'deeppink',
              'violet', 'olivedrab', 'chocolate', 'black', 'darkorange',
              'lightslategray', 'crimson']
    for i, year in enumerate(range(1995, 2008)):
        data_year = Israel_data[Israel_data.Year == year]
        plt.scatter(data_year.DayOfYear, data_year.Temp, c=colors[i],
                    label=str(year))

    plt.legend(bbox_to_anchor=(0.95, 1))
    plt.xlabel("DayOfYear")
    plt.ylabel("Temp")
    plt.title("Temperature in Israel as for day of the year")
    plt.show()
    # plt.savefig("ex2_2_2")

    Israel_temp_std_month = []
    for i in range(1, 13):
        std_i = Israel_data.groupby("Month_" + str(i)).Temp.std()[1]
        Israel_temp_std_month.append(std_i)

    plt.bar(range(1, 13), Israel_temp_std_month)
    plt.xlabel("month")
    plt.ylabel("std")
    plt.title("std in Israel as for month")
    plt.show()

    # Question 3 - Exploring differences between countries
    # raise NotImplementedError()
    stds = [[], [], [], []]
    means = [[], [], [], []]
    for j, country in enumerate(countries):
        for i in range(1, 13):
            month_i = data[data["Month_" + str(i)] == 1]
            filtered = month_i[month_i["Country_" + country] == 1]

            stds[j].append(filtered.Temp.std())
            means[j].append(filtered.Temp.mean())

        plt.errorbar(range(1, 13), means[j], yerr=stds[j], label=country)

    plt.legend()
    plt.xlabel("Month")
    plt.ylabel("avg Temp.")
    plt.title("avg Temp. as for each month with std as error bars")
    plt.show()
    # Question 4 - Fitting model for different values of `k`
    # raise NotImplementedError()

    y_label = "Temp"
    x = Israel_data.drop(columns=[y_label])
    y = Israel_data[y_label]
    train_x, train_y, test_x, test_y = split_train_test(x, y, 0.75)

    train_x = train_x.DayOfYear
    test_x = test_x.DayOfYear

    losses = []
    for k in range(1, 11):
        poly = PolynomialFitting(k)
        poly.fit(train_x.to_numpy(), train_y.to_numpy())
        loss_k = poly.loss(test_x.to_numpy(), test_y.to_numpy())
        loss_k = round(loss_k, 2)
        print(loss_k)
        losses.append(loss_k)

    plt.bar(range(1, 11), losses)
    plt.xlabel("k")
    plt.ylabel("loss")
    plt.title("loss as for k")
    plt.show()
# Question 5 - Evaluating fitted model on different countries
    # raise NotImplementedError()
    best_k = min(losses)

    poly_best_k = PolynomialFitting(int(best_k))
    poly_best_k.fit(Israel_data.DayOfYear, Israel_data.Temp)

    errors = []
    for country in countries:
        country_j = data[data["Country_" + country] == 1]
        errors.append(poly_best_k.loss(country_j.DayOfYear, country_j.Temp))

    plt.bar(countries, errors)
    plt.xlabel("country")
    plt.ylabel("error")
    plt.title("error in country as for Israel fit")
    plt.show()
