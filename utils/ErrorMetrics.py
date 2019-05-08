import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import *

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri, numpy2ri

# Setup to parse to R
pandas2ri.activate()

# Import R libraries
ts = ro.r('ts')
forecast = importr('forecast')
thetaModel = importr('forecTheta')
smooth = importr('smooth')

__error_metrics__ = ['MAE', 'RMSE', 'MAPE', 'sMAPE', 'MASE', 'MASE1', 'MEAN_ASE']


def calculate_all_errors(training, actual, prediction, horizon, expand_actual=True):
    """
    Calculate all error metrics used within the project.
    """
    if expand_actual:
        indexer = np.arange(horizon)[None, :] + np.arange((len(actual) - horizon))[:, None]
        actual = actual[indexer]
    prediction = prediction[:len(actual), :]
    print(actual.shape, prediction.shape)

    training, actual, prediction = training.flatten(), actual.flatten(), prediction.flatten()

    errors = {}

    # pass the data to R
    rPred = ts(prediction, frequency=1)
    rActual = ts(actual, frequency=1)

    # Calculate each error metric
    errors['MAE'] = mean_absolute_error(actual, prediction)
    errors['RMSE'] = np.sqrt(mean_squared_error(actual, prediction))
    errors['MAPE'] = thetaModel.errorMetric(obs=rActual, forec=rPred, type="APE", statistic="M")[0]
    errors['sMAPE'] = thetaModel.errorMetric(obs=rActual, forec=rPred, type="sAPE", statistic="M")[0]
    errors['MASE'] = MASE(training, actual, prediction)
    errors['MASE1'] = smooth.MASE(rActual, rPred, np.mean(training), digits=5)[0]
    errors['MEAN_ASE'] = calculate_MASE(training, prediction, actual)
    #     errors['RW_ASE']   = calculate_rw_MASE(training, prediction, actual, horizon)

    return errors


def MASE(training_series, testing_series, prediction_series):
    """
    Computes the MEAN-ABSOLUTE SCALED ERROR forcast error for univariate time series prediction.

    See "Another look at measures of forecast accuracy", Rob J Hyndman.

    MASE is the out of sample forecasting error over the in sample naive forecasting error (simply using the previous value).

    parameters:
        training_series: the series used to train the model, 1d numpy array
        testing_series: the test series to predict, 1d numpy array or float
        prediction_series: the prediction of testing_series, 1d numpy array (same size as testing_series) or float
        absolute: "squares" to use sum of squares and root the result, "absolute" to use absolute values.

    """
    # Take the average of the in sample Naive (Use the previous value) error
    naive_forecast = training_series[1:] - training_series[0:-1]
    avg_naive_error = mean_absolute_error(naive_forecast, training_series[1:])

    # Find the out of sample error of our forecast
    avg_forecast_error = mean_absolute_error(prediction_series, testing_series)

    # MASE is the forecasting error over the average in sample naive error
    return avg_forecast_error / avg_naive_error


#     n = training_series.shape[0]
#     d = np.abs(  np.diff(training_series) ).sum()/(n-1)

#     errors = np.abs(testing_series - prediction_series )
#     return errors.mean()/d

def my_MASE(training, test, pred):
    n = len(training)

    numerator = np.sum(np.abs(test - pred))
    denominator = (n / (n - 1)) * np.sum(np.abs(train[1:] - train[:-1]))
    print(numerator, denominator)

    q = numerator / denominator

    return np.mean(np.abs(q))


class _RandomWalk():

    def __init__(self, horizon):
        self.horizon = horizon

    def fit(self, train, labels):
        self.train = labels

    def predict(self, data):
        comb_data = np.vstack([self.train, data])

        results = np.zeros((data.shape[0], self.horizon))

        std = np.std(comb_data)

        results[:, :1] = data[:]

        noise = np.random.normal(0, std, (len(data), self.horizon))
        results += noise

        return results.cumsum(axis=1)

    def __repr__(self):
        return 'Random Walk'


def calculate_rw_MASE(training, test, pred, horizon):
    #     test = test.reshape((len(test), horizon))
    #     pred = pred.reshape((len(pred), horizon))

    naive_estimator = _RandomWalk(horizon)
    naive_estimator.fit(training, training)
    naive_estimate = naive_estimator.predict(test[np.arange(0, len(test), horizon)]).flatten()
    naive_estimate = naive_estimate.reshape((len(naive_estimate), horizon))

    return np.mean(np.abs(test - pred)) / np.mean(np.abs(test - naive_estimate))


"""
Mean absolute scaled error.  
"""


def calculate_MASE(trainingData: np.ndarray, prediction: np.ndarray, actual: np.ndarray) -> float:
    assert len(prediction) == len(actual)

    # Train a naive estimator as the mean estimator
    naive_estimate = np.repeat(np.mean(trainingData), len(prediction))

    # calculate errors
    naive_error = mean_absolute_error(naive_estimate, actual)
    estimator_error = mean_absolute_error(prediction, actual)

    return estimator_error / naive_error


"""
Symmetric mean absolute percentage error. 

Faults: Over penalises large positive errors more then negative errors.
"""


def calculate_sMAPE(prediction: np.ndarray, actual: np.ndarray) -> float:
    assert len(prediction) == len(actual)

    return 1 / len(prediction) * np.sum(2 * np.abs(prediction - actual) / (np.abs(actual) + np.abs(prediction)))


#     return 2 / len(prediction) * np.sum( np.abs(actual - prediction) / ( np.abs(actual) + np.abs(prediction)) )


def calculate_MAPE(pred, actual):
    return np.mean(np.abs((actual - pred) / actual))